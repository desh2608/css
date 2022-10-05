#! /usr/bin/env python
# Copyright 2021  Johns Hopkins University (Author: Desh Raj)
# Apache 2.0

import os
import argparse
import json
import random
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist

from tensorboardX import SummaryWriter

import css.models as models
import css.objectives as objectives
from css.datasets.feature_separation_dataset import (
    FeatureSeparationDataset,
    feature_collater,
)

from css.trainer import LRScheduler, train_one_epoch, validate

from itertools import chain

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


parser = argparse.ArgumentParser()
parser.add_argument("--expdir", type=str, help="Experiment directory")
parser.add_argument("--config", type=str, help="Configuration file (YAML)")
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--resume", default=None)
parser.add_argument("--init", default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gpu", action="store_true", default=False)
parser.add_argument("--num-epochs", type=int, default=10)
parser.add_argument("--world-size", type=int, default=1)
parser.add_argument("--master-port", type=int, default=12354)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def setup_dist(rank, world_size, master_port=None):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12354" if master_port is None else str(master_port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_dist():
    dist.destroy_process_group()


def main(rank, world_size, conf):

    # Set the random seed
    torch.manual_seed(conf["seed"])
    random.seed(conf["seed"])
    np.random.seed(conf["seed"])

    if world_size > 1:
        setup_dist(rank, world_size, conf["master_port"])

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)

    # Create the conf file if it doesn't exist
    if not os.path.exists("{}/conf.json".format(conf["expdir"])):
        json.dump(
            conf,
            open("{}/conf.json".format(conf["expdir"]), "w"),
            indent=4,
            separators=(",", ": "),
        )
        conf["epoch"] = 0

    # Resume training with same configuration / dump training configurations
    if conf["resume"] is not None:
        print("Loading former training configurations ...")
        old_conf = json.load(open("{}/conf.json".format(conf["expdir"])))
        old_conf.update(conf)
        conf = old_conf
    else:
        # Dump training configurations when resuming but the conf file already
        # existing, i.e. from a previous training run.
        json.dump(
            conf,
            open("{}/conf.json".format(conf["expdir"]), "w"),
            indent=4,
            separators=(",", ": "),
        )
        conf["epoch"] = 0

    train_set = FeatureSeparationDataset(conf["data"]["train_dir"])
    val_set = FeatureSeparationDataset(conf["data"]["valid_dir"])
    collate_fn = feature_collater

    # Create distributed sampler pinned to rank
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=conf["seed"]
    )
    val_sampler = DistributedSampler(
        val_set, num_replicas=world_size, rank=rank, shuffle=False
    )

    # Prepare dataloaders
    train_dataloader = DataLoader(
        train_set,
        batch_size=conf["dataloader"]["batch_size"],
        num_workers=conf["dataloader"]["num_workers"],
        sampler=train_sampler,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_set,
        batch_size=conf["dataloader"]["batch_size"],
        num_workers=conf["dataloader"]["num_workers"],
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn,
    )

    # Define the model
    if rank == 0:
        logging.info("Defining model ...")
    conf["model"]["idim"] = conf["feature"]["idim"]
    conf["model"]["num_bins"] = conf["feature"]["num_bins"]
    model = models.MODELS[conf["model"].pop("model_type")].build_model(conf["model"])
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        logging.info(f"Traning model with {total_params} parameters.")

    # Define the objective
    objective = objectives.OBJECTIVES[
        conf["objective"]["objective_type"]
    ].build_objective(conf["objective"])

    if conf["resume"] is not None:
        if rank == 0:
            logging.info("Resuming ...")
        # Loads state dict
        mdl = torch.load(
            os.path.sep.join([conf["expdir"], conf["resume"] + ".mdl"]),
            map_location="cpu",
        )

        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in mdl["model"].items():
            name = k.replace("module.", "")  # remove 'module.' of dataparallel
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

        objective.load_state_dict(mdl["objective"])

    # Send model, feature extractor, and objective function to GPU (or keep on CPU)
    model.to(device)
    objective.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    # Define trainable parameters
    params = list(
        filter(
            lambda p: p.requires_grad,
            chain(model.parameters(), objective.parameters()),
        )
    )

    # Define optimizer over trainable parameters and a learning rate schedule
    if conf["trainer"]["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=conf["trainer"]["lr"],
            weight_decay=conf["trainer"].get("weight_decay", 0),
        )
    elif conf["trainer"]["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=conf["trainer"]["lr"],
            momentum=conf["trainer"].get("momentum", 0),
        )

    # Check if training is resuming from a previous epoch
    if conf["resume"] is not None:
        optimizer.load_state_dict(mdl["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        lr_sched = LRScheduler(optimizer, conf["trainer"])
        lr_sched.load_state_dict(mdl["lr_sched"])
        conf["epoch"] = mdl["epoch"]
        conf["global_step"] = mdl["global_step"]

    else:
        lr_sched = LRScheduler(optimizer, conf["trainer"])

    # Initializing with a pretrained model
    if conf["init"] is not None:
        mdl = torch.load(conf["init"], map_location=device)
        for name, p in model.named_parameters():
            if name in mdl["model"]:
                p.data.copy_(mdl["model"][name].data)

    # Initialize tensorboard writer (only for the first job)
    if rank == 0:
        (Path(conf["expdir"]) / conf["tensorboard"]["log_dir"]).mkdir(
            parents=True, exist_ok=True
        )
        writer = SummaryWriter(
            os.path.join(conf["expdir"], conf["tensorboard"]["log_dir"])
        )
    else:
        writer = None

    # train
    train(
        conf,
        train_dataloader,
        val_dataloader,
        model,
        objective,
        optimizer,
        lr_sched,
        device,
        writer,
    )

    print("Done.")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def train(
    conf,
    train_dataloader,
    val_dataloader,
    model,
    objective,
    optimizer,
    lr_sched,
    device,
    writer,
):
    """
    Runs the training defined in args and conf, on the datasets, using
    the model, objective, optimizer and lr_scheduler. Performs training
    on the specified device.
    """
    global_step = conf.get("global_step", 0)
    for e in range(conf["epoch"], conf["epoch"] + conf["num_epochs"]):
        # Important for DDP training
        train_dataloader.sampler.set_epoch(e)

        avg_loss, global_step = train_one_epoch(
            conf,
            train_dataloader,
            model,
            objective,
            optimizer,
            lr_sched,
            device,
            writer,
            global_step,
        )

        # Run validation set
        if val_dataloader is not None:
            avg_loss_val = validate(
                val_dataloader,
                model,
                objective,
                device,
                writer,
                conf,
                global_step=global_step,
            )
            if writer is not None:
                writer.add_scalar("loss/valid", avg_loss_val, global_step=e)
                logging.info(
                    f"Epoch: {e + 1} :: Avg train loss = {avg_loss}, avg. valid loss = {avg_loss_val}"
                )
        else:
            if writer is not None:
                logging.info(f"Epoch: {e + 1} :: Avg train loss = {avg_loss}")

        # Save checkpoint
        state_dict = {
            "model": model.state_dict(),
            "objective": objective.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_sched": lr_sched.state_dict(),
            "epoch": e + 1,
            "global_step": global_step,
            "loss": avg_loss,
        }

        if not np.isnan(avg_loss):
            torch.save(
                state_dict,
                conf["expdir"] + f"/{e+1}.mdl",
            )


if __name__ == "__main__":
    import yaml
    from pprint import pprint
    from css.utils.parser_util import prepare_parser_from_dict, parse_args_as_dict

    # Read raw config file. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    raw_args = parser.parse_args()
    with open(raw_args.config) as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse).
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    arg_dic.update(arg_dic.pop("main_args"))
    pprint(arg_dic)
    world_size = raw_args.world_size
    if world_size > 1:
        mp.spawn(main, args=(world_size, arg_dic), nprocs=world_size, join=True)
    else:
        main(rank=0, world_size=1, args=arg_dic)
