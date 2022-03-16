#! /usr/bin/env python
# Copyright 2021  Johns Hopkins University (Author: Desh Raj)
# Apache 2.0

import os
import argparse
import json
import random
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import css.models as models
import css.objectives as objectives
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
parser.add_argument("--job", type=int, default=1)
parser.add_argument("--nj", type=int, default=1)
parser.add_argument("--gpu", action="store_true", default=False)
parser.add_argument("--num-epochs", type=int, default=10)


def main(conf):

    device = torch.device("cuda" if conf["gpu"] else "cpu")
    _ = torch.ones(1).to(device)

    # Set the random seed
    torch.manual_seed(conf["seed"])
    random.seed(conf["seed"])
    np.random.seed(conf["seed"])

    # Create the conf file if it doesn't exist
    if not os.path.exists("{}/conf.{}.json".format(conf["expdir"], conf["job"])):
        json.dump(
            conf,
            open("{}/conf.{}.json".format(conf["expdir"], conf["job"]), "w"),
            indent=4,
            separators=(",", ": "),
        )
        conf["epoch"] = 0

    # Resume training with same configuration / dump training configurations
    if conf["resume"] is not None:
        print("Loading former training configurations ...")
        old_conf = json.load(
            open("{}/conf.{}.json".format(conf["expdir"], conf["job"]))
        )
        old_conf.update(conf)
        conf = old_conf
    else:
        # Dump training configurations when resuming but the conf file already
        # existing, i.e. from a previous training run.
        json.dump(
            conf,
            open("{}/conf.{}.json".format(conf["expdir"], conf["job"]), "w"),
            indent=4,
            separators=(",", ": "),
        )
        conf["epoch"] = 0

    if conf["data"]["feature"] == "on_the_fly":
        from css.executor.feature import FeatureExtractor
        from css.datasets.raw_waveform_separation_dataset import (
            RawWaveformSeparationDataset,
            raw_waveform_collater,
        )

        # Feature extractor
        feat = FeatureExtractor(
            frame_len=conf["feature"]["frame_length"],
            frame_hop=conf["feature"]["frame_shift"],
            ipd_index=conf["feature"]["ipd"],
        )
        collate_fn = raw_waveform_collater

        # Load the dataset. For the training data, we only use the part of data relevant for
        # the current split, i.e., if we are training with 4 jobs, each of them will only use
        # 25% of the data.
        train_set = RawWaveformSeparationDataset(
            conf["data"]["train_json"], job=conf["job"], nj=conf["nj"]
        )
        val_set = RawWaveformSeparationDataset(conf["data"]["valid_json"])

    elif conf["data"]["feature"] == "precomputed":
        from css.datasets.feature_separation_dataset import (
            FeatureSeparationDataset,
            feature_collater,
        )

        feat = None
        # Load the dataset. For the training data, we only use the part of data relevant for
        # the current split, i.e., if we are training with 4 jobs, each of them will only use
        # 25% of the data.
        train_set = FeatureSeparationDataset(
            conf["data"]["train_dir"], job=conf["job"], nj=conf["nj"]
        )
        val_set = FeatureSeparationDataset(conf["data"]["valid_dir"])
        collate_fn = feature_collater

    # Prepare dataloaders
    train_dataloader = DataLoader(
        train_set,
        batch_size=conf["dataloader"]["batch_size"],
        num_workers=conf["dataloader"]["num_workers"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        val_set,
        batch_size=conf["dataloader"]["batch_size"],
        num_workers=conf["dataloader"]["num_workers"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Define the model
    logging.info("Defining model ...")
    conf["model"]["idim"] = conf["feature"]["idim"]
    conf["model"]["num_bins"] = conf["feature"]["num_bins"]
    model = models.MODELS[conf["model"].pop("model_type")].build_model(conf["model"])
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Traning model with {total_params} parameters.")

    # Define the objective
    objective = objectives.OBJECTIVES[
        conf["objective"]["objective_type"]
    ].build_objective(conf["objective"])

    if conf["resume"] is not None:
        logging.info("Resuming ...")
        # Loads state dict
        mdl = torch.load(
            os.path.sep.join([conf["expdir"], conf["resume"]]), map_location="cpu"
        )
        model.load_state_dict(mdl["model"])
        objective.load_state_dict(mdl["objective"])

    # Send model, feature extractor, and objective function to GPU (or keep on CPU)
    model.to(device)
    if feat is not None:
        feat.to(device)
    objective.to(device)

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
    if not os.path.exists(os.path.join(conf["expdir"], conf["tensorboard"]["log_dir"])):
        os.mkdir(os.path.join(conf["expdir"], conf["tensorboard"]["log_dir"]))
    writer = (
        SummaryWriter(os.path.join(conf["expdir"], conf["tensorboard"]["log_dir"]))
        if conf["job"] == 1
        else None
    )

    # train
    train(
        conf,
        train_dataloader,
        val_dataloader,
        model,
        feat,
        objective,
        optimizer,
        lr_sched,
        device,
        writer,
    )

    print("Done.")


def train(
    conf,
    train_dataloader,
    val_dataloader,
    model,
    feat,
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

        avg_loss, global_step = train_one_epoch(
            conf,
            train_dataloader,
            model,
            feat,
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
                feat,
                objective,
                device,
            )
            logging.info(
                f"Epoch: {e + 1} :: Avg train loss = {avg_loss}, avg. valid loss = {avg_loss_val}"
            )
            writer.add_scalar("loss/valid", avg_loss_val, global_step=e)
        else:
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
                conf["expdir"] + "/{}.{}.mdl".format(e + 1, conf["job"]),
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
    main(arg_dic)
