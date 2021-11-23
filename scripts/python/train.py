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

import css.models as models
import css.objectives as objectives
import css.datasets as datasets
from css.trainer import LRScheduler, train_one_epoch, validate

from itertools import chain

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Dataset and model related config
    parser.add_argument(
        "--train-manifests",
        type=str,
        required=False,
        nargs="+",
        help="Lhotse CutSet manifests for training data. Multiple manifests can be provided.",
    )
    parser.add_argument(
        "--dev-manifests",
        type=str,
        required=False,
        nargs="+",
        help="Lhotse CutSet manifests for validation data",
    )
    parser.add_argument(
        "--rir-manifest",
        type=str,
        required=False,
        help="Lhotse recording manifest for real or simulated RIRs used for reverberation",
    )
    parser.add_argument(
        "--noise-manifest",
        type=str,
        required=False,
        help="Lhotse CutSet manifest for isotropic noises",
    )
    parser.add_argument("--expdir", type=str, help="Experiment directory")
    parser.add_argument(
        "--model",
        default="Conformer",
        choices=["Conformer"],
    )
    parser.add_argument(
        "--objective",
        default="MSE",
        choices=["MSE"],
    )
    parser.add_argument(
        "--dataset",
        default="CSS",
        choices=["CSS"],
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    # Optimizer related config
    parser.add_argument("--grad-thresh", type=float, default=30.0)
    parser.add_argument("--optim", default="sgd")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-08)
    # Training related config
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--job", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batches-per-epoch", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=0)

    # Args specific to different components. See model,LRScheduler,dataset}.py.
    args, leftover = parser.parse_known_args()
    models.MODELS[args.model].add_args(parser)
    datasets.DATASETS[args.dataset].add_args(parser)
    objectives.OBJECTIVES[args.objective].add_args(parser)
    LRScheduler.add_args(parser)
    parser.parse_args(leftover, namespace=args)
    return args


def main(args):

    device = torch.device("cuda" if args.gpu else "cpu")
    _ = torch.ones(1).to(device)

    # Set the random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create the conf file if it doesn't exist
    if not os.path.exists("{}/conf.{}.json".format(args.expdir, args.job)):
        json.dump(
            vars(args),
            open("{}/conf.{}.json".format(args.expdir, args.job), "w"),
            indent=4,
            separators=(",", ": "),
        )
        conf = vars(args)
        conf["epoch"] = 0

    # Resume training with same configuration / dump training configurations
    if args.resume is not None:
        logging.info("Loading former training configurations ...")
        conf = json.load(open("{}/conf.{}.json".format(args.expdir, args.job)))
    else:
        # Dump training configurations when resuming but the conf file already
        # existing, i.e. from a previous training run.
        json.dump(
            vars(args),
            open("{}/conf.{}.json".format(args.expdir, args.job), "w"),
            indent=4,
            separators=(",", ": "),
        )
        conf = vars(args)
        conf["epoch"] = 0

    # Prepare dataloader for training sets
    logging.info("Defining dataset object ...")
    train_datasets = []
    for ds in conf["train_manifests"]:
        train_datasets.append(
            datasets.DATASETS[args.dataset].build_dataset(ds, conf),
        )
    train_datasets = torch.utils.data.ChainDataset(train_datasets)
    train_dataloader = torch.utils.data.DataLoader(
        train_datasets,
        num_workers=args.num_workers,
        worker_init_fn=lambda x: random.seed(x + args.seed),
    )

    # Prepare dataloader for validation sets if provided
    if args.dev_manifests is not None:
        valid_datasets = []
        for ds in conf["dev_manifests"]:
            valid_datasets.append(
                datasets.DATASETS[args.dataset].build_dataset(ds, conf)
            )
        valid_datasets = torch.utils.data.ChainDataset(valid_datasets)
        valid_dataloader = torch.utils.data.DataLoader(
            valid_datasets,
            num_workers=args.num_workers,
            worker_init_fn=lambda x: random.seed(x + args.seed),
        )
    else:
        valid_dataloader = None

    # Define model
    logging.info("Defining model ...")
    model = models.MODELS[args.model].build_model(conf)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Traning model with {total_params} parameters.")
    logging.info(model)
    objective = objectives.OBJECTIVES[args.objective].build_objective(conf)

    if args.resume is not None:
        logging.info("Resuming ...")
        # Loads state dict
        mdl = torch.load(
            os.path.sep.join([args.expdir, args.resume]), map_location="cpu"
        )
        model.load_state_dict(mdl["model"])
        objective.load_state_dict(mdl["objective"])

    # Send model and objective function to GPU (or keep on CPU)
    model.to(device)
    objective.to(device)

    # Define trainable parameters
    params = list(
        filter(
            lambda p: p.requires_grad,
            chain(model.parameters(), objective.parameters()),
        )
    )

    # Define optimizer over trainable parameters and a learning rate schedule
    optimizers = {
        "sgd": torch.optim.SGD(params, lr=conf["lr"], momentum=0.0),
        "adam": torch.optim.Adam(
            params, lr=conf["lr"], weight_decay=conf["weight_decay"]
        ),
    }

    optimizer = optimizers[conf["optim"]]

    # Check if training is resuming from a previous epoch
    if args.resume is not None:
        optimizer.load_state_dict(mdl["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        lr_sched = LRScheduler(optimizer, args)
        lr_sched.load_state_dict(mdl["lr_sched"])
        conf["epoch"] = mdl["epoch"]

    else:
        lr_sched = LRScheduler(optimizer, args)

    # Initializing with a pretrained model
    if args.init is not None:
        mdl = torch.load(args.init, map_location=device)
        for name, p in model.named_parameters():
            # if 'xent_layer' not in name and 'linear' not in name:
            if args.replace_output:
                if not any(
                    [x in name for x in ["xent_layer", "linear", "final_affine"]]
                ):
                    if name in mdl["model"]:
                        p.data.copy_(mdl["model"][name].data)
            else:
                if name in mdl["model"]:
                    p.data.copy_(mdl["model"][name].data)

    # train
    train(
        args,
        conf,
        train_dataloader,
        model,
        objective,
        optimizer,
        lr_sched,
        device,
        valid_dataloader=valid_dataloader,
    )

    print("Done.")


def train(
    args,
    conf,
    train_dataloader,
    model,
    objective,
    optimizer,
    lr_sched,
    device,
    valid_dataloader=None,
):
    """
    Runs the training defined in args and conf, on the datasets, using
    the model, objective, optimizer and lr_scheduler. Performs training
    on the specified device.
    """
    for e in range(conf["epoch"], conf["epoch"] + args.num_epochs):

        avg_loss = train_one_epoch(
            args, train_dataloader, model, objective, optimizer, lr_sched, device=device
        )

        # Run validation set
        if valid_dataloader is not None:
            avg_loss_val = validate(
                valid_dataloader,
                model,
                objective,
                device=device,
            )
            logging.info(
                f"Epoch: {e + 1} :: Avg train loss = {avg_loss}, avg. valid loss = {avg_loss_val}"
            )
        else:
            logging.info(f"Epoch: {e + 1} :: Avg train loss = {avg_loss}")

        # Save checkpoint
        state_dict = {
            "model": model.state_dict(),
            "objective": objective.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_sched": lr_sched.state_dict(),
            "epoch": e + 1,
            "loss": avg_loss,
        }

        if not np.isnan(avg_loss):
            torch.save(
                state_dict,
                args.expdir + "/{}.{}.mdl".format(e + 1, args.job),
            )


if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    main(args)
