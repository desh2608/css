#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2020  Matthew Wiesner
#           2021  Desh Raj
# Apache 2.0

import logging
import random
import torch

from css.utils.tensorboard_utils import make_grid_from_tensors

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def train_one_epoch(
    conf,
    dataloader,
    model,
    objective,
    optim,
    lr_sched,
    device,
    writer,
    global_step,
):
    """
    The training interator: It takes
         - a data loader
         - a model
         - a training objective
         - an optimizer
         - a learning rate scheduler

    It defines how to use these components to perform 1 epoch of training.
    """
    total_loss = 0.0
    num_batches = len(dataloader)

    for i, b in enumerate(dataloader):
        global_step += 1
        log = f"Iter: {i} of {num_batches} LR:{lr_sched.curr_lr:0.5e} "

        batch = {
            "mix": b["mix"],  # B x T x F
            "targets": torch.stack(
                [b[src] for src in ["src0", "src1", "noise"]], dim=1
            ),  # B x 3 x T x F
            "feats": b["feats"],  # B x T x *
            "len": b["len"],
        }

        loss = objective(model, batch, device=device)

        log += f"Loss: {loss.data.item():0.5f} "
        total_loss += loss.data.item()

        loss.backward()
        loss.detach()
        del b

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), conf["trainer"]["grad_thresh"]
        )
        log += f"Grad_norm: {grad_norm.data.item():0.5f}"

        if writer is not None:
            writer.add_scalar("loss/train", loss.data.item(), global_step=global_step)
            logging.info(log)
        optim.step()
        optim.zero_grad()
        lr_sched.step(1.0)
    return total_loss / num_batches, global_step


def validate(dataloader, model, objective, device, writer, conf, global_step):
    model.eval()
    with torch.no_grad():
        avg_loss = 0.0
        num_batches = len(dataloader)
        # We will plot spectrogram for a randomly selected batch (since first batch always
        # has just 1 input in the mixture)
        rand_batch = random.randint(0, num_batches - 1)
        for i, b in enumerate(dataloader):
            batch = {
                "mix": b["mix"].to(device),  # B x T x F
                "targets": torch.stack(
                    [b[src].to(device) for src in ["src0", "src1", "noise"]], dim=1
                ),  # B x 3 x T x F
                "feats": b["feats"].to(device),  # B x T x *
                "len": b["len"].to(device),
            }
            loss, tensors = objective(model, batch, device=device, return_est=True)
            if i == rand_batch and writer is not None:
                grids = make_grid_from_tensors(
                    tensors, num_samples=conf["tensorboard"]["num_samples"]
                )
                for key, grid in grids.items():
                    writer.add_image(key, grid, global_step=global_step)
            avg_loss += loss
        avg_loss /= num_batches
        print()
    model.train()
    return avg_loss
