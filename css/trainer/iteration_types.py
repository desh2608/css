#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2020  Matthew Wiesner
#           2021  Desh Raj
# Apache 2.0

import logging
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
    feat,
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

        if feat is not None:
            # Feature extraction
            mix_stft, f, _, _ = feat.forward(b["mix"].to(device))
            targets_stft = torch.stack(
                [feat.forward(t.to(device))[0] for t in b["targets"]], dim=1
            )
            batch = {
                "mix": mix_stft.transpose(1, 2),  # B x T x F
                "targets": targets_stft.transpose(2, 3),  # B x 3 x T x F
                "feats": f.transpose(1, 2),  # B x T x *
                "len": b["len"].to(device),
            }
        else:
            batch = {
                "mix": b["mix"],  # B x T x F
                "targets": torch.stack(
                    [b[src] for src in ["src0", "src1", "noise"]], dim=1
                ),  # B x 3 x T x F
                "feats": b["feats"],  # B x T x *
                "len": b["len"],
            }

        if i % conf["tensorboard"]["log_interval"] == 0:
            loss, tensors = objective(model, batch, device=device, return_est=True)
            grids = make_grid_from_tensors(
                tensors, num_samples=conf["tensorboard"]["num_samples"]
            )
            for key, grid in grids.items():
                writer.add_image(key, grid, global_step=global_step)
        else:
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
        logging.info(log)

        writer.add_scalar("loss/train", loss.data.item(), global_step=global_step)
        optim.step()
        optim.zero_grad()
        lr_sched.step(1.0)
    return total_loss / num_batches, global_step


def validate(dataloader, model, feat, objective, device):
    model.eval()
    with torch.no_grad():
        avg_loss = 0.0
        num_batches = len(dataloader)
        for b in dataloader:
            if feat is not None:
                # Feature extraction
                mix_stft, f, _, _ = feat.forward(b["mix"].to(device))
                targets_stft = torch.stack(
                    [feat.forward(t.to(device))[0] for t in b["targets"]], dim=1
                )
                batch = {
                    "mix": mix_stft.transpose(1, 2),  # B x T x F
                    "targets": targets_stft.transpose(2, 3),  # B x 3 x T x F
                    "feats": f.transpose(1, 2),  # B x T x *
                    "len": b["len"].to(device),
                }
            else:
                batch = {
                    "mix": b["mix"].to(device),  # B x T x F
                    "targets": torch.stack(
                        [b[src].to(device) for src in ["src0", "src1", "noise"]], dim=1
                    ),  # B x 3 x T x F
                    "feats": b["feats"].to(device),  # B x T x *
                    "len": b["len"].to(device),
                }
            avg_loss += objective(model, batch, device=device)
        avg_loss /= num_batches
        print()
    model.train()
    return avg_loss
