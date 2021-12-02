#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2020  Matthew Wiesner
#           2021  Desh Raj
# Apache 2.0

import logging
import torch

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def train_one_epoch(args, generator, model, objective, optim, lr_sched, device="cpu"):
    """
    The training interator: It takes
         - a data loader
         - a model
         - a training objective
         - an optimizer
         - a learning rate scheduler

    It defines how to use these components to perform 1 epoch of training.
    """
    if args.gpu and args.fp16:
        logging.info("Using fp16 operations")
        scaler = torch.cuda.amp.GradScaler()

    total_loss = 0.0

    for i in range(1, args.batches_per_epoch + 1):
        b = next(generator)
        log = f"Iter: {i} of {args.batches_per_epoch} LR:{lr_sched.curr_lr:0.5e} bsize: {b['mix'].size(0)} window (# frames): {b['mix'].size(1)} ovl: {b['ovl']:0.4f} "

        if args.gpu and args.fp16:
            with torch.cuda.amp.autocast():
                loss = objective(model, b, device=device)
        else:
            loss = objective(model, b, device=device)

        log += f"Loss: {loss.data.item():0.5f} "
        total_loss += loss.data.item()

        if args.gpu and args.fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        loss.detach()
        del b

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_thresh)
        log += f"Grad_norm: {grad_norm.data.item():0.5f}"
        logging.info(log)
        if args.gpu and args.fp16:
            scaler.step(optim)
            scaler.update()
        else:
            optim.step()
        optim.zero_grad()
        lr_sched.step(1.0)
    return total_loss / args.batches_per_epoch


def validate(generator, model, objective, device="cpu"):
    model.eval()
    with torch.no_grad():
        avg_loss = 0.0
        for _ in range(100):  # 100 batches of size 32
            b = next(generator)
            avg_loss += objective(model, b, device=device)
        avg_loss /= 100
        print()
    model.train()
    return avg_loss
