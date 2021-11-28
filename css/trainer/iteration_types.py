#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2020  Matthew Wiesner
#           2021  Desh Raj
# Apache 2.0

import torch


def train_one_epoch(args, dataloader, model, objective, optim, lr_sched, device="cpu"):
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

    for i in range(1, args.batches_per_epoch + 1):
        b = next(iter(dataloader))
        print(
            "Iter: ",
            i,
            " of ",
            args.batches_per_epoch,
            "LR: {:0.5e}".format(lr_sched.curr_lr),
            "bsize: ",
            b["mix"].size(1),
            "window (# frames): ",
            b["mix"].size(2),
            end=" ",
        )
        loss = objective(model, b, device=device)
        print("Loss: {:0.5f}".format(loss.data.item()), end=" ")
        total_loss += loss.data.item()

        loss.backward()
        loss.detach()
        del b

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_thresh)
        print("Grad_norm: {:0.5f}".format(grad_norm.data.item()), end="")
        print()
        optim.step()
        optim.zero_grad()
        lr_sched.step(1.0)
    return total_loss / args.batches_per_epoch


def validate(dataloader, model, objective, device="cpu"):
    model.eval()
    with torch.no_grad():
        avg_loss = 0.0
        for _ in range(100):  # 100 batches of size 32
            b = next(iter(dataloader))
            avg_loss += objective(model, b, device=device)
        avg_loss /= 100
        print()
    model.train()
    return avg_loss
