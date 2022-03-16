# -*- coding: utf-8 -*-
# Copyright 2020  Matthew Wiesner
# Apache 2.0

import math

DEFAULT_LR_SCHEDULER_CONFIG = {
    "warmup": 0,
    "decay": 0.0,
    "fixed": 0,
    "min_lr": 1.0e-9
}


class LRScheduler(object):
    def __init__(self, optimizer, conf):
        lr_conf = DEFAULT_LR_SCHEDULER_CONFIG
        lr_conf.update(conf)
        
        self.optimizer = optimizer
        self.warmup = lr_conf["warmup"]
        self.fixed = lr_conf["fixed"]
        self.decay = lr_conf["decay"]
        self.min_lr = lr_conf["min_lr"]

        self.num_warmup_updates = 0
        self.num_fixed_updates = 0
        self.num_decay_updates = 0
        self.lr = self.optimizer.param_groups[0]["lr"]
        if self.warmup > 0:
            self.set_lr(self.min_lr)
            self.curr_lr = self.min_lr
        else:
            self.curr_lr = self.lr

    def step(self, num_new_updates):
        if self.warmup > 0 and self.num_warmup_updates < self.warmup:
            self.num_warmup_updates += num_new_updates
            slope = (self.lr - self.min_lr) / float(self.warmup)
            new_lr = self.min_lr + slope * self.num_warmup_updates
        elif self.fixed > 0 and self.num_fixed_updates < self.fixed:
            self.num_fixed_updates += num_new_updates
            new_lr = self.lr
        else:
            self.num_decay_updates += num_new_updates
            factor = math.exp(-self.decay * self.num_decay_updates)
            new_lr = self.lr * factor
        self.set_lr(new_lr)
        self.curr_lr = new_lr

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def state_dict(self):
        return {
            "warmup": self.warmup,
            "fixed": self.fixed,
            "decay": self.decay,
            "warmup_updates": self.num_warmup_updates,
            "fixed_updates": self.num_fixed_updates,
            "decay_updates": self.num_decay_updates,
            "lr": self.lr,
            "curr_lr": self.curr_lr,
            "min_lr": self.min_lr,
        }

    def load_state_dict(self, state_dict):
        self.warmup = state_dict["warmup"]
        self.fixed = state_dict["fixed"]
        self.decay = state_dict["decay"]
        self.num_warmup_updates = state_dict["warmup_updates"]
        self.num_fixed_updates = state_dict["fixed_updates"]
        self.num_decay_updates = state_dict["decay_updates"]
        self.lr = state_dict["lr"]
        self.curr_lr = state_dict["curr_lr"]
        self.min_lr = state_dict["min_lr"]
