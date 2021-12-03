#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2021  Johns Hopkins University (author: Desh Raj)
# Apache 2.0

import torch

DEFAULT_BLSTM_CONF = {
    "hidden_dim": 512,
    "num_layers": 4,
    "dropout_rate": 0.1,
}

EPSILON = torch.finfo(torch.float32).eps


class BLSTM(torch.nn.Module):
    """
    BLSTM model
    """

    @staticmethod
    def add_args(parser):
        parser.add_argument("--idim", type=int, default=257)
        parser.add_argument("--num-bins", type=int, default=257)
        parser.add_argument("--num-spk", type=int, default=2)
        parser.add_argument("--num-noise", type=int, default=1)
        parser.add_argument("--blstm-hdim", type=int, default=1024)
        parser.add_argument("--blstm-num-layers", type=int, default=3)
        parser.add_argument("--blstm-dropout-rate", type=float, default=0.1)

    @classmethod
    def build_model(cls, conf):
        blstm_conf = {
            "hidden_dim": int(conf["blstm_hdim"]),
            "num_layers": int(conf["blstm_num_layers"]),
            "dropout_rate": float(conf["blstm_dropout_rate"]),
        }
        model = BLSTM(
            in_features=conf["idim"],
            num_bins=conf["num_bins"],
            num_spk=conf["num_spk"],
            num_noise=conf["num_noise"],
            blstm_conf=blstm_conf,
        )
        return model

    def __init__(
        self,
        in_features=257,
        num_bins=257,
        num_spk=2,
        num_noise=1,
        blstm_conf=DEFAULT_BLSTM_CONF,
    ):
        super(BLSTM, self).__init__()

        # BLSTM Encoders
        self.blstm = BLSTMEncoder(in_features, **blstm_conf)

        self.num_bins = num_bins
        self.num_spk = num_spk
        self.num_noise = num_noise
        self.linear = torch.nn.Linear(
            blstm_conf["hidden_dim"], num_bins * (num_spk + num_noise)
        )

    def forward(self, f):
        """
        args
            f: N x T x F
        return
            m: [N x T x F, ...]
        """
        if f.ndim == 4:
            f = f.squeeze(0)
        f_orig = f.clone().detach()

        # MVN
        f = (f - f.mean(-2, keepdim=True)) / (f.std(-2, keepdim=True) + EPSILON)

        f = self.blstm(f)
        masks = self.linear(f)

        masks = torch.sigmoid(masks)
        masks = torch.chunk(masks, self.num_spk + self.num_noise, -1)
        y_pred = torch.stack([m * f_orig for m in masks[:-1]], dim=1)
        return y_pred


class BLSTMEncoder(torch.nn.Module):
    """
    BLSTM encoder
    """

    def __init__(
        self,
        idim=257,
        hidden_dim=1024,
        num_layers=4,
        dropout_rate=0.1,
    ):
        super(BLSTMEncoder, self).__init__()

        self.embed = torch.nn.Sequential(
            torch.nn.Linear(idim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU(),
        )

        self.encoders = torch.nn.Sequential(
            *[
                BLSTMLayer(
                    hidden_dim,
                    dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, xs):
        xs = self.embed(xs)

        for layer in self.encoders:
            xs = layer(xs)

        return xs


class BLSTMLayer(torch.nn.Module):
    """
    BLSTM layer
    """

    def __init__(self, h_dim, dropout_rate):
        """Construct an EncoderLayer object."""
        super(BLSTMLayer, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size=h_dim,
            hidden_size=h_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.layer_norm = torch.nn.LayerNorm(h_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        """Compute encoded features.

        :param torch.Tensor x: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        x = self.lstm(x)[0]
        x = self.layer_norm(x)
        x = self.dropout(x)

        return x
