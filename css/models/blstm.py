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

    @classmethod
    def build_model(cls, conf):
        conf_dict = DEFAULT_BLSTM_CONF
        conf_dict.update(conf)
        model = BLSTM(**conf_dict)
        return model

    def __init__(
        self,
        idim=257,
        num_bins=257,
        num_spk=2,
        num_noise=1,
        hidden_dim=512,
        num_layers=4,
        dropout_rate=0.1,
    ):
        super(BLSTM, self).__init__()

        # BLSTM Encoders
        self.blstm = BLSTMEncoder(
            idim=idim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
        )

        self.num_spk = num_spk
        self.num_noise = num_noise
        self.num_bins = num_bins
        self.linear = torch.nn.Linear(hidden_dim, num_bins * (num_spk + num_noise))

    def forward(self, f):
        """
        args
            f: B x T x F
        return
            m: [B x T x F, ...]
        """
        f = self.blstm(f)
        masks = self.linear(f)
        masks = torch.nn.functional.relu(masks)
        masks = torch.chunk(masks, self.num_spk + self.num_noise, 2)
        return masks


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
        self.dropout = torch.nn.Dropout(dropout_rate, inplace=True)

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
