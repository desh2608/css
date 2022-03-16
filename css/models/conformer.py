#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2020  Microsoft (author: Sanyuan Chen)
#           2021  Johns Hopkins University (author: Desh Raj)
# Apache 2.0

import math
import numpy
import torch

DEFAULT_CONFORMER_CONF = {
    "attention_dim": 256,
    "attention_heads": 4,
    "linear_units": 1024,
    "num_blocks": 16,
    "kernel_size": 33,
    "dropout_rate": 0.1,
    "relative_pos_emb": True,
    "causal": False,
}

EPSILON = torch.finfo(torch.float32).eps


class Conformer(torch.nn.Module):
    """
    Conformer model
    """

    @classmethod
    def build_model(cls, conf):
        conf_dict = DEFAULT_CONFORMER_CONF
        conf_dict.update(conf)
        model = Conformer(**conf_dict)
        return model

    def __init__(
        self,
        idim=257,
        num_bins=257,
        num_spk=2,
        num_noise=1,
        attention_dim=256,
        attention_heads=4,
        linear_units=1024,
        num_blocks=16,
        kernel_size=33,
        dropout_rate=0.1,
        causal=False,
        relative_pos_emb=True,
    ):
        super(Conformer, self).__init__()

        # Conformer Encoders
        self.conformer = ConformerEncoder(
            idim=idim,
            attention_dim=attention_dim,
            attention_heads=attention_heads,
            linear_units=linear_units,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            causal=causal,
            relative_pos_emb=relative_pos_emb,
        )
        self.num_spk = num_spk
        self.num_noise = num_noise
        self.num_bins = num_bins
        self.linear = torch.nn.Linear(attention_dim, num_bins * (num_spk + num_noise))

    def forward(self, f):
        """
        args
            f: B x T x F
        return
            m: [B x T x F, ...]
        """
        f, _ = self.conformer(f, masks=None)
        masks = self.linear(f)
        masks = torch.nn.functional.relu(masks)
        masks = torch.chunk(masks, self.num_spk + self.num_noise, 2)
        return masks


class ConformerEncoder(torch.nn.Module):
    """Conformer Encoder https://arxiv.org/abs/2005.08100"""

    def __init__(
        self,
        idim=257,
        attention_dim=256,
        attention_heads=4,
        linear_units=1024,
        num_blocks=16,
        kernel_size=33,
        dropout_rate=0.1,
        causal=False,
        relative_pos_emb=True,
    ):
        super(ConformerEncoder, self).__init__()

        self.embed = torch.nn.Sequential(
            torch.nn.Linear(idim, attention_dim),
            torch.nn.LayerNorm(attention_dim),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU(),
        )

        if relative_pos_emb:
            self.pos_emb = RelativePositionalEncoding(
                attention_dim // attention_heads, 1000, False
            )
        else:
            self.pos_emb = None

        self.encoders = torch.nn.Sequential(
            *[
                EncoderLayer(
                    attention_dim,
                    attention_heads,
                    linear_units,
                    kernel_size,
                    dropout_rate,
                    causal=causal,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, xs, masks):
        xs = self.embed(xs)

        if self.pos_emb is not None:
            x_len = xs.shape[1]
            pos_seq = torch.arange(0, x_len).long().to(xs.device)
            pos_seq = pos_seq[:, None] - pos_seq[None, :]
            pos_k, _ = self.pos_emb(pos_seq)
        else:
            pos_k = None
        for layer in self.encoders:
            xs = layer(xs, pos_k, masks)

        return xs, masks


class RelativePositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, maxlen=1000, embed_v=False):
        super(RelativePositionalEncoding, self).__init__()

        self.d_model = d_model
        self.maxlen = maxlen
        self.pe_k = torch.nn.Embedding(2 * maxlen, d_model)
        if embed_v:
            self.pe_v = torch.nn.Embedding(2 * maxlen, d_model)
        self.embed_v = embed_v

    def forward(self, pos_seq):
        pos_seq.clamp_(-self.maxlen, self.maxlen - 1)
        pos_seq = pos_seq + self.maxlen
        if self.embed_v:
            return self.pe_k(pos_seq), self.pe_v(pos_seq)
        else:
            return self.pe_k(pos_seq), None


class MultiHeadedAttention(torch.nn.Module):
    """Multi-Head Attention layer.

    :param int n_head: the number of head s
    :param int n_feat: the number of features
    :param float dropout_rate: dropout rate

    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.layer_norm = torch.nn.LayerNorm(n_feat)
        self.linear_q = torch.nn.Linear(n_feat, n_feat)
        self.linear_k = torch.nn.Linear(n_feat, n_feat)
        self.linear_v = torch.nn.Linear(n_feat, n_feat)

        self.linear_out = torch.nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x, pos_k, mask):
        """Compute 'Scaled Dot Product Attention'.

        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = x.size(0)
        x = self.layer_norm(x)
        q = self.linear_q(x).view(n_batch, -1, self.h, self.d_k)  # (b, t, d)
        k = self.linear_k(x).view(n_batch, -1, self.h, self.d_k)  # (b, t, d)
        v = self.linear_v(x).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        A = torch.matmul(q, k.transpose(-2, -1))
        reshape_q = q.contiguous().view(n_batch * self.h, -1, self.d_k).transpose(0, 1)
        if pos_k is not None:
            B = torch.matmul(reshape_q, pos_k.transpose(-2, -1))
            B = B.transpose(0, 1).view(n_batch, self.h, pos_k.size(0), pos_k.size(1))
            scores = (A + B) / math.sqrt(self.d_k)
        else:
            scores = A / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(
                numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
            )
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)
        return self.dropout(self.linear_out(x))  # (batch, time1, d_model)


class ConvModule(torch.nn.Module):
    def __init__(self, input_dim, kernel_size, dropout_rate, causal=False):
        super(ConvModule, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(input_dim)

        self.pw_conv_1 = torch.nn.Conv2d(1, 2, 1, 1, 0)
        self.glu_act = torch.torch.nn.Sigmoid()
        self.causal = causal
        if causal:
            self.dw_conv_1d = torch.nn.Conv1d(
                input_dim,
                input_dim,
                kernel_size,
                1,
                padding=(kernel_size - 1),
                groups=input_dim,
            )
        else:
            self.dw_conv_1d = torch.nn.Conv1d(
                input_dim,
                input_dim,
                kernel_size,
                1,
                padding=(kernel_size - 1) // 2,
                groups=input_dim,
            )
        self.BN = torch.nn.BatchNorm1d(input_dim)
        self.act = torch.nn.ReLU()
        self.pw_conv_2 = torch.nn.Conv2d(1, 1, 1, 1, 0)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layer_norm(x)
        x = self.pw_conv_1(x)
        x = x[:, 0] * self.glu_act(x[:, 1])
        x = x.permute([0, 2, 1])
        x = self.dw_conv_1d(x)
        if self.causal:
            x = x[:, :, : -(self.kernel_size - 1)]
        x = self.BN(x)
        x = self.act(x)
        x = x.unsqueeze(1).permute([0, 1, 3, 2])
        x = self.pw_conv_2(x)
        x = self.dropout(x).squeeze(1)
        return x


class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_inner, dropout_rate):
        super(FeedForward, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner

        self.layer_norm = torch.nn.LayerNorm(d_model)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_inner),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(d_inner, d_model),
            torch.nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        x = self.layer_norm(x)
        out = self.net(x)

        return out


class EncoderLayer(torch.nn.Module):
    """Encoder layer module.

    :param int d_model: attention vector size
    :param int n_head: number of heads
    :param int d_ffn: feedforward size
    :param int kernel_size: cnn kernal size, it must be an odd
    :param int dropout_rate: dropout_rate
    """

    def __init__(self, d_model, n_head, d_ffn, kernel_size, dropout_rate, causal=False):
        """Construct an EncoderLayer object."""
        super(EncoderLayer, self).__init__()
        self.feed_forward_in = FeedForward(d_model, d_ffn, dropout_rate)
        self.self_attn = MultiHeadedAttention(n_head, d_model, dropout_rate)
        self.conv = ConvModule(d_model, kernel_size, dropout_rate, causal=causal)
        self.feed_forward_out = FeedForward(d_model, d_ffn, dropout_rate)
        self.layer_norm = torch.nn.LayerNorm(d_model)

    def forward(self, x, pos_k, mask):
        """Compute encoded features.

        :param torch.Tensor x: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        x = x + 0.5 * self.feed_forward_in(x)
        x = x + self.self_attn(x, pos_k, mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.feed_forward_out(x)

        out = self.layer_norm(x)

        return out
