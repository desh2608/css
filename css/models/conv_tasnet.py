#!/usr/bin/env python3
# This module is taken from: https://github.com/JusperLee/Conv-TasNet/blob/master/Conv_TasNet_Pytorch/Conv_TasNet.py

import torch

DEFAULT_CONV_TASNET_CONF = {
    "num_filters": 512,
    "filter_length": 16,
    "bottleneck_channels": 128,
    "conv_channels": 512,
    "kernel_size": 3,
    "num_blocks": 8,
    "num_layers": 3,
}


class ConvTasNet(torch.nn.Module):
    """
    Conformer model
    """

    @staticmethod
    def add_args(parser):
        parser.add_argument("--num-spk", type=int, default=2)
        parser.add_argument("--num-noise", type=int, default=1)
        parser.add_argument("--conv-tasnet-num-filters", type=int, default=256)
        parser.add_argument("--conv-tasnet-filter-length", type=int, default=16)
        parser.add_argument("--conv-tasnet-bottleneck-channels", type=int, default=128)
        parser.add_argument("--conv-tasnet-conv-channels", type=int, default=256)
        parser.add_argument("--conv-tasnet-kernel-size", type=int, default=3)
        parser.add_argument("--conv-tasnet-num-blocks", type=int, default=8)
        parser.add_argument("--conv-tasnet-num-layers", type=int, default=3)
        parser.add_argument(
            "--conv-tasnet-norm", type=str, default="gln", choices=["gln", "cln", "bn"]
        )

    @classmethod
    def build_model(cls, conf):
        conv_tasnet_conf = {
            "num_filters": int(conf["conv_tasnet_num_filters"]),
            "filter_length": int(conf["conv_tasnet_filter_length"]),
            "bottleneck_channels": int(conf["conv_tasnet_bottleneck_channels"]),
            "conv_channels": int(conf["conv_tasnet_conv_channels"]),
            "kernel_size": int(conf["conv_tasnet_kernel_size"]),
            "num_blocks": int(conf["conv_tasnet_num_blocks"]),
            "num_layers": int(conf["conv_tasnet_num_layers"]),
            "norm": conf["conv_tasnet_norm"],
        }
        model = ConvTasNet(
            num_spk=conf["num_spk"],
            num_noise=conf["num_noise"],
            conv_tasnet_conf=conv_tasnet_conf,
        )
        return model

    def __init__(
        self,
        num_spk=2,
        num_noise=1,
        conv_tasnet_conf=DEFAULT_CONV_TASNET_CONF,
        activate="relu",
        causal=False,
    ):
        N = conv_tasnet_conf["num_filters"]
        L = conv_tasnet_conf["filter_length"]
        B = conv_tasnet_conf["bottleneck_channels"]
        H = conv_tasnet_conf["conv_channels"]
        P = conv_tasnet_conf["kernel_size"]
        X = conv_tasnet_conf["num_blocks"]
        R = conv_tasnet_conf["num_layers"]
        norm = conv_tasnet_conf["norm"]

        super(ConvTasNet, self).__init__()
        # n x 1 x T => n x N x T
        self.encoder = Conv1D(1, N, L, stride=L // 2, padding=0)
        # n x N x T  Layer Normalization of Separation
        self.LayerN_S = select_norm("cln", N)
        # n x B x T  Conv 1 x 1 of  Separation
        self.BottleN_S = Conv1D(N, B, 1)
        # Separation block
        # n x B x T => n x B x T
        self.separation = self._Sequential_repeat(
            R, X, in_channels=B, out_channels=H, kernel_size=P, norm=norm, causal=causal
        )
        # n x B x T => n x 2*N x T
        self.gen_masks = Conv1D(B, (num_spk + num_noise) * N, 1)
        # n x N x T => n x 1 x L
        self.decoder = ConvTrans1D(N, 1, L, stride=L // 2)
        # activation function
        active_f = {
            "relu": torch.nn.ReLU(),
            "sigmoid": torch.nn.Sigmoid(),
            "softmax": torch.nn.Softmax(dim=0),
        }
        self.activation_type = activate
        self.activation = active_f[activate]
        self.num_spk = num_spk
        self.num_noise = num_noise

    def _Sequential_block(self, num_blocks, **block_kwargs):
        """
        Sequential 1-D Conv Block
        input:
              num_block: how many blocks in every repeats
              **block_kwargs: parameters of Conv1D_Block
        """
        Conv1D_Block_lists = [
            Conv1D_Block(**block_kwargs, dilation=(2 ** i)) for i in range(num_blocks)
        ]

        return torch.nn.Sequential(*Conv1D_Block_lists)

    def _Sequential_repeat(self, num_repeats, num_blocks, **block_kwargs):
        """
        Sequential repeats
        input:
              num_repeats: Number of repeats
              num_blocks: Number of block in every repeats
              **block_kwargs: parameters of Conv1D_Block
        """
        repeats_lists = [
            self._Sequential_block(num_blocks, **block_kwargs)
            for i in range(num_repeats)
        ]
        return torch.nn.Sequential(*repeats_lists)

    def forward(self, x):
        if x.dim() >= 3:
            raise RuntimeError(
                "{} accept 1/2D tensor as input, but got {:d}".format(
                    self.__name__, x.dim()
                )
            )
        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)
        # x: n x 1 x L => n x N x T
        w = self.encoder(x)
        # n x N x L => n x B x L
        e = self.LayerN_S(w)
        e = self.BottleN_S(e)
        # n x B x L => n x B x L
        e = self.separation(e)
        # n x B x L => n x (num_spk+num_noise)*N x L
        m = self.gen_masks(e)
        # n x N x L x num_spks
        m = torch.chunk(m, chunks=self.num_spk + self.num_noise, dim=1)
        # (num_spks + num_noise) x n x N x L
        m = self.activation(torch.stack(m, dim=0))
        d = [w * m[i] for i in range(self.num_spk + self.num_noise)]
        # decoder part (num_spks + num_noise) x n x L
        s = [
            self.decoder(d[i], squeeze=True)
            for i in range(self.num_spk + self.num_noise)
        ]
        return torch.stack(s[:-1], dim=1)


class GlobalLayerNorm(torch.nn.Module):
    """
    Calculate Global Layer Normalization
    dim: (int or list or torch.Size) –
         input shape from an expected input of size
    eps: a value added to the denominator for numerical stability.
    elementwise_affine: a boolean value that when set to True,
        this module has learnable per-element affine parameters
        initialized to ones (for weights) and zeros (for biases).
    """

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = torch.nn.Parameter(torch.ones(self.dim, 1))
            self.bias = torch.nn.Parameter(torch.zeros(self.dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        # x = N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x L
        # gln: mean,var N x 1 x 1
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(self.__name__))

        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
        # N x C x L
        if self.elementwise_affine:
            x = self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class CumulativeLayerNorm(torch.nn.LayerNorm):
    """
    Calculate Cumulative Layer Normalization
    dim: you want to norm dim
    elementwise_affine: learnable per-element affine parameters
    """

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine
        )

    def forward(self, x):
        # x: N x C x L
        # N x L x C
        x = torch.transpose(x, 1, 2)
        # N x L x C == only channel norm
        x = super().forward(x)
        # N x C x L
        x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim):
    if norm == "gln":
        return GlobalLayerNorm(dim, elementwise_affine=True)
    if norm == "cln":
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    elif norm == "bn":
        return torch.nn.BatchNorm1d(dim)
    else:
        raise ValueError("Unknown normalization: {}".format(norm))


class Conv1D(torch.nn.Conv1d):
    """
    Applies a 1D convolution over an input signal composed of several input planes.
    """

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        # x: N x C x L
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


class ConvTrans1D(torch.nn.ConvTranspose1d):
    """
    This module can be seen as the gradient of Conv1d with respect to its input.
    It is also known as a fractionally-strided convolution
    or a deconvolution (although it is not an actual deconvolution operation).
    """

    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


class Conv1D_Block(torch.nn.Module):
    """
    Consider only residual links
    """

    def __init__(
        self,
        in_channels=256,
        out_channels=512,
        kernel_size=3,
        dilation=1,
        norm="gln",
        causal=False,
    ):
        super(Conv1D_Block, self).__init__()
        # conv 1 x 1
        self.conv1x1 = Conv1D(in_channels, out_channels, 1)
        self.PReLU_1 = torch.nn.PReLU()
        self.norm_1 = select_norm(norm, out_channels)
        # not causal don't need to padding, causal need to pad+1 = kernel_size
        self.pad = (
            (dilation * (kernel_size - 1)) // 2
            if not causal
            else (dilation * (kernel_size - 1))
        )
        # depthwise convolution
        self.dwconv = Conv1D(
            out_channels,
            out_channels,
            kernel_size,
            groups=out_channels,
            padding=self.pad,
            dilation=dilation,
        )
        self.PReLU_2 = torch.nn.PReLU()
        self.norm_2 = select_norm(norm, out_channels)
        self.Sc_conv = torch.nn.Conv1d(out_channels, in_channels, 1, bias=True)
        self.causal = causal

    def forward(self, x):
        # x: N x C x L
        # N x O_C x L
        c = self.conv1x1(x)
        # N x O_C x L
        c = self.PReLU_1(c)
        c = self.norm_1(c)
        # causal: N x O_C x (L+pad)
        # noncausal: N x O_C x L
        c = self.dwconv(c)
        # N x O_C x L
        if self.causal:
            c = c[:, :, : -self.pad]
        c = self.PReLU_2(c)
        c = self.norm_2(c)
        c = self.Sc_conv(c)
        return x + c
