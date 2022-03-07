import math
import torch

import torch.nn.functional as F
import torch.nn as nn

from packaging import version

EPSILON = torch.finfo(torch.float32).eps
MATH_PI = math.pi


def init_kernel(frame_len, round_pow_of_two=True):
    def _rfft(x, n):
        if version.parse(torch.__version__) <= version.parse("1.7.1"):
            return torch.rfft(x, n)
        else:
            return torch.view_as_real(torch.fft.rfft(x, dim=n))

    # FFT points
    N = 2 ** math.ceil(math.log2(frame_len)) if round_pow_of_two else frame_len
    # window
    W = torch.hann_window(frame_len)
    # F x N/2+1 x 2
    K = _rfft(torch.eye(N), 1)[:frame_len]
    # 2 x N/2+1 x F
    K = torch.transpose(K, 0, 2) * W
    # N+2 x 1 x F
    K = torch.reshape(K, (N + 2, 1, frame_len))
    return K


class STFT:
    """
    Short-time Fourier Transform as a Layer
    """

    def __init__(self, frame_len, frame_hop, round_pow_of_two=True):
        super(STFT, self).__init__()
        K = init_kernel(frame_len, round_pow_of_two=round_pow_of_two)
        self.K = nn.Parameter(K, requires_grad=False)
        self.stride = frame_hop
        self.num_bins = self.K.shape[0] // 2

    def forward(self, x):
        """
        Accept (single or multiple channel) raw waveform and output magnitude and phase
        args
            x: input signal, B x N (batch x num_samples) or B x D x N (batch x num_channels x num_samples)
        return
            m: magnitude, B x F x T or B x D x F x T
            p: phase, B x F x T or B x D x F x T
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError(
                "{} expect 2D/3D tensor, but got {:d}D signal".format(
                    self.__name__, x.dim()
                )
            )

        # if B x N, reshape B x 1 x N
        if x.dim() == 2:
            x = torch.unsqueeze(x, 1)
            # B x 2F x T
            c = F.conv1d(x, self.K, stride=self.stride, padding=0)
            # B x F x T
            r, i = torch.chunk(c, 2, dim=1)

        # else reshape BD x 1 x N
        else:
            B, D, N = x.shape
            x = x.view(B * D, 1, N)
            # BD x 2F x T
            c = F.conv1d(x, self.K, stride=self.stride, padding=0)
            # B x D x 2F x T
            c = c.view(B, D, -1, c.shape[-1])
            # B x D x F x T
            r, i = torch.chunk(c, 2, dim=2)

        m = (r ** 2 + i ** 2) ** 0.5
        p = torch.atan2(i, r)
        return m, p, r, i


class IPDFeature(nn.Module):
    """
    Compute inter-channel phase difference
    """

    def __init__(
        self,
        ipd_index="1,0;2,0;3,0;4,0;5,0;6,0",
    ):
        super(IPDFeature, self).__init__()
        split_index = lambda sstr: [
            tuple(map(int, p.split(","))) for p in sstr.split(";")
        ]
        # ipd index
        pair = split_index(ipd_index)
        self.index_l = [t[0] for t in pair]
        self.index_r = [t[1] for t in pair]
        self.ipd_index = ipd_index
        self.num_pairs = len(pair)

    def forward(self, p):
        """
        Accept multi-channel phase and output inter-channel phase difference
        args
            p: phase matrix, N x C x F x T
        return
            ipd: N x MF x T
        """
        if p.dim() not in [3, 4]:
            raise RuntimeError(
                "{} expect 3/4D tensor, but got {:d} instead".format(
                    self.__name__, p.dim()
                )
            )
        # C x F x T => 1 x C x F x T
        if p.dim() == 3:
            p = p.unsqueeze(0)
        N, _, _, T = p.shape
        pha_dif = p[:, self.index_l] - p[:, self.index_r]

        # IPD mean normalization
        yr = torch.cos(pha_dif)
        yi = torch.sin(pha_dif)
        yrm = yr.mean(-1, keepdim=True)
        yim = yi.mean(-1, keepdim=True)
        ipd = torch.atan2(yi - yim, yr - yrm)

        # N x MF x T
        ipd = ipd.view(N, -1, T)
        return ipd


class FeatureExtractor(nn.Module):
    """
    A PyTorch module to handle spectral & spatial features
    """

    def __init__(
        self, frame_len=512, frame_hop=256, round_pow_of_two=True, ipd_index=None
    ):
        super(FeatureExtractor, self).__init__()
        # forward STFT
        self.forward_stft = STFT(
            frame_len, frame_hop, round_pow_of_two=round_pow_of_two
        )
        num_bins = self.forward_stft.num_bins
        self.feature_dim = num_bins
        self.num_bins = num_bins
        # inter-channel phase difference
        self.ipd_feature = IPDFeature(ipd_index) if ipd_index else None

    def forward(self, x):
        """
        Compute spectral and spatial features
        args
            x: B x N
        return:
            mag & pha: B x F x T
            feature: B x * x T
        """
        mag, p, r, i = self.forward_stft.forward(x)
        if mag.dim() == 4:
            # just pick first channel
            mag = mag[:, 0, ...]
        f = torch.clamp(mag, min=EPSILON)
        # mvn
        f = (f - f.mean(-1, keepdim=True)) / (f.std(-1, keepdim=True) + EPSILON)
        if self.ipd_feature:
            ipd = self.ipd_feature.forward(p)
            f = torch.cat([f, ipd], dim=1)
        return mag, f, r, i
