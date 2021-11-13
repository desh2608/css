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
            x: input signal, B x N (batch x num_samples)
        return
            m: magnitude, B x F x T
            p: phase, B x F x T
        """
        assert x.ndim == 2, "Expected input of shape (batch size, num samples)"
        x = torch.unsqueeze(x, 1)
        # N x 2F x T
        c = F.conv1d(x, self.K, stride=self.stride, padding=0)
        # N x F x T
        r, i = torch.chunk(c, 2, dim=1)
        i = -i
        m = (r ** 2 + i ** 2) ** 0.5
        p = torch.atan2(i, r)
        return m, p, r, i


class FeatureExtractor(nn.Module):
    """
    A PyTorch module to handle spectral & spatial features
    """

    def __init__(self, frame_len=512, frame_hop=256, round_pow_of_two=True, num_spks=2):
        super(FeatureExtractor, self).__init__()
        # forward STFT
        self.forward_stft = STFT(
            frame_len, frame_hop, round_pow_of_two=round_pow_of_two
        )
        num_bins = self.forward_stft.num_bins
        self.feature_dim = num_bins
        self.num_bins = num_bins
        self.num_spks = num_spks

    def forward(self, x):
        """
        Compute spectra features
        args
            x: B x N
        return:
            mag & pha: B x F x T
            feature: B x * x T
        """
        mag, _, _, _ = self.forward_stft.forward(x)
        f = torch.clamp(mag, min=EPSILON)
        # mvn
        f = (f - f.mean(-1, keepdim=True)) / (f.std(-1, keepdim=True) + EPSILON)
        return mag, f
