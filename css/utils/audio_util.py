from pathlib import Path
import numpy as np
import torchaudio
import torch

MAX_INT16 = np.iinfo(np.int16).max
EPSILON = np.finfo(np.float32).eps


class EgsReader(object):
    """
    Egs reader
    """

    def __init__(self, recordings):
        self.mix_reader = SingleChannelWaveReader(recordings)

    def __len__(self):
        return len(self.mix_reader)

    def __iter__(self):
        for key, mix in self.mix_reader:
            egs = dict()
            egs["mix"] = mix
            yield key, egs


class SingleChannelWaveReader(object):
    """
    Sequential/Random Reader for single channel wave based on Lhotse.
    """

    def __init__(self, recordings, normalize=True):
        super(SingleChannelWaveReader, self).__init__()
        self.normalize = normalize
        self.recordings = recordings

    def _load(self, key):
        # return C x N or N
        sr, samps = read_wav(
            self.recordings[key], normalize=self.normalize, return_rate=True
        )
        return samps

    # number of utterance
    def __len__(self):
        return len(self.recordings)

    # avoid key error
    def __contains__(self, key):
        return key in self.recordings.ids

    # sequential index
    def __iter__(self):
        for key in self.recordings.ids:
            yield key, self._load(key)


def read_wav(recording, normalize=True, return_rate=False):
    """
    Read wave files using Lhotse `load_audio` method
    """
    # samps_int16: N x C or N
    #   N: number of samples
    #   C: number of channels
    samps = torch.Tensor(recording.load_audio(channels=0))
    samp_rate = recording.sampling_rate

    # normalize like MATLAB and librosa
    if normalize:
        samps = samps / MAX_INT16
    if return_rate:
        return samp_rate, samps
    return samps


def write_wav(fname, samps, sr=16000, normalize=True):
    """
    Write wav files in int16, support single/multi-channel
    """
    if normalize:
        samps = samps * MAX_INT16
    fname = Path(fname)
    fdir = fname.parent
    if not fdir.exists():
        fdir.mkdir(parents=True)
    torchaudio.save(fname, samps, sr)
