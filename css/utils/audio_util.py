from pathlib import Path
import numpy as np
import torchaudio
import torch


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

    def __init__(self, recordings):
        super(SingleChannelWaveReader, self).__init__()
        self.recordings = recordings

    def _load(self, key):
        # return C x N or N
        samps = torch.Tensor(self.recordings[key].load_audio(channels=0))
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
