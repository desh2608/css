from pathlib import Path
import numpy as np
import torchaudio
import torch


class EgsReader(object):
    """
    Egs reader
    """

    def __init__(self, recordings, multi_channel=False):
        self.mix_reader = WaveReader(recordings, multi_channel=multi_channel)

    def __len__(self):
        return len(self.mix_reader)

    def __iter__(self):
        for key, mix in self.mix_reader:
            egs = dict()
            egs["mix"] = mix
            yield key, egs


class WaveReader(object):
    """
    Sequential/Random Reader for single channel wave based on Lhotse.
    """

    def __init__(self, recordings, multi_channel=False):
        super(WaveReader, self).__init__()
        self.recordings = recordings
        self.multi_channel = multi_channel

    def _load(self, key):
        # return C x N or N
        if self.multi_channel:
            samps = torch.Tensor(self.recordings[key].load_audio())
        else:
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
