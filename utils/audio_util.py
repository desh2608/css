import os
import numpy as np

import scipy.io.wavfile as wf

MAX_INT16 = np.iinfo(np.int16).max
EPSILON = np.finfo(np.float32).eps


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
    samps = recording.load_audio(channels=0)
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
    # scipy.io.wavfile.write could write single/multi-channel files
    # for multi-channel, accept ndarray [Nsamples, Nchannels]
    if samps.ndim != 1 and samps.shape[0] < samps.shape[1]:
        samps = np.transpose(samps)
        samps = np.squeeze(samps)
    # same as MATLAB and kaldi
    samps_int16 = samps.astype(np.int16)
    fdir = os.path.dirname(fname)
    if fdir:
        os.makedirs(fdir, exist_ok=True)
    # NOTE: librosa 0.6.0 seems could not write non-float narray
    #       so use scipy.io.wavfile instead
    wf.write(fname, sr, samps_int16)
