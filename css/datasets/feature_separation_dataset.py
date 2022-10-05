import os

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import kaldi_native_io as kio


class FeatureSeparationDataset(Dataset):
    """
    Uses precomputed features. Initialize with a data dir containing the following
    files: mix.scp, source0.scp, source1.scp, noise.scp, feats.scp
    """

    def __init__(self, data_dir):
        # Get list of utterances
        with open(os.path.join(data_dir, "feats.scp"), "r") as f:
            self.utt_ids = [line.split(" ")[0] for line in f]
        # Lazy reading (done by worker so that reader object does not get pickled)
        self._readers = None
        self._data_dir = data_dir
        self.num_sources = 3

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, index):
        if self._readers is None:
            # Load features lazily
            self._readers = {
                key: kio.RandomAccessFloatMatrixReader(
                    f"scp:{os.path.join(self._data_dir, key + '.scp')}"
                )
                for key in ["mix", "src0", "src1", "noise", "feats"]
            }
        utt_id = self.utt_ids[index]
        sample_dict = {
            key: torch.tensor(self._readers[key][utt_id]) for key in self._readers
        }
        sample_dict["utt_id"] = utt_id
        return sample_dict


def feature_collater(data):
    """
    data is a list of dicts containing keys "mix", "src0", "src1", "noise", "feats"
    """
    data = sorted(data, key=lambda x: x["mix"].shape[0], reverse=True)
    lengths = torch.tensor([x["mix"].shape[0] for x in data])
    padded_batch = {
        key: pad_sequence([x[key] for x in data], batch_first=True)
        for key in ["mix", "src0", "src1", "noise", "feats"]
    }
    padded_batch["len"] = lengths
    return padded_batch
