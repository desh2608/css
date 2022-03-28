import json
import soundfile as sf

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class RawWaveformSeparationDataset(Dataset):
    """
    Example entry in JSON file:
    {
    "output": "/export/c07/draj/sim-LibriMix/data/SimLibriUttmix-dev/wav/2428-83699-0029_0.wav",
    "inputs": [
      "/export/c07/draj/sim-LibriMix/data/dev/wav_newseg/2428/83699/2428-83699-0029_0.wav"
    ],
    "mixer": "ReverbMixPartialNoise",
    "implementation": "libaueffect mixers_twosources.rmix_partial",
    "sir": Infinity,
    "amplitude": 0.3353882934109555,
    "overlap length in seconds": 0,
    "t60": 0.254389243537674,
    "angles": [
      294.06338912907876,
      272.52352309916245
    ],
    "snr": 7.107776824519226,
    "source0": "/export/c07/draj/sim-LibriMix/data/SimLibriUttmix-dev/wav/2428-83699-0029_0_s0.wav",
    "source1": "/export/c07/draj/sim-LibriMix/data/SimLibriUttmix-dev/wav/2428-83699-0029_0_s1.wav",
    "noise": "/export/c07/draj/sim-LibriMix/data/SimLibriUttmix-dev/wav/2428-83699-0029_0_s2.wav"
    },
    """

    def __init__(self, mix_json):
        with open(mix_json, "r") as f:
            self.mix_dict = json.load(f)
        self.num_sources = 3

    def __len__(self):
        return len(self.mix_dict)

    def __getitem__(self, index):
        sample = self.mix_dict[index]
        sample_dict = {}
        sample_dict["mix"] = torch.tensor(
            sf.read(sample["output"], always_2d=True)[0], dtype=torch.float
        )
        sample_dict["targets"] = torch.stack(
            [
                torch.tensor(sf.read(sample[key], always_2d=True)[0], dtype=torch.float)
                for key in ["source0", "source1", "noise"]
            ],
            dim=-1,
        )
        return sample_dict


def raw_waveform_collater(data):
    """
    data is a list of dicts containing keys "mix", "targets"
    """
    data = sorted(data, key=lambda x: x["mix"].shape[0], reverse=True)
    mix_batch = pad_sequence([d["mix"] for d in data], batch_first=True)
    targets_batch = pad_sequence([d["targets"] for d in data], batch_first=True)
    lengths = torch.LongTensor([d["mix"].shape[0] for d in data])
    return {
        "mix": mix_batch.transpose(1, 2),  # B x D x N
        "targets": targets_batch.transpose(1, 2).unbind(dim=-1),  # 3 x [B x D x N]
        "len": lengths,
    }
