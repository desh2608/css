# Copyright      2021  Piotr Żelasko
#                2021  Desh Raj
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import random
import numpy as np

from lhotse import load_manifest, CutSet
from lhotse.dataset import CutMix, ReverbWithImpulseResponse, OnTheFlyFeatures
from lhotse.features import Spectrogram, SpectrogramConfig
from lhotse.utils import add_durations

import torch
from torch.utils.data import IterableDataset


class ContinuousSpeechSeparationDataset(IterableDataset):
    """
    Dataset for training mask estimation model for continuous speech separation.
    It contains all the common data pipeline modules used in separation
    experiments, e.g.:
    - cut mixing based on randomly sampled overlap ratios,
    - augmentation with RIR and noise,
    - on-the-fly feature extraction
    """

    @staticmethod
    def add_args(parser):
        parser.add_argument("--stft-frame-length", type=int, default=512)
        parser.add_argument("--stft-frame-shift", type=int, default=256)
        parser.add_argument("--stft-window-type", default="hanning")
        parser.add_argument(
            "--min-window-size",
            type=float,
            default=2,
            help="Minimum window size for each batch",
        )
        parser.add_argument(
            "--max-window-size",
            type=float,
            default=4,
            help="Maximum window size for each batch",
        )
        parser.add_argument(
            "--min-snr", type=float, default=5, help="Minimum SNR for isotropic noise"
        )
        parser.add_argument(
            "--max-snr", type=float, default=20, help="Maximum SNR for isotropic noise"
        )

    @classmethod
    def build_dataset(cls, manifest, conf):
        return ContinuousSpeechSeparationDataset(
            manifest,
            rir_manifest=conf["rir_manifest"],
            noise_manifest=conf["noise_manifest"],
            batch_size=conf["batch_size"],
            use_stft=conf["use_stft"],
            stft_frame_length=conf["stft_frame_length"],
            stft_frame_shift=conf["stft_frame_shift"],
            stft_window_type=conf["stft_window_type"],
            min_window_size=conf["min_window_size"],
            max_window_size=conf["max_window_size"],
            min_snr=conf["min_snr"],
            max_snr=conf["max_snr"],
            num_workers=conf["num_workers"],
        )

    def __init__(
        self,
        manifest,
        rir_manifest=None,
        noise_manifest=None,
        batch_size=32,
        use_stft=True,
        stft_frame_length=512,
        stft_frame_shift=256,
        stft_window_type="hann",
        min_window_size=2,
        max_window_size=5,
        min_snr=0,
        max_snr=10,
        num_workers=0,
    ):
        super(ContinuousSpeechSeparationDataset, self).__init__()
        self.manifest = manifest
        self.rir_manifest = rir_manifest
        self.noise_manifest = noise_manifest
        self.batch_size = batch_size
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.num_workers = num_workers  # Number of worker processes for multiprocessing

        logging.info("About to get single utterances")
        self.cuts_single = load_manifest(self.manifest)
        self.speakers = self.cuts_single.speakers
        self.sampling_rate = self.cuts_single.sample().sampling_rate

        # Spectrogram config for feature extraction
        if use_stft:
            self.stft_config = SpectrogramConfig(
                frame_length=stft_frame_length / self.sampling_rate,
                frame_shift=stft_frame_shift / self.sampling_rate,
                window_type=stft_window_type,
            )

            self.extractor = OnTheFlyFeatures(Spectrogram(self.stft_config))
        else:
            self.extractor = None

        # Create dictionary from speaker ids to list of cuts
        self.cuts_by_speaker = {
            speaker_id: self.cuts_single.filter(
                lambda c: c.supervisions[0].speaker == speaker_id
            )
            for speaker_id in self.speakers
        }

        self.transforms = []
        if self.rir_manifest is not None:
            logging.info("About to get RIR recordings")
            recordings_rir = load_manifest(self.rir_manifest)
            self.transforms.append(
                ReverbWithImpulseResponse(
                    rir_recordings=recordings_rir,
                    p=0.5,
                    normalize_output=True,
                    preserve_id=True,
                )
            )
        if self.noise_manifest is not None:
            logging.info("About to get isotropic noises")
            cuts_noise = load_manifest(self.noise_manifest)
            self.transforms.append(
                CutMix(
                    cuts=cuts_noise,
                    prob=0.5,
                    snr=(self.min_snr, self.max_snr),
                    preserve_id=True,
                )
            )

    def __iter__(self):
        return self

    def __next__(self):
        """
        This method is called by the PyTorch DataLoader to generate batches of data.
        Our data generation mechanism is as follows: we sample 2 speakers from the
        training set, and 1 utterance for each speaker. We then mix the utterances
        by randomly sampling a start time for the second utterance which could result
        in overlap ratio between 0 and 1. The resulting mixed utterance is chunked into
        windows of some size sampled between the minimum and maximum window sizes, and
        the chunks are added to the batch. This process is repeated until we filled the batch.
        We do the chunking to reduce the number of times mixing is done.
        Augmentation transforms are then applied to the chunks.
        """
        window_size = random.uniform(self.min_window_size, self.max_window_size)
        cuts1_padded = CutSet()
        cuts2_padded = CutSet()
        mix_cuts = CutSet()

        total_length = 0
        total_overlap = 0
        while len(mix_cuts) < self.batch_size:
            # Sample 2 cuts with different speaker ids
            spk1, spk2 = random.sample(self.speakers, 2)
            cut1 = random.choice(self.cuts_by_speaker[spk1])
            cut2 = random.choice(self.cuts_by_speaker[spk2])
            if cut1.duration < cut2.duration:
                # Swap cut1 and cut2 if cut1 is shorter
                cut1, cut2 = cut2, cut1

            # Randomly sample a start time for utterance 2
            cut2_start_time = random.uniform(0, cut1.duration / 2)
            cut2_end_time = add_durations(
                cut2_start_time, cut2.duration, sampling_rate=self.sampling_rate
            )
            mix_end_time = max(cut1.duration, cut2_end_time)
            total_length += mix_end_time
            total_overlap += min(cut2.duration, cut1.duration - cut2_start_time)

            if mix_end_time < window_size:
                continue

            # Pad utterance 1 on the right if needed
            cut1_padded = cut1.pad(duration=mix_end_time, preserve_id=True)
            # Pad utterance 2 on both sides if needed
            cut2_padded = cut2.pad(
                duration=cut2_end_time, direction="left", preserve_id=True
            ).pad(duration=mix_end_time, direction="right", preserve_id=True)

            num_windows = int(mix_end_time / window_size)

            cuts_windowed = (
                CutSet.from_cuts([cut1.mix(cut2, offset_other_by=cut2_start_time)])
                .cut_into_windows(window_size)
                .subset(first=num_windows)  # remove last window which may be shorter
            )

            if len(cuts_windowed) == 0:
                continue

            # Chunk the padded utterances into windows and add to batch
            cuts1_padded += (
                CutSet.from_cuts([cut1_padded])
                .cut_into_windows(window_size)
                .subset(first=num_windows)
            )
            cuts2_padded += (
                CutSet.from_cuts([cut2_padded])
                .cut_into_windows(window_size)
                .subset(first=num_windows)
            )

            # Apply augmentation transforms
            for transform in self.transforms:
                cuts_windowed = transform(cuts_windowed)
            mix_cuts += cuts_windowed

        # We could have more than batch_size cuts in the batch, so we need to
        # trim the batch to the correct size
        cuts1_padded = cuts1_padded.subset(first=self.batch_size)
        cuts2_padded = cuts2_padded.subset(first=self.batch_size)
        mix_cuts = mix_cuts.subset(first=self.batch_size)

        if self.extractor is not None:
            # Extract features for mixture and sources
            xs_feats, xs_lens = self.extractor(mix_cuts)
            y1_feats, _ = self.extractor(cuts1_padded)
            y2_feats, _ = self.extractor(cuts2_padded)
        else:
            xs_feats = torch.from_numpy(
                np.concatenate([c.load_audio() for c in mix_cuts], axis=0)
            )
            xs_lens = torch.tensor([c.duration for c in mix_cuts])
            y1_feats = torch.from_numpy(
                np.concatenate([c.load_audio() for c in cuts1_padded], axis=0)
            )
            y2_feats = torch.from_numpy(
                np.concatenate([c.load_audio() for c in cuts2_padded], axis=0)
            )

        return {
            "mix": xs_feats,
            "lens": xs_lens,
            "source1": y1_feats,
            "source2": y2_feats,
            "ovl": total_overlap / total_length,
        }
