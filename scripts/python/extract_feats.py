#! /usr/bin/env python
# Copyright 2021  Johns Hopkins University (Author: Desh Raj)
# Apache 2.0

import yaml
import argparse
import json
import os
import logging
import itertools
import soundfile as sf

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from css.executor.feature import FeatureExtractor
import kaldi_native_io as kio

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class JsonSeparationDataset(Dataset):
    def __init__(self, mix_json):
        self.mix_dict = []
        with open(mix_json, "r") as f:
            for line in f:
                self.mix_dict.append(json.loads(line))

    def __len__(self):
        return len(self.mix_dict)

    def __getitem__(self, index):
        sample = self.mix_dict[index]
        return sample


def collater(batch):
    """
    Collate function for data loader
    """
    mix = [
        torch.tensor(sf.read(sample["output"], always_2d=True)[0], dtype=torch.float)
        for sample in batch
    ]
    lengths = torch.tensor([x.shape[0] for x in mix], dtype=torch.int)
    mix = pad_sequence(mix, batch_first=True,).transpose(
        1, 2
    )  # B x D x N
    targets = list(
        itertools.chain(
            *[
                [
                    torch.tensor(
                        sf.read(sample[key], always_2d=True)[0], dtype=torch.float
                    )
                    for key in ["source0", "source1", "noise"]
                ]
                for sample in batch
            ]
        )
    )
    targets = pad_sequence(targets, batch_first=True,).transpose(
        1, 2
    )  # 3B x D x N
    return mix, targets, lengths


def read_args():
    parser = argparse.ArgumentParser(
        description="Extract features from datasets in config file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to data JSONL file"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--dump_name", type=str, required=True, help="Path to dump features"
    )
    parser.add_argument("--job_id", type=int, required=False)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument(
        "--dl_batch_size",
        type=int,
        default=16,
        help="Batch size for data loader (only used when extracting on GPU)",
    )
    parser.add_argument(
        "--dl_num_workers",
        type=int,
        default=4,
        help="Number of workers for data loader (only used when extracting on GPU)",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = read_args()
    with open(args.config) as f:
        conf_dict = yaml.safe_load(f)

    logging.info("Creating feature extractor")
    feat = FeatureExtractor(
        frame_len=conf_dict["feature"]["frame_length"],
        frame_hop=conf_dict["feature"]["frame_shift"],
        ipd_index=conf_dict["feature"]["ipd"],
    )

    logging.info("Creating dataset from JSON")
    dataset = JsonSeparationDataset(args.data)
    N = len(dataset)

    logging.info("Creating writer objects")
    base_writer = "ark,scp,t:{args.dump_name}_{key}.{args.job_id}.ark,{args.dump_name}_{key}.{args.job_id}.scp"
    writer_mix = kio.FloatMatrixWriter(base_writer.format(**locals(), key="mix"))
    writer_feats = kio.FloatMatrixWriter(base_writer.format(**locals(), key="feats"))
    writer_src0 = kio.FloatMatrixWriter(base_writer.format(**locals(), key="src0"))
    writer_src1 = kio.FloatMatrixWriter(base_writer.format(**locals(), key="src1"))
    writer_noise = kio.FloatMatrixWriter(base_writer.format(**locals(), key="noise"))

    if args.gpu:
        dataloader = DataLoader(
            dataset,
            batch_size=args.dl_batch_size,
            num_workers=args.dl_num_workers,
            collate_fn=collater,
        )
        NB = len(dataloader)
        feat.cuda()
        idx = 1
        for i, batch in enumerate(dataloader):
            logging.info("Processing batch {}/{}".format(i + 1, NB))
            mix, targets, lengths = batch
            B = mix.shape[0]
            mix = mix.cuda()
            targets = targets.cuda()

            # Get lengths in frames
            nframes = torch.ceil(lengths / feat.forward_stft.stride).to(torch.int)

            # Feature extraction
            mix_stft, feats, _, _ = feat.forward(mix)
            targets_stft = feat.forward(targets)[0]

            mix_stft = mix_stft.transpose(1, 2)  # B x T x F
            targets_stft = torch.chunk(
                targets_stft.transpose(1, 2), B, 0
            )  # B x [3 x T x F]
            feats = feats.transpose(1, 2)  # B x T x *

            for b in range(B):
                l = nframes[b].item()
                # Convert to numpy array on CPU
                mix_b = mix_stft[b, :l].cpu().numpy()
                source0_b = targets_stft[b][0, :l].cpu().numpy()
                source1_b = targets_stft[b][1, :l].cpu().numpy()
                noise_b = targets_stft[b][2, :l].cpu().numpy()
                feats_b = feats[b, :l].cpu().numpy()

                utt_id = f"utt-{args.job_id:02d}-{idx:0{len(str(N))}d}"

                # Write output matrices
                writer_mix.write(utt_id, mix_b)
                writer_src0.write(utt_id, source0_b)
                writer_src1.write(utt_id, source1_b)
                writer_noise.write(utt_id, noise_b)
                writer_feats.write(utt_id, feats_b)
                idx += 1
    else:
        for idx, sample in enumerate(dataset):
            logging.info("Processing sample {}/{}".format(idx + 1, N))
            mix = torch.tensor(
                sf.read(sample["output"], always_2d=True)[0], dtype=torch.float
            ).T.unsqueeze(
                0
            )  # 1 x D x N
            targets = torch.stack(
                [
                    torch.tensor(
                        sf.read(sample[key], always_2d=True)[0].T, dtype=torch.float
                    )
                    for key in ["source0", "source1", "noise"]
                ],
                dim=0,
            )  # 3 x D x N

            # Feature extraction
            mix_stft, feats, _, _ = feat.forward(mix)
            targets_stft = feat.forward(targets)[0]

            mix_stft = mix_stft.transpose(1, 2)  # 1 x T x F
            targets_stft = torch.chunk(
                targets_stft.transpose(1, 2), 3, 0
            )  # 3 x [1 x T x F]
            feats = feats.transpose(1, 2)  # 1 x T x *

            # Convert to numpy array on CPU
            mix_b = mix_stft[0].cpu().numpy()
            source0_b = targets_stft[0][0].cpu().numpy()
            source1_b = targets_stft[1][0].cpu().numpy()
            noise_b = targets_stft[2][0].cpu().numpy()
            feats_b = feats[0].cpu().numpy()

            utt_id = f"utt-{args.job_id:02d}-{idx+1:0{len(str(N))}d}"

            # Write output matrices
            writer_mix.write(utt_id, mix_b)
            writer_src0.write(utt_id, source0_b)
            writer_src1.write(utt_id, source1_b)
            writer_noise.write(utt_id, noise_b)
            writer_feats.write(utt_id, feats_b)

    # Close writers
    writer_mix.close()
    writer_feats.close()
    writer_src0.close()
    writer_src1.close()
    writer_noise.close()
