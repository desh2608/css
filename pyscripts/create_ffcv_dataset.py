#! /usr/bin/env python
# Copyright 2021  Johns Hopkins University (Author: Desh Raj)
# Apache 2.0

import yaml
import argparse
import logging

import numpy as np

from css.datasets.feature_separation_dataset import FeatureSeparationDataset

from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def read_args():
    parser = argparse.ArgumentParser(
        description="Extract features from datasets in config file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to data containing JSONL files",
    )
    parser.add_argument(
        "--beton-file", type=str, required=True, help="Path to out BETON file"
    )
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = read_args()
    with open(args.config) as f:
        conf_dict = yaml.safe_load(f)

    logging.info("Loading feature dataset")
    feature_dataset = FeatureSeparationDataset(
        args.data_dir, as_dict=False, to_tensor=False
    )

    logging.info("Creating FFCV Writer")
    writer = DatasetWriter(
        args.beton_file,
        {
            "mix": NDArrayField(shape=(d,), dtype=np.dtype("float32")),
            "src0": NDArrayField(shape=(d,), dtype=np.dtype("float32")),
            "src1": NDArrayField(shape=(d,), dtype=np.dtype("float32")),
            "noise": NDArrayField(shape=(d,), dtype=np.dtype("float32")),
            "feats": NDArrayField(shape=(d,), dtype=np.dtype("float32")),
        },
        num_workers=args.num_workers,
    )

    logging.info("Writing FFCV file")
    writer.from_indexed_dataset(feature_dataset)
