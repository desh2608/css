#!/usr/bin/env python

from pathlib import Path
import argparse
import yaml
import logging
import sys

import torch
import onnxruntime
import numpy as np

from pathlib import Path

from css.utils.audio_util import EgsReader, write_wav
from css.executor.separator import Separator
from css.executor.stitcher import Stitcher
from css.executor.beamformer import Beamformer

from lhotse.recipes.libricss import prepare_libricss

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def run(args):
    # prepare LibriCSS data
    manifests = prepare_libricss(args.corpus_dir)
    # egs reader
    egs_reader = EgsReader(manifests["recordings"])

    # Settings for onnxruntime.
    opts = onnxruntime.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    opts.use_deterministic_compute = True

    # Load the config
    exp_config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    sampling_rate = exp_config["sampling_rate"]

    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(exist_ok=True, parents=True)

    # Create class objects for separation, stitching, and beamforming
    separator = Separator(
        exp_config["separation"], sess_options=opts, sr=sampling_rate, device=device
    )
    stitcher = Stitcher(exp_config["stitching"])
    beamformer = Beamformer(exp_config["beamforming"])

    for key, egs in egs_reader:
        if key != "OV40_session1":
            continue
        logging.info(f"Processing utterance {key}...")
        mixed = egs["mix"]  # (D x T)

        # Apply separation to get masks. The masks here will be a list of masks, each
        # corresponding to one window
        window_masks, mag_specs = separator.separate(mixed)

        # Stitch window-level masks to get session-level masks
        mask_permutation = stitcher.get_stitch(mag_specs, window_masks)
        stitched_masks = stitcher.get_connect(mask_permutation, window_masks)

        # Apply beamforming using masks
        wav_ch0, wav_ch1 = beamformer.continuous_process(mixed, stitched_masks)

        # Write out the separated audio
        write_wav(dump_dir / f"{key}_ch0.wav", wav_ch0)
        write_wav(dump_dir / f"{key}_ch1.wav", wav_ch1)

    logging.info(f"Processed {len(egs_reader)} utterances")


if __name__ == "__main__":
    rootdir = Path(__file__).resolve().parents[0]
    default_cfg = rootdir.joinpath("conf/config.yaml")
    parser = argparse.ArgumentParser(
        description="Command to do speech separation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        metavar="<yaml-file>",
        default=str(default_cfg),
        help="CSS configuration file.",
    )
    parser.add_argument("--corpus-dir", type=str, help="Directory of LibriCSS corpus")
    parser.add_argument(
        "--num_spks", type=int, default=2, help="Number of the speakers"
    )
    parser.add_argument(
        "--dump-dir",
        type=str,
        default="sep",
        help="Directory to dump separated speakers",
    )
    parser.add_argument(
        "--gpu",
        type=bool,
        default=True,
        help="Use GPU for separation",
    )
    args = parser.parse_args()
    run(args)