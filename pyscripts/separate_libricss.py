#!/usr/local/env python

from pathlib import Path
import argparse
import yaml
import logging

import torch
import torchaudio

from pathlib import Path

from css.utils.audio_util import EgsReader
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
    recordings = manifests["recordings"]
    if args.session is not None:
        recordings = recordings.filter(lambda x: f"session{args.session}" in x.id)
    egs_reader = EgsReader(recordings, multi_channel=args.multi_channel)

    if args.backend == "onnx":
        import onnxruntime

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
        exp_config["separation"],
        sess_options=opts,
        sr=sampling_rate,
        device=device,
        backend=args.backend,
    )
    stitcher = Stitcher(exp_config["stitching"])
    beamformer = Beamformer(exp_config["beamforming"])

    for key, egs in egs_reader:
        logging.info(f"Processing utterance {key}...")
        mixed = egs["mix"]  # (D x T)

        # Apply separation to get masks. The masks here will be a list of masks, each
        # corresponding to one window.
        window_masks, mag_specs = separator.separate(mixed)

        # Stitch window-level masks to get session-level masks
        mask_permutation = stitcher.get_stitch(mag_specs, window_masks)
        stitched_masks = stitcher.get_connect(mask_permutation, window_masks)

        # Apply beamforming using masks
        wav_ch0, wav_ch1 = beamformer.continuous_process(mixed, stitched_masks)

        # Write out the separated audio
        torchaudio.save(dump_dir / f"{key}_0.wav", wav_ch0, sampling_rate)
        torchaudio.save(dump_dir / f"{key}_1.wav", wav_ch1, sampling_rate)

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
    parser.add_argument(
        "--train-config",
        metavar="<yaml-file>",
        default=None,
        help="CSS configuration file for training (will be used to initialize model).",
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
        "--backend", type=str, default="onnx", help="Backend to use (onnx | pytorch)"
    )
    parser.add_argument(
        "--gpu",
        default=False,
        action="store_true",
        help="Use GPU for separation",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Specify subset of sessions to process (useful for multi-GPU processing)",
    )
    parser.add_argument(
        "--multi-channel",
        default=False,
        action="store_true",
        help="Use multi-channel separation with MVDR beamformer",
    )
    args = parser.parse_args()
    run(args)
