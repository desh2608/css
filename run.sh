#!/bin/bash
. ./path.sh

export MODEL_NAME=1ch_conformer_base

(
  kaldi_utils/queue-freegpu.pl \
    -l "hostname=c*" \
    --gpu 1 --mem 2G \
    exp/log/run_libricss_${MODEL_NAME}.log \
    python3 separate_libricss.py \
      --config conf/config.yaml \
      --corpus-dir /export/c01/corpora6/LibriCSS \
      --dump-dir exp/libricss/out_$MODEL_NAME \
      --num_spks 2
)

