#!/bin/bash
. ./path.sh

export MODEL_NAME=1ch_conformer_large

(
  utils/queue-freegpu.pl \
    -l "hostname=c*" \
    --gpu 1 --mem 2G \
    exp/log/run.log \
    python3 separate_libricss.py \
      --checkpoint checkpoints/$MODEL_NAME \
      --corpus-dir /export/c01/corpora6/LibriCSS \
      --dump-dir separated_speech/monaural/utterances_with_$MODEL_NAME \
      --device-id ${CUDA_VISIBLE_DEVICES} \
      --num_spks 2
)

