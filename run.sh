#!/bin/bash
. ./path.sh

export MODEL_NAME=1ch_conformer_large

# Single GPU
# (
#   utils/queue-freegpu.pl \
#     -l "hostname=c*" \
#     --gpu 1 --mem 2G \
#     exp/log/run_libricss_${MODEL_NAME}.log \
#     python3 separate_libricss.py \
#       --config conf/config.yaml \
#       --corpus-dir /export/c01/corpora6/LibriCSS \
#       --dump-dir exp/libricss/out_$MODEL_NAME \
#       --backend onnx \
#       --num_spks 2
# )

# Multi-GPU (1 GPU per session)
(
for session in `seq 0 9`; do
    utils/queue-freegpu.pl \
      -l "hostname=c*" \
      --gpu 1 --mem 2G \
      exp/log/run_libricss_${MODEL_NAME}_${session}.log \
      python3 separate_libricss.py \
        --config conf/config.yaml \
        --corpus-dir /export/c01/corpora6/LibriCSS \
        --dump-dir exp/libricss/out_$MODEL_NAME \
        --backend onnx \
        --num_spks 2 \
        --session $session &
done
wait
)

