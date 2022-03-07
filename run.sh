#!/bin/bash
. ./path.sh

export MODEL_NAME=7ch_conformer_large_doa

# Multi-GPU (1 GPU per session) single channel
# (
# for session in `seq 0 9`; do
#     utils/queue-freegpu.pl \
#       -l "hostname=c*" \
#       --gpu 1 --mem 2G \
#       exp/libricss/log/out_${MODEL_NAME}_session${session}.log \
#       python scripts/python/separate_libricss.py \
#         --config conf/config_1ch.yaml \
#         --corpus-dir /export/c01/corpora6/LibriCSS \
#         --dump-dir exp/libricss/out_$MODEL_NAME \
#         --num_spks 2 \
#         --gpu \
#         --session $session &
# done
# wait
# )

# Multi-GPU (1 GPU per session) multi channel
(
for session in `seq 0 9`; do
    utils/queue-freegpu.pl \
      -l "hostname=c*" \
      --gpu 1 --mem 2G \
      exp/libricss/log/out_${MODEL_NAME}_session${session}.log \
      python scripts/python/separate_libricss.py \
        --config conf/config_7ch.yaml \
        --corpus-dir /export/c01/corpora6/LibriCSS \
        --dump-dir exp/libricss/out_$MODEL_NAME \
        --num_spks 2 \
        --gpu \
        --multi-channel \
        --session $session &
done
wait
)
