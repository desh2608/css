#!/bin/bash
# Copyright 2021  Johns Hopkins University (Author: Desh Raj)
# This is the top-level training script for the CSS model.
. ./path.sh

lr=0.001
warmup=15000
decay=1e-05
weight_decay=1e-02
batches_per_epoch=500
num_epochs=100
nj_init=2
nj_final=4
batchsize=32
num_workers=4
grad_thresh=5.0
seed=0
resume=
init=
exp_dir=exp/conformer_libri_360_convtasnet

. ./utils/parse_options.sh

set -euo pipefail

# Prepare the data in the form of Lhotse manifests.
if [ ! -f data/cuts_train-clean-360.json ]; then
  echo "Preparing the data..."
  lhotse prepare librispeech -p train-clean-360 -p dev-clean \
    /export/corpora5/LibriSpeech data

  # Prepare cut manifests for train and dev
  echo "Preparing cut manifests for train and dev..."
  lhotse cut simple -r data/recordings_train-clean-360.json -s data/supervisions_train-clean-360.json \
    data/cuts_train-clean-360.json
  lhotse cut simple -r data/recordings_dev-clean.json -s data/supervisions_dev-clean.json \
    data/cuts_dev-clean.json

  # Prepare RIRs and noise manifests (we use real RIRs with isotropic noises)
  echo "Preparing RIRs and noise manifests..."
  lhotse prepare rir-noise -p sim_rir -p iso_noise /export/c01/corpora6/RIRS_NOISES data/
  lhotse cut simple -r data/recordings_iso_noise.json data/cuts_iso_noise.json
fi

resume_opts=
if [ ! -z $resume ]; then
  resume_opts="--resume ${resume}"
fi 

# Note: Remove `--fp16` for CLSP grid
train_script="train.py ${resume_opts} \
  --gpu --fp16 \
  --expdir ${exp_dir} \
  --model ConvTasNet \
  --objective MSE \
  --dataset CSS \
  --batch-size ${batchsize} \
  --num-workers ${num_workers} \
  --warmup ${warmup} \
  --decay ${decay} \
  --weight-decay ${weight_decay} \
  --lr ${lr} \
  --optim adam \
  --batches-per-epoch ${batches_per_epoch} \
  --num-epochs 1 \
  --grad-thresh ${grad_thresh} \
  --train-manifests data/cuts_train-clean-360.json \
  --dev-manifests data/cuts_dev-clean.json \
  --rir-manifest data/recordings_sim_rir.json \
  --noise-manifest data/cuts_iso_noise.json
  "

# For COE grid
train_cmd="utils/queue-freegpu.pl --mem 12G --gpu 1 --config conf/gpu.conf"

train_parallel.sh ${resume_opts} \
  --cmd "$train_cmd" \
  --nj-init ${nj_init} \
  --nj-final ${nj_final} \
  --num-epochs ${num_epochs} \
  --seed ${seed} \
  "${train_script}" ${exp_dir}
