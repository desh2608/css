#!/bin/bash
# Copyright 2021  Johns Hopkins University (Author: Desh Raj)
# This is the top-level training script for the CSS model.
. ./path.sh

lr=0.0001
warmup=10000
decay=1e-05
weight_decay=1e-07
batches_per_epoch=500
num_epochs=50
nj_init=1
nj_final=1
batchsize=32
num_workers=4
grad_thresh=5.0
seed=0
resume=
init=
exp_dir=exp/conformer_libri_100

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
  lhotse prepare rir-noise -p real_rir -p iso_noise /export/c01/corpora6/RIRS_NOISES data/
  lhotse cut simple -r data/recordings_iso_noise.json data/cuts_iso_noise.json
fi

resume_opts=
if [ ! -z $resume ]; then
  resume_opts="--resume ${resume}"
fi 

train_script="train.py ${resume_opts} \
  --gpu \
  --expdir ${exp_dir} \
  --model Conformer \
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
  --train-cuts data/cuts_train-clean-100.json \
  --dev-cuts data/cuts_dev-clean.json
  --rir-recordings data/recordings_real_rir.json \
  --noise-cuts data/cuts_iso_noise.json
  "

train_cmd="utils/retry.pl utils/queue-freegpu.pl --mem 8G --gpu 1"

train_parallel.sh ${resume_opts} \
  --cmd "$train_cmd" \
  --nj-init ${nj_init} \
  --nj-final ${nj_final} \
  --num-epochs ${num_epochs} \
  --seed ${seed} \
  "${train_script}" ${exp_dir}
