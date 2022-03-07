#!/bin/bash
# Copyright 2021  Johns Hopkins University (Author: Desh Raj)
# This is the top-level training script for the CSS model.
. ./path.sh

stage=0
lr=0.0001
warmup=20000
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
exp_dir=exp/train_conformer_librimix

. ./utils/parse_options.sh

corpus_dir=/export/c07/draj/sim-LibriMix

set -euo pipefail

# Prepare the data in the form of Lhotse manifests.
if [ $stage -le 0 ]; then
  echo "Preparing the data..."
  
fi

resume_opts=
if [ ! -z $resume ]; then
  resume_opts="--resume ${resume}"
fi 

train_script="train.py ${resume_opts} \
  --gpu --debug \
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
  --grad-thresh ${grad_thresh}
  "

train_cmd="utils/queue-freegpu.pl --mem 12G --gpu 1 -l 'hostname=c*"

train_parallel.sh ${resume_opts} \
  --cmd "$train_cmd" \
  --nj-init ${nj_init} \
  --nj-final ${nj_final} \
  --num-epochs ${num_epochs} \
  --seed ${seed} \
  "${train_script}" ${exp_dir}
