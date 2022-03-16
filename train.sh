#!/bin/bash
# Copyright 2021  Johns Hopkins University (Author: Desh Raj)
# This is the top-level training script for the CSS model.
. ./path.sh

stage=0
num_epochs=20
seed=0
resume=
init=
nj_init=1
nj_final=4
exp_dir=exp/debug_conformer_librimix
config_file=conf/train_conformer_7ch.yaml

# Command for training
train_cmd="queue-freegpu.pl --config conf/gpu.conf --gpu 1 --mem 6G"

. ./utils/parse_options.sh

set -euo pipefail

resume_opts=
if [ ! -z $resume ]; then
  resume_opts="--resume ${resume}"
fi

init_opts=
if [ ! -z $init ]; then
  init_opts="--init ${init}"
fi

train_script="train.py ${resume_opts} \
  --gpu \
  --config ${config_file} \
  --expdir ${exp_dir} \
  --num-epochs 1 \
"

echo $train_script

train_parallel.sh ${resume_opts} ${init_opts} \
  --cmd "$train_cmd" \
  --nj-init ${nj_init} \
  --nj-final ${nj_final} \
  --num-epochs ${num_epochs} \
  --seed ${seed} \
  "${train_script}" ${exp_dir}
