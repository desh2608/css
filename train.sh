#!/bin/bash
# Copyright 2021  Johns Hopkins University (Author: Desh Raj)
# This is the top-level training script for the CSS model.
. ./path.sh

num_epochs=20
seed=0
resume=
init=
exp_dir=exp/train_blstm_librimix
config_file=conf/train_blstm_7ch.yaml

# Command for training
train_cmd="queue-freegpu.pl --config conf/gpu.conf --gpu 2 --mem 8G"

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

$train_cmd $exp_dir/train.log \
    python train.py ${resume_opts} ${init_opts} \
    --gpu \
    --world-size 2 \
    --config ${config_file} \
    --expdir ${exp_dir} \
    --num-epochs ${num_epochs} \
    --seed ${seed}
