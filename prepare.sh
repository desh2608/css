#!/bin/bash
# Copyright 2021  Johns Hopkins University (Author: Desh Raj)
# This script extracts STFT and IPD features for the whole data in advance so that we
# don't perform repeated computations and model training can be sped up.
. ./path.sh

stage=0
config_file=conf/train_conformer_7ch.yaml
train_json=/export/c07/draj/sim-LibriMix/data/SimLibriUttmix-train/mixlog.json
valid_json=/export/c07/draj/sim-LibriMix/data/SimLibriUttmix-dev/mixlog.json
dump_dir=dump
nj=12

. ./utils/parse_options.sh

set -euo pipefail

cmd="queue.pl --mem 2G -l hostname=c[01]*\&!c06*"
cuda_cmd="queue-freegpu.pl --config conf/gpu.conf --gpu 1 --mem 4G"

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

mkdir -p data/{train,valid}

if [ $stage -le 0 ]; then
  log "Creating JSONL files from JSON"
  jq -c '.[]' $train_json > data/train/data.jsonl
  jq -c '.[]' $valid_json > data/valid/data.jsonl
fi

if [ $stage -le 1 ]; then
  log "Splitting data into parts for parallel processing"
  for part in train valid; do
    mkdir -p data/$part/split${nj}
    split --number l/${nj} --numeric-suffixes=1 --additional-suffix=".jsonl" \
      data/${part}/data.jsonl data/${part}/split${nj}/data.
    # Remove 0 padding from filenames
    rename 's/\.0/\./' data/${part}/split${nj}/*.jsonl
  done
fi

if [ $stage -le 2 ]; then
  log "Creating dump dir to spread the features over various machines"
  if [[  $(hostname -f) ==  *.clsp.jhu.edu ]]; then
    utils/create_split_dir.pl /export/b{11,12,14,15}/$USER/css-data/dump/storage \
     $dump_dir/storage
  fi
fi

if [ $stage -le 3 ]; then
  for part in train valid; do
    
    for n in $(seq $nj); do
      # the next command does nothing unless $dump_dir/storage/ exists, see
      # utils/create_data_link.pl for more info.
      utils/create_data_link.pl $dump_dir/${part}_raw_mix.$n.ark
      utils/create_data_link.pl $dump_dir/${part}_raw_feats.$n.ark
      utils/create_data_link.pl $dump_dir/${part}_raw_src0.$n.ark
      utils/create_data_link.pl $dump_dir/${part}_raw_src1.$n.ark
      utils/create_data_link.pl $dump_dir/${part}_raw_noise.$n.ark
    done
    
    log "Extracting features for ${part}"
    $cuda_cmd JOB=1:${nj} exp/log/prepare_${part}/extract.JOB.log \
      python pyscripts/extract_feats.py \
        --data data/$part/split${nj}/data.JOB.jsonl \
        --config $config_file \
        --dump_name $dump_dir/${part}_raw \
        --job_id JOB \
        --gpu \
        --dl_batch_size 16 \
        --dl_num_workers 4
  done
fi

if [ $stage -le 4 ]; then
  log "concatenating scp files"
  for part in train valid; do
    for key in mix feats src0 src1 noise; do
      for n in $(seq $nj); do
        cat $dump_dir/${part}_raw_${key}.${n}.scp
      done > data/$part/$key.scp
    done
  done
fi

if [ $stage -le 5 ]; then
  log "Creating FFCV dataset"
  for part in train valid; do
    python pyscripts/create_ffcv_dataset.py \
      --data_dir data/$part \
      --out_path $dump_dir \
      --num_workers 8
  done
fi
