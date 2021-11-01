# Continuous Speech Separation with Conformer

## Introduction

This repository uses the Conformer-based CSS, but adds Lhotse for data preparation, so
that the separation can be performed for any general dataset whose recipe is available
in Lhotse. We provide examples for LibriCSS, AMI, and AISHELL-4.

## Environment
python 3.6.9, torch 1.7.1, lhotse 0.11.0

## Get Started
1. Set up the environment with PyTorch and Lhotse.

2. Download the Conformer separation models.

    ```bash
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1OlTbEvxYUoqWIHfeAXCftL9srbWUo4I1' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1OlTbEvxYUoqWIHfeAXCftL9srbWUo4I1" -O checkpoints.zip && rm -rf /tmp/cookies.txt && unzip checkpoints.zip && rm checkpoints.zip
    ```

3. Run the separation.

    3.1  Single-channel separation
    
    ```bash
    export MODEL_NAME=1ch_conformer_base
    python3 separate_libricss.py \
        --checkpoint checkpoints/$MODEL_NAME \
        --corpus-dir /export/corpora/LibriCSS \
        --dump-dir separated_speech/monaural/utterances_with_$MODEL_NAME \
        --device-id 0 \
        --num_spks 2
    ```
        
    The separated speech can be found in the directory 'separated_speech/monaural/utterances_with_$MODEL_NAME'

## Credits
Please cite the original work from Chen et al.
```
@inproceedings{CSS_with_Conformer,
  title={Continuous speech separation with conformer},
  author={Chen, Sanyuan and Wu, Yu and Chen, Zhuo and Wu, Jian and Li, Jinyu and Yoshioka, Takuya and Wang, Chengyi and Liu, Shujie and Zhou, Ming},
  booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={5749--5753},
  year={2021},
  organization={IEEE}
}
```