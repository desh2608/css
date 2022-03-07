# css
PyTorch implementation of Continuous Speech Separation

>>> This repository is under development.

This repository contains training and inference pipelines for continuous speech
separation on long recordings, similar to that in the [LibriCSS paper](https://arxiv.org/abs/2001.11482).

Salient features:

* On-the-fly training data creation using [Lhotse](https://github.com/lhotse-speech/lhotse).
* Conformer and BLSTM encoders.
* Multi-node training is based on Matthew Wiesner's [nnet_pytorch](https://github.com/m-wiesner/nnet_pytorch). It is a naive form of multi-node training
suitable for clusters where DDP may be slowed down due to an I/O bottleneck.