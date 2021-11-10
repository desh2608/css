import numpy as np
import torch
import math

from css.utils.model_util import get_model, to_numpy
from css.executor.feature import FeatureExtractor


class Separator:
    def __init__(self, separation_config, sr=16000, sess_options=None, device="cpu"):

        if separation_config["model_path"].endswith(".onnx"):
            assert (
                separation_config["batch_size"] == 1
            ), "ONNX models only support batch size 1"

        self.separator, device = get_model(
            separation_config["model_path"], sess_options, device
        )
        self.batch_size = separation_config["batch_size"]
        self.sr = sr
        self.device = device
        self.eval_win = int(separation_config["eval_win"] * sr)
        self.eval_hop = int(separation_config["eval_hop"] * sr)
        self.frame_length = separation_config["frame_length"] / sr
        self.frame_shift = separation_config["frame_shift"] / sr
        self.n_fft = separation_config["n_fft"]
        self.feature = FeatureExtractor()

    def separate(self, s):
        """
        s: waveform (1 x T), where 1 is number of channels and T is number of samples
        """
        assert s.shape[0] == 1, "Separator expects mono input"
        s = s.squeeze(dim=0).unfold(0, self.eval_win, self.eval_hop)  # B x T

        # Run separation in batches of segments
        masks = []
        mags = []
        n_batches = math.ceil(s.shape[0] / self.batch_size)
        for i in range(n_batches):
            batch_start = i * self.batch_size
            batch_end = min((i + 1) * self.batch_size, s.shape[0])
            s_batch = s[batch_start:batch_end]
            batch_masks, batch_mags = self.separate_one_batch(s_batch)
            masks.append(batch_masks)
            mags.append(batch_mags)

        # Concatenate the results
        masks = torch.cat(masks, dim=0)
        mags = torch.cat(mags, dim=0)
        return masks, mags

    def separate_one_batch(self, s):
        """
        s: waveform (B x T), where B is batch size and T is number of samples
        """
        # Extract STFT features
        mag_specs, f = self.feature.forward(s)

        # ONNX model expects input of shape (1, F, batch_size)
        B, F, T = f.shape
        f = f.reshape(1, F, B * T)
        ort_inputs = {self.separator.get_inputs()[0].name: to_numpy(f)}  # 1 x F x BT
        this_res = self.separator.run(None, ort_inputs)

        masks = torch.Tensor(
            np.stack((this_res[0], this_res[1], this_res[2]), axis=0)
        )  # (3, F, BT) The 3 masks correspond to spk1, spk2, noise
        masks = masks.reshape(3, B, F, T).permute(1, 2, 3, 0)  # (B, F, T, 3)
        masks = torch.clamp(masks, max=1.0)  # Cap the mask values at 1.0.
        return masks, mag_specs
