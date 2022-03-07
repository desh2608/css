import numpy as np
import torch
import math

from css.utils.model_util import get_onnx_model, to_numpy
from css.executor.feature import FeatureExtractor


class Separator:
    def __init__(
        self,
        separation_config,
        sr=16000,
        sess_options=None,
        device="cpu",
        backend="onnx",
        lowcut=80,
        highcut=2000,
    ):

        if backend == "onnx":
            assert (
                separation_config["batch_size"] == 1
            ), "ONNX models only support batch size 1"

        if backend == "onnx":
            self.separator, device = get_onnx_model(
                separation_config["model_path"], sess_options, device
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self.batch_size = separation_config["batch_size"]
        self.device = device

        # Config options for CSS windowing
        self.eval_win = int(separation_config["eval_win"] * sr)
        self.eval_hop = int(separation_config["eval_hop"] * sr)

        # Config options for STFT features
        self.sr = sr
        self.frame_length = separation_config["frame_length"]
        self.frame_shift = separation_config["frame_shift"]
        self.ipd = separation_config.get("ipd", None)
        self.feature = FeatureExtractor(
            frame_len=self.frame_length, frame_hop=self.frame_shift, ipd_index=self.ipd
        )

        # Merge based on DOA
        self.merge = separation_config["merge"]
        self.merge_threshold = separation_config["merge_threshold"]

        if self.merge:
            self.steervec, self.angles = self.steervec_7ch(
                nfreqs=257, nvecs=30, samplerate=sr, return_angles=True
            )

            # Determine the low-cut and high-cut indeces.
            freq_step = (sr // 2) / (257 - 1)
            self._lo = int(math.floor(lowcut / freq_step))
            self._hi = int(math.ceil(highcut / freq_step))

    def separate(self, s):
        """
        s: waveform (D x T), where D is number of channels and T is number of samples
        """
        s = s.unfold(-1, self.eval_win + 256, self.eval_hop)  # D x B x T

        # Run separation in batches of segments
        masks = []
        mags = []
        n_batches = math.ceil(s.shape[1] / self.batch_size)
        for i in range(n_batches):
            batch_start = i * self.batch_size
            batch_end = min((i + 1) * self.batch_size, s.shape[1])
            s_batch = s[:, batch_start:batch_end, :].permute(1, 0, 2)
            batch_masks, batch_mags = self.separate_one_batch(s_batch)
            masks.append(batch_masks)
            mags.append(batch_mags)

        # Concatenate the results
        masks = torch.cat(masks, dim=0)
        mags = torch.cat(mags, dim=0)
        return masks, mags

    def separate_one_batch(self, s):
        """
        s: waveform (B x D x T), where B is batch size, D is #channels, and T is number of samples
        """
        # Extract STFT features
        B, D, T = s.shape
        mag_specs, f, re, im = self.feature.forward(s)

        # ONNX model expects input of shape (1, F, batch_size)
        B, F, T = f.shape
        f = f.reshape(1, F, B * T)
        ort_inputs = {self.separator.get_inputs()[0].name: to_numpy(f)}  # 1 x F x BT
        this_res = self.separator.run(None, ort_inputs)

        masks = torch.Tensor(
            np.stack((this_res[0], this_res[1], this_res[2]), axis=0)
        )  # (3, F, BT) The 3 masks correspond to spk1, spk2, noise
        masks = masks.reshape(3, B, F // D, T).permute(1, 2, 3, 0)  # (B, F, T, 3)
        masks = torch.clamp(masks, max=1.0)  # Cap the mask values at 1.0.

        if self.merge:
            spec = re + 1j * im
            masks[:, :, :, :2] = self.angle_merge(
                spec, masks[:, :, :, :2], thresh=self.merge_threshold
            )
        return masks, mag_specs

    def steervec_7ch(
        self,
        nfreqs,
        *,
        nvecs,
        radius=0.0425,
        sndvelocity=340,
        samplerate=16000,
        return_angles=False,
        reference=0,
        inverse_shift=False,
    ):
        # Get angles of arrival.
        angles = torch.tensor([2 * math.pi * k / nvecs for k in range(nvecs)])

        # Generage time delays.
        distances = radius * torch.stack(
            (
                torch.zeros(nvecs),
                torch.cos(angles + math.pi / 6),
                torch.cos(angles - math.pi / 6),
                torch.cos(angles - math.pi / 2),
                torch.cos(angles - 5 * math.pi / 6),
                torch.cos(angles + 5 * math.pi / 6),
                torch.cos(angles + math.pi / 2),
            ),
            dim=1,
        )

        if reference != 0:
            distances -= distances[:, reference, None]

        deltas = distances / sndvelocity * samplerate

        # Generate steering vectors.
        steervecs = []
        for f in range(nfreqs):
            if inverse_shift:
                steervecs.append(torch.exp(-1j * deltas * math.pi * f / (nfreqs - 1)))
            else:
                steervecs.append(torch.exp(1j * deltas * math.pi * f / (nfreqs - 1)))

        steervecs = torch.stack(steervecs)

        # Normalize the SVs.
        steervecs /= math.sqrt(7)

        if return_angles:
            return steervecs, angles * 360 / (2 * math.pi)
        else:
            return steervecs

    def angle_merge(self, X, mask, thresh=16, mask_binarize_thres=0.5, compression=0.5):

        batch, nfreqs, nframes, noutputs = mask.shape
        batch, nchannels, _, _ = X.shape

        # Binarize the masks.
        binmask = (mask > mask_binarize_thres).float()

        # Calculate the likelihoods.
        likelihoods = self.doa_likelihood(
            X,
            self.steervec,
            binmask,
            compression=compression,
            epsilon=1e-12,
            lowcut=self._lo,
            highcut=self._hi,
            softmax=False,
        )

        DOAs = [self.angles[i] for i in torch.argmax(likelihoods, dim=-1)]

        # When the DOA difference is self._tolerance or less, pick the primary mask.
        new_mask = torch.clone(mask)
        for b in range(batch):
            DOA = DOAs[b]
            if min((DOA[0] - DOA[1]) % 360, (DOA[1] - DOA[0]) % 360) <= thresh:
                masked_magnitude = binmask[b] * torch.abs(
                    X[b, 0:1, :, :].permute(1, 2, 0)
                )
                masked_energy = torch.sum(masked_magnitude[b], dim=(0, 1))
                _channel_to_kill = torch.argmin(masked_energy)
                new_mask[b, ..., _channel_to_kill] = (
                    torch.ones((nfreqs, nframes)) * 1e-12
                )  # very small values
        return new_mask

    def doa_likelihood(
        self,
        X,
        steervec,
        mask,
        compression=1 / 3,
        epsilon=1e-12,
        lowcut=0,
        highcut=None,
        softmax=True,
    ):
        batch, nchannels, nfreqs, nframes = X.shape
        batch, nfreqs, nframes, noutputs = mask.shape
        nfreqs, nangles, nchannels = steervec.shape

        Xpow = torch.abs(
            torch.einsum(
                "bmft,bmft->bft",
                X[:, :, lowcut:highcut, :],
                X[:, :, lowcut:highcut, :].conj(),
            )
        )
        XHpow = (
            torch.abs(
                torch.einsum(
                    "bmft,fam->bfta",
                    X[:, :, lowcut:highcut, :].conj(),
                    steervec[lowcut:highcut],
                )
            )
            ** 2
        )

        if compression <= 0:
            tfwise_likelihood = -torch.log(Xpow[..., None] - XHpow / (1 + epsilon))
        else:
            tfwise_likelihood = -torch.pow(
                Xpow[..., None] - XHpow / (1 + epsilon), compression
            )

        ret = torch.einsum("bftc,bfta->bca", mask[:, lowcut:highcut], tfwise_likelihood)

        if softmax:
            ret -= torch.amax(ret, dim=-1, keepdim=True)
            ret = torch.exp(
                ret - torch.log(torch.sum(torch.exp(ret), dim=-1, keepdim=True))
            )

        return ret
