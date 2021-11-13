import torch
import math

from asteroid.dsp.beamforming import compute_scm


class Beamformer:
    def __init__(self, beamforming_config, sr=16000, rescale=True):

        self.type = beamforming_config["type"]
        try:
            self.bf_module = getattr(
                __import__("asteroid.dsp.beamforming", fromlist=[self.type]), self.type
            )
        except ImportError:
            raise ValueError(f"Beamformer type '{self.type}' not found")

        self.n_fft = beamforming_config["n_fft"]
        self.hop_length = beamforming_config["hop_size"]

        self.eval_win = int(beamforming_config["eval_win"] * sr)
        self.eval_hop = int(beamforming_config["eval_hop"] * sr)
        self.proceed_margin = beamforming_config["proceed_margin"] * sr

        self.batch_size = beamforming_config["batch_size"]
        self.sr = sr
        self.mask_hop = int(self.eval_hop / self.hop_length)
        self.mask_win = int(self.eval_win / self.hop_length)

        self.rescale = rescale

    def continuous_process(self, X, masks):
        """
        Process in an block continuous way
        X: waveform D x T
        mask: tuple of (speech_mask1, speech_mask2, noise_mask)
        """

        D, T = X.shape
        mask_ch0, mask_ch1, noise_mask = masks

        X = X.squeeze(dim=0).unfold(0, self.eval_win + 256, self.eval_hop)  # B x T
        mask_ch0 = mask_ch0.unfold(1, self.mask_win, self.mask_hop).permute(1, 0, 2)
        mask_ch1 = mask_ch1.unfold(1, self.mask_win, self.mask_hop).permute(1, 0, 2)
        noise_mask = noise_mask.unfold(1, self.mask_win, self.mask_hop).permute(1, 0, 2)

        n_batches = math.ceil(X.shape[0] / self.batch_size)

        result_ch0 = torch.zeros(T)
        result_ch1 = torch.zeros(T)

        for i in range(n_batches):

            batch_start = i * self.batch_size
            batch_end = min((i + 1) * self.batch_size, X.shape[0]) + 1

            X_batch = X[batch_start:batch_end]
            mask_ch0_batch = mask_ch0[batch_start:batch_end]
            mask_ch1_batch = mask_ch1[batch_start:batch_end]
            noise_mask_batch = noise_mask[batch_start:batch_end]

            out_ch0 = self.process_one_batch(X_batch, mask_ch0_batch, noise_mask_batch)
            out_ch1 = self.process_one_batch(X_batch, mask_ch1_batch, noise_mask_batch)

            # deduplication by masking - give up recovering low-volume competitor
            S = torch.stack((out_ch0, out_ch1), dim=1)  # B x 2 x F x T
            S_pow = 10 * torch.log10(torch.sum(torch.abs(S) ** 2, dim=(2, 3)))
            S_abs = torch.abs(S)
            gain = torch.divide(S_abs, torch.amax(S_abs, dim=1, keepdim=True))
            for b in range(S.shape[0]):  # Apply deduplication to each chunk in batch
                if S_pow[b, 0] - S_pow[b, 1] > 15:
                    S[b, 1] = torch.multiply(
                        torch.clamp(gain[b, 1], min=10 ** (-40 / 20)), S[b, 1]
                    )
                elif S_pow[b, 1] - S_pow[b, 0] > 15:
                    S[b, 0] = torch.multiply(
                        torch.clamp(gain[b, 0], min=10 ** (-40 / 20)), S[b, 0]
                    )

            out_ch0, out_ch1 = S[:, 0, ...], S[:, 1, ...]
            wav1 = torch.istft(
                out_ch0,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                center=True,
                onesided=True,
                window=torch.hann_window(self.n_fft),
                return_complex=False,
            )
            wav1 = wav1[:, : X_batch.shape[-1]]
            wav2 = torch.istft(
                out_ch1,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                center=True,
                onesided=True,
                window=torch.hann_window(self.n_fft),
                return_complex=False,
            )
            wav2 = wav2[:, : X_batch.shape[-1]]

            st = self.eval_hop * batch_start
            en = self.eval_hop * batch_end
            # fmt: off
            for b in range(len(wav1)):
                if st == 0:
                    result_ch0[st : st + self.proceed_margin] += wav1[b, : self.proceed_margin]
                    result_ch1[st : st + self.proceed_margin] += wav2[b, : self.proceed_margin]
                elif en == T:
                    result_ch0[st + self.proceed_margin - self.eval_hop : st + wav1.shape[1]] += wav1[b, self.proceed_margin - self.eval_hop :]
                    result_ch1[st + self.proceed_margin - self.eval_hop : st + wav2.shape[1]] += wav2[b, self.proceed_margin - self.eval_hop :]
                else:
                    result_ch0[st + self.proceed_margin - self.eval_hop : st + self.proceed_margin] += wav1[b, self.proceed_margin - self.eval_hop : self.proceed_margin]
                    result_ch1[st + self.proceed_margin - self.eval_hop : st + self.proceed_margin] += wav2[b, self.proceed_margin - self.eval_hop : self.proceed_margin]
                st += self.eval_hop
                en += self.eval_hop
            # fmt: on

        # Normalize scale
        result_ch0 = result_ch0 * 0.9 / torch.max(torch.abs(result_ch0))
        result_ch1 = result_ch1 * 0.9 / torch.max(torch.abs(result_ch1))
        return result_ch0.unsqueeze(dim=0), result_ch1.unsqueeze(dim=0)

    def process_one_batch(self, x, speech_mask, noise_mask):
        """
        Process one batch of wav chunks.
        x: B x N
        speech_mask: B x F X T
        noise_mask: B x F x T
        Returns: beamformed wav chunks (B x F x T)
        """
        S = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=True,
            onesided=True,
            window=torch.hann_window(self.n_fft),
            return_complex=True,
        ).unsqueeze(
            1
        )  # B x D x F x T
        L = min([S.shape[-1], speech_mask.shape[-1]])

        target_scm = compute_scm(S[..., :L], speech_mask[..., :L].unsqueeze(1))
        distortion_scm = compute_scm(S[..., :L], noise_mask[..., :L].unsqueeze(1))

        result = (
            self.bf_module()
            .forward(mix=S[..., :L], target_scm=target_scm, noise_scm=distortion_scm)
            .squeeze(dim=1)
        )

        if self.rescale:
            result = self.normalize_scale(result, S[..., :L], speech_mask[..., :L])

        return result  # B x F x T

    def normalize_scale(self, result_spec, ch0_spec, mask):
        """
        Normalize scale of the result.
        result_spec: B x F x T
        ch0_spec: B x D x F x T
        mask: B x F x T
        """
        ch0_spec = ch0_spec.squeeze(dim=1)  # B x F x T
        masked = mask * ch0_spec  # B x F x T

        masked_energy = torch.sqrt(
            torch.mean(torch.abs(masked) ** 2, dim=(1, 2), keepdim=True)
        )
        mvdr_energy = torch.sqrt(
            torch.mean(torch.abs(result_spec) ** 2, dim=(1, 2), keepdim=True)
        )

        result_spec = result_spec / mvdr_energy * masked_energy

        return result_spec
