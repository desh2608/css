from lhotse.recipes.libricss import prepare_libricss
import torch

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

        self.eval_win = beamforming_config["eval_win"]
        self.eval_hop = beamforming_config["eval_hop"]
        self.proceed_margin = beamforming_config["proceed_margin"]

        self.sr = sr
        self.mask_hop = int(self.eval_hop * self.sr / self.hop_length)
        self.mask_win = int(self.eval_win * self.sr / self.hop_length)

        self.rescale = rescale

    def continuous_process(self, X, masks):
        """
        Process in an block continuous way
        X: waveform D x T
        mask: tuple of (speech_mask1, speech_mask2, noise_mask)
        """

        D, T = X.shape
        mask_ch0, mask_ch1, noise_mask = masks

        nwin = int(T // int(self.eval_hop * self.sr))

        result_ch0 = torch.zeros((T,))
        result_ch1 = torch.zeros((T,))

        for i in range(nwin):

            st = int(self.eval_hop * self.sr * i)
            en = st + int(self.eval_win * self.sr) + 256

            if en > T:
                en = T

            mask_st = self.mask_hop * i
            mask_en = mask_st + self.mask_win

            if mask_en > mask_ch0.shape[1]:
                mask_en = mask_ch0.shape[1]

            this_wav = X[:, st:en]
            this_mask_ch0 = mask_ch0[:, mask_st:mask_en]
            this_mask_ch1 = mask_ch1[:, mask_st:mask_en]
            this_noise_mask = noise_mask[:, mask_st:mask_en]

            out_ch0 = self.process_one_wav(
                this_wav,
                this_mask_ch0,
                this_noise_mask,
            )
            out_ch1 = self.process_one_wav(
                this_wav,
                this_mask_ch1,
                this_noise_mask,
            )

            # deduplication by masking - give up recovering low-volume competitor
            S = torch.stack((out_ch0, out_ch1), dim=0)  # 2 x F x T
            S_pow = 10 * torch.log10(torch.sum(torch.abs(S) ** 2, dim=(1, 2)))
            if S_pow[0] - S_pow[1] > 15:
                S_abs = torch.abs(S)
                gain = torch.divide(S_abs, torch.amax(S_abs, dim=0, keepdim=True))
                S[1] = torch.multiply(torch.clamp(gain[1], min=10 ** (-40 / 20)), S[1])
            elif S_pow[1] - S_pow[0] > 15:
                S_abs = torch.abs(S)
                gain = torch.divide(S_abs, torch.amax(S_abs, dim=0, keepdim=True))
                S[0] = torch.multiply(torch.clamp(gain[0], min=10 ** (-40 / 20)), S[0])

            out_ch0, out_ch1 = S[0], S[1]
            wav1 = torch.istft(
                out_ch0,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                center=True,
                onesided=True,
                window=torch.hann_window(self.n_fft),
                return_complex=False,
            )
            wav1 = wav1[: this_wav.shape[-1]]
            wav2 = torch.istft(
                out_ch1,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                center=True,
                onesided=True,
                window=torch.hann_window(self.n_fft),
                return_complex=False,
            )
            wav2 = wav2[: this_wav.shape[-1]]

            if st == 0:
                result_ch0[st : st + self.proceed_margin * self.sr] += wav1[
                    : self.proceed_margin * self.sr
                ]
                result_ch1[st : st + self.proceed_margin * self.sr] += wav2[
                    : self.proceed_margin * self.sr
                ]
            elif en == T:
                result_ch0[
                    st
                    + self.proceed_margin * self.sr
                    - int(self.eval_hop * self.sr) : st
                    + wav1.shape[0]
                ] += wav1[
                    self.proceed_margin * self.sr - int(self.eval_hop * self.sr) :
                ]
                result_ch1[
                    st
                    + self.proceed_margin * self.sr
                    - int(self.eval_hop * self.sr) : st
                    + wav2.shape[0]
                ] += wav2[
                    self.proceed_margin * self.sr - int(self.eval_hop * self.sr) :
                ]
            else:
                result_ch0[
                    st
                    + self.proceed_margin * self.sr
                    - int(self.eval_hop * self.sr) : st
                    + self.proceed_margin * self.sr
                ] += wav1[
                    self.proceed_margin * self.sr
                    - int(self.eval_hop * self.sr) : self.proceed_margin * self.sr
                ]
                result_ch1[
                    st
                    + self.proceed_margin * self.sr
                    - int(self.eval_hop * self.sr) : st
                    + self.proceed_margin * self.sr
                ] += wav2[
                    self.proceed_margin * self.sr
                    - int(self.eval_hop * self.sr) : self.proceed_margin * self.sr
                ]

        result_ch0 = result_ch0 * 0.9 / torch.max(torch.abs(result_ch0))
        result_ch1 = result_ch1 * 0.9 / torch.max(torch.abs(result_ch1))
        return result_ch0.unsqueeze(dim=0), result_ch1.unsqueeze(dim=0)

    def process_one_wav(self, x, speech_mask, noise_mask):

        S = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=True,
            onesided=True,
            window=torch.hann_window(self.n_fft),
            return_complex=True,
        ).unsqueeze(
            0
        )  # 1 x D x F x T
        L = min([S.shape[-1], speech_mask.shape[-1]])

        target_scm = compute_scm(S[..., :L], speech_mask[..., :L].unsqueeze(0))
        distortion_scm = compute_scm(S[..., :L], noise_mask[..., :L].unsqueeze(0))

        result = (
            self.bf_module()
            .forward(mix=S[..., :L], target_scm=target_scm, noise_scm=distortion_scm)
            .squeeze(dim=1)
        )

        if self.rescale:
            result = self.normalize_scale(result, S[..., :L], speech_mask[..., :L])

        return result.squeeze(dim=0)  # F x T

    def normalize_scale(self, result_spec, ch0_spec, mask):
        masked = mask * ch0_spec

        masked_energy = torch.sqrt(torch.mean(torch.abs(masked) ** 2))
        mvdr_energy = torch.sqrt(torch.mean(torch.abs(result_spec) ** 2))

        result_spec = result_spec / mvdr_energy * masked_energy

        return result_spec
