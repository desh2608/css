import torch


class Stitcher:
    def __init__(self, stitching_config, sr=16000):
        self.eval_win = stitching_config["eval_win"]
        self.eval_hop = stitching_config["eval_hop"]
        self.fft_hop = stitching_config["hop_size"]
        self.sr = sr
        self.stitch_margin = int(
            (self.eval_win * 10 - self.eval_hop * 10) / 10 * self.sr / self.fft_hop
        )  # 16ms window

    def get_stitch(self, x, masks):

        """
        this method use mask as stitching rule
        x: original magnitude spectrogram features corresponding to each window (expects single channel)
        masks: mask for each window (2 sources + 1 noise)
        """
        PERM = []
        for n in range(len(masks) - 1):
            # first find the permutations for each segments
            past = masks[n][:, :, :-1].permute(2, 0, 1)  # 2 x F x T
            now = masks[n + 1][:, :, :-1].permute(2, 0, 1)  # 2 x F x T

            E_prev = past * torch.abs(x[n])  # 2 x F x T
            E_now = now * torch.abs(x[n + 1])

            # Calculate a similarity matrix.
            similarity_matrix = torch.zeros((2, 2))
            for i in range(2):
                for j in range(2):
                    d = (
                        E_prev[j, :, -self.stitch_margin :]
                        - E_now[i, :, : self.stitch_margin]
                    )
                    similarity_matrix[i, j] = -torch.sum(
                        torch.pow(torch.abs(d), 0.5)
                    )  # 0.5

            sim0 = similarity_matrix[0, 0] + similarity_matrix[1, 1]
            sim1 = similarity_matrix[0, 1] + similarity_matrix[1, 0]

            if sim0 >= sim1:
                perm = [0, 1]

            else:
                perm = [1, 0]
            PERM.append(perm)

        return PERM

    def get_connect(self, PERM, mask):
        state = 0
        N_M1 = [0]
        for i, item in enumerate(PERM):
            if item[0] == 1:
                state = 1 - state
            N_M1.append(state)

        res1 = []
        res2 = []
        noise = []

        # perm
        for i in range(len(N_M1)):
            if N_M1[i] == 0:
                res1.append(mask[i][:, :, 0])
                res2.append(mask[i][:, :, 1])
            else:
                res1.append(mask[i][:, :, 1])
                res2.append(mask[i][:, :, 0])
            noise.append(mask[i][:, :, 2])

        # winner-take-tall
        for i, (r1, r2, n) in enumerate(zip(res1, res2, noise)):
            m = torch.stack((r1, r2, n), dim=2)  # F x T x 3
            m_max = torch.amax(m, dim=2, keepdim=True)
            m = torch.where(m == m_max, m, torch.tensor(1e-4, dtype=torch.float32))
            res1[i] = m[:, :, 0]
            res2[i] = m[:, :, 1]
            noise[i] = m[:, :, 2]

        # Average the masks of the overlapping region.
        hop = int(self.eval_hop * self.sr / self.fft_hop)

        F, win = res1[0].shape

        all_L = int(hop * (len(mask) - 1) + win)

        res_1 = torch.zeros((F, all_L))
        res_2 = torch.zeros((F, all_L))
        res_noise = torch.zeros((F, all_L))
        indicator = torch.zeros((1, all_L))
        for i in range(len(mask)):

            wav = mask[i]
            st = hop * i
            if wav.shape[1] < win:
                en = st + wav.shape[1]
            else:
                en = st + win
            # need to normalize it
            res_1[:, st:en] += res1[i]
            res_2[:, st:en] += res2[i]
            res_noise[:, st:en] += noise[i]
            indicator[:, st:en] += 1
        indicator[indicator == 0] = 1
        return (res_1 / indicator, res_2 / indicator, res_noise / indicator)
