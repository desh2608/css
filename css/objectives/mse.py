import torch

from css.objectives.pit import pit_loss


class MeanSquaredError(torch.nn.Module):
    @classmethod
    def build_objective(cls, conf):
        return MeanSquaredError(reduction=conf["reduction"])

    @classmethod
    def add_state_dict(cls, s1, s2, fraction, iteration=None):
        return s1

    def __init__(self, reduction="mean"):
        super(MeanSquaredError, self).__init__()
        self.reduction = reduction

    def forward(self, model, sample, device="cpu", return_est=False):
        xs = sample["feats"].to(device)  # B x T x F
        mix = sample["mix"].to(device)  # B x T x F
        masks = model(xs)  # [B x T x F]

        # Apply masks to the input
        y_pred = torch.stack([mix * masks[i] for i in range(len(masks))], dim=1)
        B, N, T, F = y_pred.shape  # N = num_spk + num_noise

        y_true = sample["targets"].to(device)
        assert y_pred.shape == y_true.shape, f"{y_pred.shape} != {y_true.shape}"

        total_loss = 0
        for b in range(B):
            L = sample["len"][b]
            y_pred_b = y_pred[b, :-1, :L]
            y_true_b = y_true[b, :-1, :L]
            # PIT on speakers
            cur_loss = pit_loss(
                y_pred_b,
                y_true_b,
                axis=0,
                loss_fn=torch.nn.functional.mse_loss,
                reduction=self.reduction,
            )
            total_loss += cur_loss / L
            # Add noise MSE loss
            noise_loss = torch.nn.functional.mse_loss(
                y_pred[b, -1, :L], y_true[b, -1, :L], reduction=self.reduction
            )
            if torch.isfinite(noise_loss):
                total_loss += noise_loss / L

        total_loss /= B * N
        if not return_est:
            return total_loss
        else:
            return total_loss, (mix.cpu(), y_true.cpu(), y_pred.cpu())
