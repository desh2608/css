import torch

from css.objectives.pit import pit_loss


class MeanSquaredError(torch.nn.Module):
    @staticmethod
    def add_args(parser):
        pass

    @classmethod
    def build_objective(cls, conf):
        return MeanSquaredError()

    @classmethod
    def add_state_dict(cls, s1, s2, fraction, iteration=None):
        return s1

    def __init__(self):
        super(MeanSquaredError, self).__init__()

    def forward(self, model, sample, device="cpu"):
        xs = sample["mix"].to(device)
        y_pred = model(xs)

        B, N, T, F = y_pred.shape
        y1_true, y2_true = sample["source1"].to(device), sample["source2"].to(device)
        y_true = torch.stack((y1_true, y2_true), dim=1)
        assert y_pred.shape == y_true.shape, f"{y_pred.shape} != {y_true.shape}"

        total_loss = 0
        for b in range(B):
            total_loss += pit_loss(
                y_pred[b], y_true[b], axis=0, loss_fn=torch.nn.functional.mse_loss
            )

        return total_loss / B
