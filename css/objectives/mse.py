import torch

from asteroid.losses import pairwise_mse, PITLossWrapper


class MeanSquaredErrorPIT(torch.nn.Module):
    @staticmethod
    def add_args(parser):
        pass

    @classmethod
    def build_objective(cls, conf):
        return MeanSquaredErrorPIT()

    @classmethod
    def add_state_dict(cls, s1, s2, fraction, iteration=None):
        return s1

    def __init__(self):
        super(MeanSquaredErrorPIT, self).__init__()
        self.loss_fn = PITLossWrapper(pairwise_mse, pit_from="perm_avg")

    def forward(self, model, sample, device="cpu"):
        xs = sample["mix"].to(device)
        _, y_pred = model(xs)
        y_pred = y_pred[:, :-1, :]  # remove noise output

        y1_true, y2_true = sample["source1"].to(device), sample["source2"].to(device)
        y_true = torch.cat((y1_true, y2_true), dim=1)

        return self.loss_fn(y_pred, y_true)
