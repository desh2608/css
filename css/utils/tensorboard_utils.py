from torchvision.utils import make_grid
from matplotlib import cm
import random
import torch


def colorize_tensors(tensors, cmap_name="viridis"):
    """
    tensors: B x 1 x H x W
    """
    cmap = cm.get_cmap(cmap_name)
    # # Normalize tensors between 0 and 1
    # tensors -= tensors.amin(dim=[2, 3], keepdim=True)
    # tensors /= tensors.amax(dim=[2, 3], keepdim=True)

    tensors = torch.from_numpy(cmap(tensors.detach().numpy()))  # B x 1 x H x W x 4
    # Reshape to B x 4 x H x W
    tensors = tensors.squeeze().permute(0, 3, 1, 2)
    return tensors


def make_grid_from_tensors(tensors, num_samples=4):
    """
    `tensors` is a tuple containing (mix, y_true, y_pred)
    mix: B x T x F
    y_true: B x 3 x T x F
    y_pred: B x 3 x T x F

    `num_samples` is the number of samples to show in the grid
    """
    mix, y_true, y_pred = tensors
    assert num_samples <= mix.shape[0]
    selected_idx = random.sample(range(mix.shape[0]), num_samples)

    # Select the indices for the tensors and make them in B x C x H x W shape
    mix = colorize_tensors(mix[selected_idx, None, ...].transpose(2, 3))
    y_true1 = colorize_tensors(y_true[selected_idx, 0:1, ...].transpose(2, 3))
    y_true2 = colorize_tensors(y_true[selected_idx, 1:2, ...].transpose(2, 3))
    y_pred1 = colorize_tensors(y_pred[selected_idx, 0:1, ...].transpose(2, 3))
    y_pred2 = colorize_tensors(y_pred[selected_idx, 1:2, ...].transpose(2, 3))
    return {
        "mix": make_grid(mix),
        "y_true1": make_grid(y_true1),
        "y_true2": make_grid(y_true2),
        "y_pred1": make_grid(y_pred1),
        "y_pred2": make_grid(y_pred2),
    }
