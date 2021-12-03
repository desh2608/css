import torch
import itertools

# This function is borrowed from: https://github.com/fgnt/padertorch/blob/master/padertorch/ops/losses/source_separation.py
def pit_loss(
    estimate: torch.Tensor,
    target: torch.Tensor,
    axis: int,
    loss_fn=torch.nn.functional.mse_loss,
    return_permutation: bool = False,
):
    """
    Permutation invariant loss function. Calls `loss_fn` on every possible
    permutation between `estimate`s and `target`s and returns the minimum
    loss among them. The tensors are permuted along `axis`.
    Does not support batch dimension. Does not support PackedSequence.
    Args:
        estimate: Padded sequence. The speaker axis is specified with `axis`,
            so the default shape is (T, K, F)
        target: Padded sequence with the same shape as `estimate` (defaults
            to (T, K, F))
        loss_fn: Loss function to apply on each permutation. It must accept two
            arguments (estimate and target) of the same shape that this function
            receives the arguments.
        axis: Speaker axis K. The permutation is applied along this axis. axis=-2
            and an input shape of (T, K, F) corresponds to the old default
            behaviour.
        return_permutation: If `True`, this function returns the permutation
            that minimizes the loss along with the minimal loss otherwise it
            only returns the loss.
    Examples:
        >>> T, K, F = 4, 2, 5
        >>> estimate, target = torch.ones(T, K, F), torch.zeros(T, K, F)
        >>> pit_loss(estimate, target, 1)
        tensor(1.)
        >>> T, K, F = 4, 2, 5
        >>> estimate, target = torch.ones(T, K, F), torch.zeros(T, F, dtype=torch.int64)
        >>> pit_loss(estimate, target, 1, loss_fn=torch.nn.functional.cross_entropy)
        tensor(0.6931)
        >>> T, K, F = 4, 2, 5
        >>> estimate, target = torch.ones(K, F, T), torch.zeros(K, F, T)
        >>> pit_loss(estimate, target, 0)
        tensor(1.)
        >>> T, K, F = 4, 2, 5
        >>> estimate = torch.stack([torch.ones(F, T), torch.zeros(F, T)])
        >>> target = estimate[(1, 0), :, :]
        >>> pit_loss(estimate, target, axis=0, return_permutation=True)
        (tensor(0.), (1, 0))
        >>> K = 5
        >>> estimate, target = torch.ones(K), torch.zeros(K)
        >>> pit_loss(estimate, target, axis=0)
        tensor(1.)
        >>> A, B, K, C, F = 4, 5, 3, 100, 128
        >>> estimate, target = torch.ones(A, B, K, C, F), torch.zeros(A, B, K, C, F)
        >>> pit_loss(estimate, target, axis=-3)
        tensor(1.)
    """
    sources = estimate.size()[axis]
    assert (
        sources < 30
    ), f"Are you sure? sources={sources}, estimate.shape={estimate.shape}, target.shape={target.shape}"

    if loss_fn in [torch.nn.functional.cross_entropy]:
        assert axis % estimate.ndimension() == 1, axis
        estimate_shape = list(estimate.shape)
        del estimate_shape[axis]
        assert estimate_shape == list(
            target.shape
        ), f"{estimate.shape} (N, K, ...) does not match {target.shape} (N, ...)"
    else:
        assert estimate.size() == target.size(), f"{estimate.size()} != {target.size()}"

    candidates = []
    indexer = [
        slice(None),
    ] * estimate.ndim
    permutations = list(itertools.permutations(range(sources)))
    for permutation in permutations:
        indexer[axis] = permutation
        candidates.append(loss_fn(estimate[tuple(indexer)], target))
    min_loss, idx = torch.min(torch.stack(candidates), dim=0)

    if return_permutation:
        return min_loss, permutations[int(idx)]
    else:
        return min_loss
