import torch
import numpy as np
from induc_gen.torch_ext import segment_indirect


def _scopes_from_lengths(arr):
    arr = np.array(arr)
    offsets = np.zeros_like(arr)
    offsets[1:] = np.cumsum(arr[:-1])
    return np.stack((offsets, arr), axis=1)


def test_segment_index_add():
    rng = np.random.RandomState(42)
    x = rng.randn(20, 10)
    segment_lengths = [5, 2, 3]
    idx = [0, 10, 19, 3, 4, 6, 7, 1, 5, 4]
    scopes = _scopes_from_lengths(segment_lengths)

    result = segment_indirect.segment_index_add(
        torch.tensor(x),
        torch.tensor(scopes),
        torch.tensor(idx)).cpu().numpy()

    assert np.allclose(result[1], x[6] + x[7])
