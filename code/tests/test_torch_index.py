import numpy as np
import torch

from induc_gen.torch_ext import index


def _scopes_from_lengths(arr):
    arr = np.array(arr)
    offsets = np.zeros_like(arr)
    offsets[1:] = np.cumsum(arr[:-1])
    return np.stack((offsets, arr), axis=1)


def test_segment_cartesian_product():
    rng = np.random.RandomState(42)

    va = torch.tensor(rng.randn(10, 2))
    vb = torch.tensor(rng.randn(8, 2))

    sa = torch.tensor(_scopes_from_lengths([3, 7]))
    sb = torch.tensor(_scopes_from_lengths([2, 6]))

    result = index.segment_cartesian_product(va, vb, sa, sb)

    i1 = torch.cartesian_prod(torch.arange(sa[0, 1]), torch.arange(sb[0, 1]))
    r1 = torch.cat((va.index_select(0, i1[:, 0]), vb.index_select(0, i1[:, 1])), dim=-1)

    assert np.allclose(r1.cpu().numpy(), result[:r1.shape[0], :])
