import torch
import numpy as np

import pytest
from genric.torch_ext import segment_pool as sp


def _scopes_from_lengths(arr):
    arr = np.array(arr)
    offsets = np.zeros_like(arr)
    offsets[1:] = np.cumsum(arr[:-1])
    return np.stack((offsets, arr), axis=1)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_segment_avg_pool(device):
    rng = np.random.RandomState(5)

    values = torch.tensor(rng.randn(20 * 5).reshape(20, -1), device=device)
    scopes = torch.tensor(_scopes_from_lengths(np.array([3, 7, 6, 4])), device=device).long()

    expected = sp.segment_avg_pool1d_loop(values, scopes)
    result = sp.segment_avg_pool1d_native(values, scopes)

    assert np.allclose(expected.cpu().numpy(), result.cpu().numpy())


@pytest.mark.parametrize("fn", [sp.segment_avg_pool1d_loop, sp.segment_avg_pool1d_native])
def test_segment_avg_pool_grad(fn):
    rng = np.random.RandomState(5)

    values = torch.tensor(rng.randn(20 * 5).reshape(20, -1))
    scopes = torch.tensor(_scopes_from_lengths(np.array([3, 7, 6, 4]))).long()

    torch.autograd.gradcheck(fn, (values.requires_grad_(), scopes))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_segment_max_pool(device):
    rng = np.random.RandomState(5)

    values = torch.tensor(rng.randn(20 * 5).reshape(20, -1), device=device)
    scopes = torch.tensor(_scopes_from_lengths(np.array([3, 7, 6, 4])), device=device).long()

    expected, e_idx = sp.segment_max_pool1d_loop(values, scopes, return_indices=True)
    result, r_idx = sp.segment_max_pool1d_native(values, scopes, return_indices=True)

    assert np.allclose(expected.cpu(), result.cpu())
    assert np.all(e_idx.cpu().numpy() == r_idx.cpu().numpy())


@pytest.mark.parametrize("fn", [sp.segment_max_pool1d_loop, sp.segment_max_pool1d_native])
def test_segment_max_pool_grad(fn):
    rng = np.random.RandomState(5)

    values = torch.tensor(rng.randn(20 * 5).reshape(20, -1))
    scopes = torch.tensor(_scopes_from_lengths(np.array([3, 7, 6, 4]))).long()

    torch.autograd.gradcheck(fn, (values.requires_grad_(), scopes))
