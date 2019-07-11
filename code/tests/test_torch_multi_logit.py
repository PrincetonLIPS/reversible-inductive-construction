import torch
import numpy as np
import pytest

from induc_gen.torch_ext import multi_logit


def _scopes_from_lengths(arr):
    arr = np.array(arr)
    offsets = np.zeros_like(arr)
    offsets[1:] = np.cumsum(arr[:-1])
    return np.stack((offsets, arr), axis=1)


def _make_data(seed=None, device=None):
    rng = np.random.RandomState(seed)
    batch_size = 4

    logits1 = torch.tensor(rng.randn(5), device=device)
    logits2 = torch.tensor(rng.randn(16), device=device)
    logits3 = torch.tensor(rng.randn(20).reshape(10, 2), device=device)

    scopes1 = None
    scopes2 = torch.tensor(_scopes_from_lengths([3, 5, 6, 2, 0]), dtype=torch.int64, device=device)
    scopes3 = torch.tensor(_scopes_from_lengths([3, 1, 2, 3, 1]), dtype=torch.int64, device=device)

    label = torch.tensor([0, 4, 8, 1, 0], dtype=torch.int64, device=device)
    logits = [logits1, logits2, logits3]
    scopes = [scopes1, scopes2, scopes3]

    logits, scopes = multi_logit.normalize_values_scopes(logits, scopes)
    return logits, scopes, label


def test_select_label_multi_segmented_python():
    logits, scopes, label = _make_data(42)

    expected = multi_logit.select_label_multi_segment_loop(
        logits, scopes, label)
    result = multi_logit.select_label_multi_segment_python(
        logits, scopes, label)

    assert np.allclose(expected.cpu().numpy(), result.cpu().numpy())


def test_multi_logit_softmax_cross_entropy():
    logits, scopes, label = _make_data(42)

    result = multi_logit.segment_multi_softmax_cross_entropy(
        logits, scopes, label)
    expected = multi_logit.segment_multi_softmax_cross_entropy_loop(
        logits, scopes, label)

    assert np.allclose(expected.cpu().numpy(), result.cpu().numpy())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_multi_logit_softmax_cross_entropy_grad(device):
    logits, scopes, label = _make_data(42, device=device)

    logits = [l.requires_grad_(True) for l in logits]

    def logits_function(l1, l2, l3):
        return multi_logit.segment_multi_softmax_cross_entropy(
            [l1, l2, l3], scopes, label)

    torch.autograd.gradcheck(logits_function, tuple(logits))


def test_multi_logit_argmax():
    logits, scopes, _ = _make_data(42)

    max_val, argmax = multi_logit.segment_multi_argmax(logits, scopes)
    max_val_expected, argmax_expected = multi_logit.segment_multi_argmax_loop(logits, scopes)
    batch_length = torch.stack([s.select(1, 1) for s in scopes], dim=1).sum(dim=1)

    assert torch.all(argmax < batch_length).item()
    assert torch.all(argmax == argmax_expected).item()
    assert torch.all(max_val == max_val_expected).item()
