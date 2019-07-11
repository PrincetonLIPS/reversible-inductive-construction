from induc_gen.model import summary
import numpy as np
import torch
import pytest


def test_summary_accuracy_correct():
    rng = np.random.RandomState(42)

    truth = rng.binomial(1, 0.5, size=40)
    predicted = rng.binomial(1, 0.1, size=40)

    summ = summary.ClassificationSummary()
    summ.record_statistics(torch.tensor(predicted), torch.tensor(truth))

    assert summ.accuracy().item() == pytest.approx(np.mean(truth == predicted))


def test_summary_kappa_correct():
    rng = np.random.RandomState(42)

    truth = rng.binomial(1, 0.5, size=40)
    predicted = rng.binomial(1, 0.1, size=40)

    summ = summary.ClassificationSummary()
    summ.record_statistics(torch.tensor(predicted), torch.tensor(truth))

    pe = np.mean(truth) * np.mean(predicted) + (1 - np.mean(truth)) * (1 - np.mean(predicted))
    po = np.mean(truth == predicted)

    kappa = 1 - (1 - po) / (1 - pe)

    assert summ.cohen_kappa().item() == pytest.approx(kappa, rel=1e-5)


def test_summary_accuracy_correct_multi():
    rng = np.random.RandomState(42)

    truth = rng.binomial(4, 0.5, size=40)
    predicted = rng.binomial(4, 0.4, size=40)

    summ = summary.ClassificationSummary(num_outcomes=5)
    summ.record_statistics(torch.tensor(predicted), torch.tensor(truth))

    assert summ.accuracy().item() == pytest.approx(np.mean(truth == predicted))


def test_summary_marginal_labels_correct():
    rng = np.random.RandomState(42)

    truth = rng.binomial(1, 0.5, size=40)
    predicted = rng.binomial(1, 0.1, size=40)

    summ = summary.ClassificationSummary()
    summ.record_statistics(torch.tensor(truth), torch.tensor(predicted))

    p_truth = np.mean(truth)

    assert summ.marginal_labels().tolist() == pytest.approx([1 - p_truth, p_truth])


def test_summary_conditional_accuracy():
    rng = np.random.RandomState(42)

    conditional = rng.binomial(1, 0.5, size=40)
    correct = rng.binomial(1, 0.1, size=40)

    summ = summary.ConditionalAccuracySummary()
    summ.record_statistics(torch.tensor(correct), torch.tensor(conditional))

    assert summ.probability_event().item() == pytest.approx(np.mean(conditional))
    assert summ.accuracy().item() == pytest.approx(np.sum(conditional * correct) / np.sum(conditional))
