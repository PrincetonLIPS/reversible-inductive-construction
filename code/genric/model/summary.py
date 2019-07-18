import torch
import io
import posixpath


class ClassificationSummary:
    """ Simple class to keep track of summaries of a classification problem. """
    def __init__(self, num_outcomes=2, device=None):
        """ Initializes a new summary class with the given number of outcomes.

        Parameters
        ----------
        num_outcomes: the number of possible outcomes of the classification problem.
        """
        self.recorded = torch.zeros(num_outcomes * num_outcomes, dtype=torch.int32, device=device)
        self.num_outcomes = num_outcomes

    @property
    def prediction_matrix(self):
        return self.recorded.view((self.num_outcomes, self.num_outcomes))

    def record_statistics(self, labels, predictions):
        """ Records statistics for a batch of predictions.

        Parameters
        ----------
        labels: an array of true labels in integer format. Each label must correspond to an
            integer in 0 to num_outcomes - 1 inclusive.
        predictions: an array of predicted labels. Must follow the same format as `labels`.
        """
        indices = torch.add(labels.int(), self.num_outcomes, predictions.int()).long()
        self.recorded = self.recorded.scatter_add_(
            0, indices, torch.ones_like(indices, dtype=torch.int32))

    def reset_statistics(self):
        """ Resets statistics recorded in this accumulator. """
        self.recorded = torch.zeros_like(self.recorded)

    def accuracy(self):
        """ Compute the accuracy of the recorded problem. """
        num_outcomes = self.num_outcomes
        num_correct = self.prediction_matrix.diag().sum()
        num_total = self.recorded.sum()

        return num_correct.float() / num_total.float()

    def confusion_matrix(self):
        return self.prediction_matrix.float() / self.prediction_matrix.sum().float()

    def cohen_kappa(self):
        pm = self.prediction_matrix.float()
        N = self.recorded.sum().float()

        p_observed = pm.diag().sum() / N
        p_expected = torch.dot(pm.sum(dim=0), pm.sum(dim=1)) / (N * N)

        return 1 - (1 - p_observed) / (1 - p_expected)

    def marginal_labels(self):
        return self.prediction_matrix.sum(dim=0).float() / self.recorded.sum().float()

    def marginal_predicted(self):
        return self.prediction_matrix.sum(dim=1).float() / self.recorded.sum().float()

    def write_tensorboard(self, writer, prefix="", **kwargs):
        writer.add_scalar(posixpath.join(prefix, "kappa"), self.cohen_kappa(), **kwargs)
        writer.add_scalar(posixpath.join(prefix, "accuracy"), self.accuracy(), **kwargs)


class ConditionalAccuracySummary:
    def __init__(self, device=None):
        self.device = device
        self.count_correct = torch.tensor(0, dtype=torch.int32, device=self.device)
        self.count_event = torch.tensor(0, dtype=torch.int32, device=self.device)
        self.count_total = torch.tensor(0, dtype=torch.int32, device=self.device)

        self.reset_statistics()

    def reset_statistics(self):
        self.count_correct = torch.tensor(0, dtype=torch.int32, device=self.device)
        self.count_event = torch.tensor(0, dtype=torch.int32, device=self.device)
        self.count_total = torch.tensor(0, dtype=torch.int32, device=self.device)

    def accuracy(self):
        return self.count_correct.float() / self.count_event.float()

    def probability_event(self):
        return self.count_event.float() / self.count_total.float()

    def record_statistics(self, correct, mask):
        self.count_event.add_(torch.sum(mask).int())
        self.count_correct.add_(torch.sum(mask * correct).int())
        self.count_total.add_(mask.shape[0])

    def write_tensorboard(self, writer, prefix="", **kwargs):
        writer.add_scalar(posixpath.join(prefix, "accuracy"), self.accuracy(), **kwargs)
        writer.add_scalar(posixpath.join(prefix, "frequency"), self.probability_event(), **kwargs)
