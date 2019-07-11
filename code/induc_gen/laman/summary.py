import torch
import io
from ..model.summary import ClassificationSummary, ConditionalAccuracySummary

_action_short_labels = ['S', 'I', 'II', 'RI', 'RII']


class MeanSummary:
    def __init__(self):
        self.value = None
        self.count = 0

    def record_statistic(self, value):
        if self.value is None:
            self.value = value.clone().detach_()
        else:
            self.value.add_(value.detach())

        self.count += 1

    def reset_statistics(self):
        self.value = None
        self.count = 0

    def mean(self):
        if self.value is None:
            return None
        else:
            return self.value / self.count


class LamanClassificationSummary:
    def __init__(self):
        self.coarse_summary = ClassificationSummary(num_outcomes=5)
        self.marginal_output_prob = MeanSummary()

    def reset_statistics(self):
        self.coarse_summary.reset_statistics()
        self.marginal_output_prob.reset_statistics()

    def record_statistics(self, predictions, label, label_offsets):
        predictions = predictions.cpu().int()
        label = label.cpu().int()
        label_offsets = label_offsets.cpu().int()

        action_type_label = sum([label >= label_offsets[:, i] for i in range(1, 5)])
        action_type_predicted = sum([predictions >= label_offsets[:, i] for i in range(1, 5)])

        self.coarse_summary.record_statistics(action_type_label, action_type_predicted)

    def record_marginal_probability(self, probability):
        self.marginal_output_prob.record_statistic(probability)

    def get_string_representation(self):
        accuracy = self.coarse_summary.accuracy()
        kappa = self.coarse_summary.cohen_kappa()

        marginal_labels = self.coarse_summary.marginal_labels()
        marginal_predicted = self.coarse_summary.marginal_predicted()
        marginal_sampled = self.marginal_output_prob.mean()

        result = io.StringIO()

        print("Marginal Observed: " + ", ".join(
            ["{0}: {1:.1%}".format(l, m) for l, m in zip(_action_short_labels, marginal_labels)]),
            file=result)
        print("Marginal Predicted: " + ", ".join(
            ["{0}: {1:.1%}".format(l, m) for l, m in zip(_action_short_labels, marginal_predicted)]),
            file=result)

        if marginal_sampled is not None:
            print("Marginal Sampled: " + ", ".join([
                "{0}: {1:.1%}".format(l, m) for l, m in zip(
                    _action_short_labels, marginal_sampled.cpu().tolist())]),
                  file=result)

        print("Global action type performance: accuracy {0:.1%}, kappa {1:.2f}\n".format(
            accuracy.item(), kappa.item()),
            file=result)

        return result.getvalue()
