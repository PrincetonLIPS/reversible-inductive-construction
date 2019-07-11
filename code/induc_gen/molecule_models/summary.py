import io
import torch
import posixpath
from ..model.summary import ClassificationSummary, ConditionalAccuracySummary
from . import action_representation as ar


def _print_summary(summary, conditional_accuracies, marginal_labels=['S', 'D', 'IA', 'IB']):
    result = io.StringIO()

    def marginal_statistics(stats):
        return "(" + ', '.join([l + ": {0:.0%}".format(v) for l, v in zip(marginal_labels, stats)]) + ")"

    def conditional_accuracy(acc):
        return "performance: accuracy {0:.1%}, frequency {1:.1%}".format(
            acc.accuracy().item(), acc.probability_event().item())

    print("Marginal predicted action: " + marginal_statistics(summary.marginal_predicted().tolist()) +
          ", marginal observed: " + marginal_statistics(summary.marginal_labels().tolist()),
          file=result)

    for key, value in conditional_accuracies.items():
        print("{0} performance: {1}".format(key, conditional_accuracy(value)), file=result)

    return result.getvalue()


class JointModelSummary:
    """ Class for collecting summary information for joint model training. """
    def __init__(self, action_encoder):
        self.action_encoder = action_encoder
        self.summary = ClassificationSummary(num_outcomes=4)
        self.accuracy_deletion = ConditionalAccuracySummary()
        self.accuracy_insert_atom = ConditionalAccuracySummary()
        self.accuracy_insert_atom_location = ConditionalAccuracySummary()
        self.accuracy_insert_atom_vocab = ConditionalAccuracySummary()

    def reset_statistics(self):
        self.summary.reset_statistics()
        self.accuracy_deletion.reset_statistics()
        self.accuracy_insert_atom.reset_statistics()
        self.accuracy_insert_atom_location.reset_statistics()
        self.accuracy_insert_atom_vocab.reset_statistics()

    def record_statistics(self, predictions, label, label_offsets, action_vector):
        predictions = predictions.cpu()
        label = label.cpu()
        label_offsets = label_offsets.cpu()
        action_vector = action_vector.cpu()

        action_type_label = sum([label >= label_offsets[:, i] for i in range(1, 4)])
        action_type_predicted = sum([predictions >= label_offsets[:, i] for i in range(1, 4)])

        self.summary.record_statistics(action_type_label, action_type_predicted)
        self.accuracy_deletion.record_statistics(predictions == label, action_type_label == 1)

        predicted_insert_atom_location = torch.tensor(ar.integer_to_insert_atom_location(
            predictions.numpy(), label_offsets.numpy(), self.action_encoder), dtype=torch.int32)

        predicted_insert_vocab_item = torch.tensor(ar.integer_to_insert_atom_vocab(
            predictions.numpy(), label_offsets.numpy(), self.action_encoder), dtype=torch.int32)

        label_insert_atom_location = torch.tensor(ar.integer_to_insert_atom_location(
            label.numpy(), label_offsets.numpy(), self.action_encoder), dtype=torch.int32)

        action_is_insert_atom = (action_type_label == 2)
        self.accuracy_insert_atom.record_statistics(predictions == label, action_is_insert_atom)
        self.accuracy_insert_atom_location.record_statistics(
            predicted_insert_atom_location == label_insert_atom_location, action_is_insert_atom)
        self.accuracy_insert_atom_vocab.record_statistics(
            predicted_insert_vocab_item == action_vector[:, 2], action_is_insert_atom)

    def get_string_representation(self):
        accuracy = self.summary.accuracy()
        kappa = self.summary.cohen_kappa()

        result = "Global action type performance: accuracy {0:.1%}, kappa {1:.2f}\n".format(
            accuracy.item(), kappa.item())

        result += _print_summary(self.summary, {
            "Deletion": self.accuracy_deletion,
            "Insert Atom": self.accuracy_insert_atom,
            "Insert Atom Location": self.accuracy_insert_atom_location,
            "Insert Atom Vocab": self.accuracy_insert_atom_vocab
        })

        return result

    def write_tensorboard(self, writer, prefix="", **kwargs):
        self.summary.write_tensorboard(writer, posixpath.join(prefix, "coarse"), **kwargs)
        self.accuracy_deletion.write_tensorboard(writer, posixpath.join(prefix, "delete"), **kwargs)
        self.accuracy_insert_atom.write_tensorboard(writer, posixpath.join(prefix, "insert_atom"), **kwargs)


class SplitModelSummary:
    """ Class for collecting summary information for split model training. """
    def __init__(self, vocab_encoder):
        self.summary_delete = ClassificationSummary(num_outcomes=2)
        self.summary_insert = ClassificationSummary(num_outcomes=3)

        self.accuracy_deletion = ConditionalAccuracySummary()
        self.accuracy_insert_atom = ConditionalAccuracySummary()
        self.accuracy_insert_bond = ConditionalAccuracySummary()

        self.encoder = vocab_encoder

    def reset_statistics(self):
        self.summary_delete.reset_statistics()
        self.summary_insert.reset_statistics()

        self.accuracy_deletion.reset_statistics()
        self.accuracy_insert_atom.reset_statistics()
        self.accuracy_insert_bond.reset_statistics()

    def record_statistics(self, predictions, label, label_offsets, action_vector, mode):
        predictions = predictions.cpu()
        label = label.cpu()
        label_offsets = label_offsets.cpu()
        action_vector = action_vector.cpu()

        if mode == 0:
            self._record_statistics_delete(predictions, label, label_offsets, action_vector)
        else:
            self._record_statistics_insert(predictions, label, label_offsets, action_vector)

    def _record_statistics_delete(self, predictions, label, label_offsets, action_vector):
        # In delete mode, we only have switch (label 0) or delete (all other labels).
        action_type_label = label != 0
        action_type_predicted = predictions != 0

        self.summary_delete.record_statistics(action_type_label, action_type_predicted)
        self.accuracy_deletion.record_statistics(predictions == label, action_type_label)

    def _record_statistics_insert(self, predictions, label, label_offsets, action_vector):
        action_type_label = (label.unsqueeze(1) >= label_offsets).sum(dim=1) - 1
        action_type_predicted = (predictions.unsqueeze(1) >= label_offsets).sum(dim=1) - 1
        correct = label == predictions

        self.summary_insert.record_statistics(action_type_label, action_type_predicted)
        self.accuracy_insert_atom.record_statistics(correct, action_type_label == 1)
        self.accuracy_insert_bond.record_statistics(correct, action_type_label == 2)

    def get_string_representation(self):
        accuracy_insert = self.summary_insert.accuracy()
        accuracy_delete = self.summary_delete.accuracy()

        kappa_insert = self.summary_insert.cohen_kappa()
        kappa_delete = self.summary_delete.cohen_kappa()

        result = "Insert Mode Perf: accuracy: {0:.1%}, kappa: {1:.1%}. ".format(accuracy_insert, kappa_insert)
        result += "Delete Mode Perf: accuracy: {0:.1%}, kappa: {1:.1%}\n".format(accuracy_delete, kappa_delete)

        result += _print_summary(self.summary_insert, {
            'Insert Atom': self.accuracy_insert_atom,
            'Insert Bond': self.accuracy_insert_bond
        }, ('S', 'IA', 'IB'))

        result += _print_summary(self.summary_delete, {
            'Delete': self.accuracy_deletion
        }, ('S', 'D'))

        return result
