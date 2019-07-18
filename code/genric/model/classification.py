import torch
import torch.nn

from ..torch_ext import multi_logit
from . import autograd_range


class MultiClassificationNetwork(torch.nn.Module):
    """ Network for multiple joint classification.

    This network is designed to aggregate various logits from
    multiple models and produce a joint classification problem
    across all output logits.

    """
    def __init__(self, task_networks, scopes, batch_size=None):
        """ Initializes a new network.

        Parameters
        ----------
        task_networks: a list of tuples `(str, torch.nn.Module)`, which output
            a logit for each different task.
        scopes: a function which returns a list of scopes for each logit. If
            the returned scope is `None`, it is assumed that the logit is scalar
            for that problem.
        batch_size: the batch size of the problem.
        """
        super(MultiClassificationNetwork, self).__init__()

        self.task_networks = torch.nn.ModuleDict(task_networks)
        self.scopes = scopes
        self.batch_size = batch_size

    def forward(self, embedding, graph):
        task_logits = []
        scopes = []

        scopes_dict = self.scopes(graph)
        for task, network in self.task_networks.items():
            with autograd_range(task):
                task_logits.append(network(embedding, graph))
                scopes.append(scopes_dict.get(task))

        return multi_logit.normalize_values_scopes(task_logits, scopes, self.batch_size)


def multi_classification_loss(logits_and_scopes, label):
    """ Loss for segmented multi-classification.

    Parameters
    ----------
    logits_and_scopes: a tuple of lists containing the task logits and
        corresponding scopes.

    label: a tensor representing the label for each element in the batch.

    Returns
    -------
    A scalar tensor representing the average loss on the batch.
    """
    task_logits, scopes = logits_and_scopes
    losses = multi_logit.segment_multi_softmax_cross_entropy(task_logits, scopes, label)
    return torch.mean(losses)


def multi_classification_coarse_to_fine_loss(logits_and_scopes, label_coarse, label_fine,
                                             alpha=0.5, summary_info=False):
    """ Loss for segmented multi-classification with coarse-to-fine refinement.

    Parameters
    ----------
    logits_and_scopes: a tuple of lists containing the task logits and corresponding scopes.
    label_coarse: the coarse label (or action type).
    label_fine: the specific label of the problem.
    alpha: the weight for the coarse loss.
    summary_info: if True, also return summary info
    """

    task_logits, scopes = logits_and_scopes
    coarse_loss, fine_loss, coarse_logit = multi_logit.segment_multi_softmax_coarse_fine(
        task_logits, scopes, label_coarse, label_fine, return_coarse_prob=True)

    loss = alpha * coarse_loss + (1 - alpha) * fine_loss

    if summary_info:
        summary = {
            'fine_loss': fine_loss,
            'coarse_loss': coarse_loss,
            'coarse_logit': coarse_logit
        }

        return loss, summary
    else:
        return loss


def multi_classification_prediction(logits_and_scopes, predict=False, num_samples=None):
    """ Prediction and sampling for segmented multi-classification task.

    Parameters
    ----------
    logits_and_scopes: a tuple of lists containing the task logits and corresponding scopes.
    predict: if True, return predictions (argmax).
    num_samples: if an integer, the number of samples (without replacement) to return.
    """
    if not predict and not num_samples:
        return None, None

    predictions = []
    samples = []

    task_logits, scopes = logits_and_scopes

    if not num_samples:
        # optimized argmax prediction operation
        _, predictions = multi_logit.segment_multi_argmax(task_logits, scopes)
        return predictions, None

    for i in range(scopes[0].shape[0]):
        logits = torch.cat([
            torch.narrow(tl, 0, s[i, 0], s[i, 1])
            for tl, s in zip(task_logits, scopes)])

        if predict:
            predictions.append(torch.argmax(logits).int())

        if num_samples:
            samples.append(torch.multinomial(torch.exp(logits), num_samples=num_samples))

    predictions = torch.stack(predictions) if predictions else None
    samples = torch.stack(samples) if samples else None

    return predictions, samples
