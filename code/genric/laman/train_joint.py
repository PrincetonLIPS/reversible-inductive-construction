import numpy as np
import collections
import numbers
import torch
import os
from . import joint_network
from .summary import LamanClassificationSummary
from .. import corruption_dataset, model as mo
from .representation import graph_to_rep, combine_graph_reps, encode_action, LamanRep, get_action_offsets
from ..molecule_models import _train_utils, _train_harness
from ._utils import cast_numpy_rec


def _transform(graph, act):
    graph_rep = graph_to_rep(graph)
    act_encoded = encode_action(act, graph)
    act_coarse = act.action_type
    offset = torch.from_numpy(get_action_offsets(graph)).int()

    return {
        'graph': graph_rep,
        'label': act_encoded,
        'label_coarse': act_coarse,
        'label_offset': offset
    }


def _collate(batch):
    graph = combine_graph_reps([b['graph'] for b in batch])
    graph = cast_numpy_rec(graph)
    label_fine = torch.LongTensor([b['label'] for b in batch])
    label_coarse = torch.LongTensor([b['label_coarse'] for b in batch])
    offsets = torch.stack([b['label_offset'] for b in batch])

    return {'graph': graph, 'label': label_fine, 'label_coarse': label_coarse, 'label_offset': offsets}


def make_dataloader(dataset, batch_size=128, num_workers=2):
    return torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, collate_fn=_collate,
        pin_memory=True, num_workers=num_workers)


class LamanJointHarness(_train_harness.TrainingHarness):
    _keys = ['label', 'label_offset']

    def __init__(self, model, optimizer, summary, task='train', profile=False):
        super(LamanJointHarness, self).__init__(model, optimizer, summary, task=task, profile=profile)

    def get_model_input(self, batch):
        graph = LamanRep.from_sequence(batch['graph'])
        return graph,

    def get_loss(self, model_output, batch):
        loss, summary_info = mo.classification.multi_classification_coarse_to_fine_loss(
            model_output, batch['label_coarse'], batch['label'], summary_info=True)

        self.summary.record_marginal_probability(
            torch.nn.functional.softmax(summary_info['coarse_logit'].detach(), dim=1).mean(dim=0))

        return loss

    def record_step_summary(self, batch, model_output):
        logits_and_scopes = model_output
        prediction, label, label_offset = _train_harness.compute_and_aggregate_predictions(
            logits_and_scopes, batch, self._keys)

        if self.summary:
            self.summary.record_statistics(prediction, label, label_offset)


def main(parameters=None):
    if parameters is None:
        parameters = {}

    task = parameters.get('task', 'train')
    batch_size = parameters.get('batch_size', 256)

    dataset_path = parameters.get('dataset_path')
    if dataset_path is None:
        dataset_path = '../data/laman/low_decomp_dataset_sample.pkl'

    dataset = corruption_dataset.LamanCorruptionDataset(dataset_path, transform=_transform)

    dataloader = make_dataloader(dataset, batch_size, num_workers=parameters.get('num_workers', 2))

    config = joint_network.JointClassificationNetworkConfig(
        5, message_size=256)
    model = joint_network.JointClassificationNetwork(config)

    if 'model_path' in parameters and parameters['model_path'] is not None:
        model.load_state_dict(torch.load(parameters['model_path'], map_location='cpu'))

    model = model.cuda()

    if task != 'train':
        model = model.eval()

    if task == 'train':
        optimizer, schedulers = _train_utils.init_optimizer(model, parameters)
    else:
        optimizer = None
        schedulers = []

    summary = LamanClassificationSummary()

    harness = LamanJointHarness(model, optimizer, summary, task)
    harness.hooks.extend([
        _train_harness.LogLossTimeHook(batch_size),
        _train_harness.PrintAccuracyHook(summary, None)
    ])

    savedir = _train_utils.get_save_dir(parameters)

    for epoch in range(30):
        dataset.set_epoch(epoch)
        harness.set_epoch(epoch)

        if task == 'train':
            for scheduler in schedulers:
                scheduler.step()

        harness.train_epoch(dataloader)

        if task == 'train':
            torch.save(
                model.state_dict(),
                os.path.join(savedir, 'laman_joint_ep_{0}.pth'.format(epoch)))


if __name__ == '__main__':
    args = _train_utils.parse_arguments()
    main(vars(args))
