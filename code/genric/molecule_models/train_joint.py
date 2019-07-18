import torch
import numpy as np
import typing

import json
import os
from functools import partial

from .. import Chem

from .. import corruption_dataset
from .. import molecule_representation as mr
from .. import vocabulary
from .summary import JointModelSummary
from . import joint_network

from . import action_representation as ar
from . import _train_utils, _train_harness, modules


def _transform(mol, act, encoder):
    mol_graph = mr.mol2graph_single(mol, include_leaves=True, include_rings=True, normalization='sqrt')
    counts = mol_graph['count']

    mol_ring_bonds = [b.GetIdx() for b in mol.GetBonds() if b.IsInRing()]

    return {
        'graph': mol_graph,
        'label': ar.action_to_integer(
            act, counts['leaf_ring'] + counts['leaf_atom'],
            counts['atom'], mol_ring_bonds, encoder),
        'action_vector': act.to_array()
    }


def _load_dataset(transform, path=None, cache_path='../data/zinc/preprocessed', no_cache=False,
                  vocab=None, expected_corruption_steps=5):
    if path is None:
        path = '../data/zinc/train.txt'

    # Hard-coded for now
    path_cache_path = '../data/zinc/preprocessed-path'

    if os.path.exists(cache_path) and expected_corruption_steps == 5 and not no_cache:
        print('Found existing cached files at {0}'.format(cache_path))
        files = sorted(os.listdir(cache_path))
        files = [os.path.join(cache_path, f) for f in files]
        dataset = corruption_dataset.CachedCorruptionDataset(files, transform=transform)
    elif os.path.exists(path_cache_path) and not no_cache:
        print('Found existing path cache at {0}'.format(path_cache_path))
        files = sorted(os.listdir(path_cache_path))
        files = [os.path.join(path_cache_path, f) for f in files]
        dataset = corruption_dataset.CachedPathCorruptionDataset(files)
        dataset = corruption_dataset.PathToSingleDataset(
            dataset, expected_corruption_steps=expected_corruption_steps,
            seed=0, transform=transform)
    else:
        print('Generating data on the fly')
        dataset = corruption_dataset.CorruptionDataset(
            path, expected_corruption_steps=expected_corruption_steps, transform=transform, vocab=vocab)

    return dataset


class DistributedTrainingInfo(typing.NamedTuple):
    sync_url: str
    world_size: int
    rank: int
    local_rank: int

    def __bool__(self):
        return self.world_size > 1


def initialize_dataset(parameters, distributed_config=None):
    corruption_steps = parameters.get('expected_corruption_steps', 5)

    vocab = vocabulary.Vocabulary(parameters.get('vocab', None))
    action_encoder = ar.VocabInsertEncoder(canonical=parameters.get('canonical', True), vocab=vocab)

    transform_encoder = partial(_transform, encoder=action_encoder)
    dataset = _load_dataset(
        transform_encoder, path=parameters.get('dataset_path'),
        no_cache=parameters.get('no_cache'), vocab=vocab,
        expected_corruption_steps=corruption_steps)

    batch_size = parameters.get('batch_size', 128)

    if distributed_config:
        from torch.utils.data.distributed import DistributedSampler
        batch_size = batch_size // distributed_config.world_size
        sampler = DistributedSampler(dataset)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        drop_last=True, collate_fn=_train_utils.collate,
        num_workers=parameters.get('num_workers', 8), pin_memory=True)

    return dataset, sampler, train_loader, action_encoder


def initialize(parameters, action_encoder, distributed_config=None):
    task = parameters.get('task', 'train')

    if task == 'train' and _train_utils.is_leader():
        savedir = _train_utils.get_save_dir(parameters)
        # Save parameters
        with open(os.path.join(savedir, 'params.json'), 'w') as fp:
            json.dump(parameters, fp, sort_keys=True, indent=4)

    Chem.disable_log('rdApp.*')

    batch_size = parameters.get('batch_size', 128)

    if distributed_config:
        batch_size = batch_size // distributed_config.world_size

    config = joint_network.JointClassificationNetworkConfiguration(
        action_encoder.get_num_atom_insert_locations(),
        action_encoder.num_insert_bond_locations,
        hidden_size=384,
        depth=parameters.get('message_depth', 5))

    model = joint_network.JointClassificationNetwork(batch_size, config)

    model_path = parameters.get('model_path')
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

    if distributed_config:
        print("Creating model on GPU {0}".format(distributed_config.local_rank))
        gpu_id = distributed_config.local_rank
        model = modules.SingleDeviceDistributedParallel(model.cuda(gpu_id), gpu_id)
    else:
        model = model.cuda()

    if task == 'train':
        model.train()
    else:
        model.eval()

    def save_model(ep, it):
        if task == 'train' and _train_utils.is_leader():
            model_filename = os.path.join(
                savedir, "joint_model_ep_{0}_it_{1:04d}.pth".format(ep, it))
            torch.save(model.state_dict(), model_filename)

    if _train_utils.is_leader() and (task == 'train'):
        from torch.utils import tensorboard
        summary_dir = os.path.join(savedir, 'summary')
        writer = tensorboard.SummaryWriter(log_dir=summary_dir)
    else:
        writer = None

    return model, save_model, writer


def train_boostrap_distributed(parameters):
    world_size = parameters.get('world_size', 1)

    if world_size == 1:
        # Single-node training, nothing to do.
        parameters['rank'] = 0
        return train(parameters)

    parameters['rank'] = -1
    from torch.multiprocessing import spawn

    spawn(train_distributed, nprocs=parameters['world_size'], args=(parameters,))


def initialize_distributed(config: DistributedTrainingInfo):
    from torch import distributed as dist

    dist.init_process_group(
        'nccl', init_method=config.sync_url,
        world_size=config.world_size, rank=config.rank)

    torch.cuda.set_device(config.local_rank)


def get_distributed_config(parameters, local_rank=None):
    world_size = parameters.get('world_size', 1)

    if world_size == 1:
        return DistributedTrainingInfo('', 1, 0, 0)

    sync_url = parameters.get('sync_url')
    if sync_url is None:
        sync_url = 'tcp://127.0.0.1:16847'

    rank = local_rank

    return DistributedTrainingInfo(sync_url, world_size, rank, local_rank)


def train_distributed(local_rank, parameters):
    config = get_distributed_config(parameters, local_rank)
    initialize_distributed(config)
    train(parameters, config)


def train(parameters=None, distributed_config=None):
    task = parameters.get('task', 'train')
    dataset, sampler, train_loader, action_encoder = initialize_dataset(parameters, distributed_config)
    model, save_model_fn, summary_writer = initialize(parameters, action_encoder, distributed_config)

    optimizer, schedulers = _train_utils.init_optimizer(model, parameters)

    if _train_utils.is_leader():
        summary = JointModelSummary(action_encoder)
    else:
        summary = None

    harness = _train_harness.TrainingHarness(
        model, optimizer, summary, task, profile=parameters.get('profile', False))

    if _train_utils.is_leader():
        hooks = [
            _train_harness.LogLossTimeHook(parameters.get('batch_size'), parameters.get('log_freq', 10), summary_writer),
            _train_harness.PrintAccuracyHook(summary, summary_writer, task),
            _train_harness.LogModelWeightsHook(model, summary_writer),
            _train_harness.LogOptimizerHook(optimizer, summary_writer)]
    else:
        hooks = []

    harness.hooks.extend(hooks)

    for ep in range(parameters.get('epochs', 10)):
        dataset.set_epoch(ep)
        harness.set_epoch(ep)

        if hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(ep)

        if task == 'train':
            for scheduler in schedulers:
                scheduler.step()

        harness.train_epoch(train_loader)
        save_model_fn(ep+1, 0)


def main():
    args = _train_utils.parse_arguments()
    if args.world_size > 1:
        train_boostrap_distributed(vars(args))
    else:
        train(vars(args))


if __name__ == '__main__':
    main()
