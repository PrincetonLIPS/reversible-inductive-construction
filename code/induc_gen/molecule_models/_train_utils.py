import os
import uuid
import torch
import numpy as np
import collections
import argparse

from functools import partial
from .. import molecule_representation as mr, corruption_dataset


def is_leader():
    from torch import distributed as dist
    return not(dist.is_available() and dist.is_initialized() and dist.get_rank() != 0)


def is_distributed():
    from torch import distributed as dist
    return dist.is_available() and dist.is_initialized()


def load_cuda_async(batch, device=None):
    if device is not None and device.type != "cuda":
        return batch

    load_cuda_async_device = partial(load_cuda_async, device=device)

    if batch is None:
        return None
    elif isinstance(batch, torch.Tensor):
        return batch.cuda(device=device, non_blocking=True)
    elif hasattr(batch, '_make'):
        # Check for namedtuple
        return batch._make(map(load_cuda_async_device, batch))
    elif isinstance(batch, collections.abc.Mapping):
        return {k: load_cuda_async_device(v) for k, v in batch.items()}
    elif isinstance(batch, collections.abc.Sequence):
        return [load_cuda_async_device(v) for v in batch]
    elif isinstance(batch, (int, np.int32, np.int64)):
        return batch
    else:
        raise ValueError("Unsupported batch collection type {0}.".format(type(batch)))


def _make_sparse_or_none(tup):
    if tup is None:
        return None

    idx, values, size = tup
    return torch.sparse_coo_tensor(idx, values, size, device=values.device)._coalesced_(True)


def replace_sparse_tensor(graph: mr.GraphInfo, mode='tuple'):
    if mode == 'tuple':
        return graph
    elif mode == 'sparse_coo':
        return graph._replace(
            atom_info=graph.atom_info._replace(
                atom_incidence=_make_sparse_or_none(graph.atom_info.atom_incidence)),
            bond_info=graph.bond_info._replace(
                bond_incidence=_make_sparse_or_none(graph.bond_info.bond_incidence)),
            leaf_info=graph.leaf_info._replace(
                leaf_ring_info=graph.leaf_info.leaf_ring_info._replace(
                    feature=_make_sparse_or_none(graph.leaf_info.leaf_ring_info.feature))))
    else:
        raise ValueError("Unexpected mode.")


def cast_numpy_to_torch(x):
    x = torch.from_numpy(x)

    if x.dtype == torch.int32:
        x = x.long()

    return x


def collate(batch, graph_keys=None):
    result = {}

    if graph_keys is None:
        graph_keys = ['graph']

    all_graphs = {
        k: [x[k] for x in batch] for k in graph_keys
    }

    for k, v in all_graphs.items():
        result[k] = mr.combine_mol_graph(v, return_namedtuple=True, cast_tensor=cast_numpy_to_torch)

    if 'label' in batch[0]:
        all_labels = [x['label'][0] for x in batch]
        all_labels_offsets = [x['label'][1] for x in batch]
        all_labels_lengths = [x['label'][2] for x in batch]
        labels = torch.tensor(all_labels, dtype=torch.int32)
        label_offsets = torch.tensor(np.stack(all_labels_offsets), dtype=torch.int32)
        label_lengths = torch.tensor(np.stack(all_labels_lengths), dtype=torch.int32)

        result['label'] = labels
        result['label_offsets'] = label_offsets
        result['label_lengths'] = label_lengths

    if 'action_vector' in batch[0]:
        all_action_vectors = [x['action_vector'] for x in batch]
        action_vector = torch.tensor(np.stack(all_action_vectors), dtype=torch.int32)
        result['action_vector'] = action_vector

    return result


def get_save_dir(parameters=None):
    if parameters is None:
        parameters = {}

    base_dir = parameters.get('save_dir')

    if base_dir is None:
        base_dir = '../output'

    for _ in range(10):
        try:
            savedir = os.path.join(base_dir, uuid.uuid4().hex[:6])
            savedir = os.path.abspath(savedir)
            os.makedirs(savedir)
            break
        except OSError:
            pass
    else:
        raise RuntimeError("Could not create unique output directory!")
    return savedir


def init_optimizer(model, parameters):
    # Set loss and optimizer
    learning_rate = parameters.get('learning_rate', 2e-3) / 128 * parameters.get('batch_size', 128)

    if parameters.get('task', 'train') == 'train':
        optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
        scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, get_learning_rate_decay)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[12, 24, 36])

        schedulers = [scheduler_warmup]
    else:
        optimizer = None
        schedulers = None

    return optimizer, schedulers


def get_learning_rate_decay(ep, warmup_epochs=5, milestones=[12, 24, 36]):
    lr_decayed = 1

    for m in milestones:
        if ep >= m:
            lr_decayed *= 0.1
        else:
            break

    ratio = warmup_epochs / 10
    lr_decayed *= min(1, (ep / ratio + 1) / (warmup_epochs / ratio))

    return lr_decayed


def load_path_dataset(transform, path=None, cache_path='../data/zinc/preprocessed-path', no_cache=False, vocab=None):
    if path is None:
        path = '../data/zinc/train.txt'

    if os.path.exists(cache_path) and not no_cache:
        print('Found existing cached files at {0}'.format(cache_path))
        files = sorted(os.listdir(cache_path))
        files = [os.path.join(cache_path, f) for f in files]
        dataset = corruption_dataset.CachedPathCorruptionDataset(files, transform=transform)
    else:
        print('Generating data on the fly')
        dataset = corruption_dataset.PathCorruptionDataset(
            path, transform=transform, vocab=vocab)

    return dataset


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['train', 'test'], default='train')
    parser.add_argument('--dataset-path', default=None)
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--save-dir', default=None)
    parser.add_argument('--message-depth', default=5, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--lr', default=2e-3, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--data-dir', default='zinc')
    parser.add_argument('--log-frequency', default=10)
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--canonical', action='store_true', dest='canonical')
    parser.add_argument('--no-canonical', action='store_false', dest='canonical')
    parser.add_argument('--max-iterations', default=2**64 - 1, type=int)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--vocab', default=None)
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--expected-corruption-steps', default=5, type=int)

    parser.set_defaults(canonical=True)

    args = parser.parse_args()
    return args
