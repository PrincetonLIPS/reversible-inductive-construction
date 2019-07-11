import numpy as np
from functools import partial

from ..chemutils import get_smiles, get_mol
from .. import molecule_edit as me
from .. import action, data_utils, vocabulary
import torch
import itertools
import bz2
import pickle
import multiprocessing
import inspect

from ._dataset import CorruptionDataset, SplitCorruptionDataset


class CachedCorruptionDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        if not isinstance(path, list):
            path = [path]

        self.transform = transform
        self.path = path
        self._mol_data = None
        self._mol_scopes = None
        self._inv = None
        self.epoch = None

        self.set_epoch(0)

    def set_epoch(self, idx):
        if self.epoch == idx:
            return

        self.epoch = idx

        data = np.load(self.path[idx])

        self._mol_data = torch.tensor(data['mol']).share_memory_()

        lengths = data['mol_len']
        offsets = np.zeros_like(lengths)
        np.cumsum(lengths[:-1], out=offsets[1:])

        scopes = np.stack((offsets, lengths), axis=1)
        self._mol_scopes = torch.tensor(scopes).share_memory_()

        self._inv = torch.tensor(data['inv']).share_memory_()

        for name, value in data.items():
            if name in ('mol', 'mol_len', 'inv'):
                continue
            setattr(self, name, torch.tensor(value).share_memory_())

    def __getitem__(self, idx):
        mol_data = self._mol_data.narrow(0, *self._mol_scopes[idx])
        mol = pickle.loads(mol_data.numpy().tobytes())

        inverse = action.Action.from_array(self._inv[idx])

        if self.transform is not None:
            return self.transform(mol, inverse)
        else:
            return {
                'mol': mol,
                'inverse': inverse
            }

    def __len__(self):
        return self._mol_scopes.shape[0]


def _collate(batch):
    return batch


def _transform_serialize(mol, inverse):
    mol_bytes = np.frombuffer(pickle.dumps(mol, pickle.HIGHEST_PROTOCOL), dtype=np.uint8)
    inverse_bytes = inverse.to_array()

    return mol_bytes, inverse_bytes


def make_dataset(args):
    common_args = {
        'seed': args.seed,
        'transform': _transform_serialize,
        'vocab': args.vocab
    }

    if args.dataset_type == 'corruption':
        return CorruptionDataset(args.path, args.expected_steps, **common_args)
    elif args.dataset_type == 'corruption_split':
        return SplitCorruptionDataset(args.path, args.expected_steps_delete, args.expected_steps_insert, **common_args)


def main():
    import argparse
    import pickle
    import bz2
    import tqdm
    import multiprocessing

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default='cached_data.npz')
    parser.add_argument('-p', '--path', type=str, default='../data/zinc/train.txt')
    parser.add_argument('-s', '--seed', type=int, default=42)
    parser.add_argument('-e', '--epoch', type=int, default=0)
    parser.add_argument('--expected-steps', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--vocab', type=str, default='zinc')
    parser.add_argument('--expected-steps-delete', type=int, default=2)
    parser.add_argument('--expected-steps-insert', type=int, default=3)
    parser.add_argument('--dataset-type', type=str, default='corruption', choices=['corruption', 'corruption_split'])

    args = parser.parse_args()

    batch_size = 32

    dataset = make_dataset(args)
    dataset.set_epoch(args.epoch)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, collate_fn=_collate, num_workers=args.num_workers,
        pin_memory=False, drop_last=False)

    if args.num_workers == 0:
        def make_postfix():
            return {
                'Cache hit rate (atom)': dataset.vocab.cache_atom_hit_rate,
                'Cache hit rate (bond)': dataset.vocab.cache_bond_hit_rate
            }
    else:
        def make_postfix():
            return {}

    output_mol = []
    output_inv = []

    progress_iter = tqdm.tqdm(itertools.chain.from_iterable(dataloader), total=len(dataset))

    for i, (mol, inverse) in enumerate(progress_iter):
        output_mol.append(mol)
        output_inv.append(inverse)
        progress_iter.set_postfix(make_postfix(), refresh=False)

    info_arr = np.array([args.seed, args.expected_steps, args.epoch])
    mol_arr = np.concatenate(output_mol)
    mol_length = np.array([len(x) for x in output_mol], dtype=np.int32)
    inv_arr = np.stack(output_inv)

    result = {
        'info': info_arr,
        'mol': mol_arr,
        'mol_len': mol_length,
        'inv': inv_arr
    }

    for name, _ in inspect.getmembers(dataset, lambda o: isinstance(o, property)):
        result[name] = getattr(dataset, name)

    np.savez_compressed(args.output, **result)

if __name__ == '__main__':
    main()
