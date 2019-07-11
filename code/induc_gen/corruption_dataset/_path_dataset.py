import numpy as np
import pickle
import torch
import itertools
import io

from .. import action
from .. import molecule_edit as me
from ._dataset import BaseDataset


class PathCorruptionDataset(BaseDataset):
    """ Corruption dataset which returns the entire path instead of a single reconstruction in the path.
    """
    def __init__(self, path, transform=None, expected_steps=5, max_steps=12, ignore_errors=True, seed=None, vocab=None):
        super(PathCorruptionDataset, self).__init__(path, transform, seed, vocab)
        self.expected_steps = expected_steps
        self.max_steps = max_steps
        self.ignore_errors = ignore_errors

    def get_data(self, idx, rng):
        mol, _ = super(PathCorruptionDataset, self).get_data(idx, rng)

        path_mols = [mol]
        path_inverses = [action.Stop()]

        num_steps = self.max_steps + 1

        while self.expected_steps is not None and num_steps > self.max_steps:
            num_steps = rng.geometric(1 / (1 + self.expected_steps)) - 1

        for _ in range(num_steps):
            for _ in range(20):
                try:
                    if rng.uniform() < 0.5 and len(me.get_leaves(mol)) >= 2:
                        mol, inverse = me.delete_random_leaf(mol, rng=rng, return_inverse=True, vocab=self.vocab)
                    else:
                        mol, inverse = me.insert_random_node(mol, self.vocab, rng=rng, return_inverse=True)
                    path_mols.append(mol)
                    path_inverses.append(inverse)
                    break
                except ValueError as e:
                    if not self.ignore_errors:
                        raise e
            else:
                print("Could not continue path for molecule.")
                return path_mols, path_inverses

        return path_mols, path_inverses


class CachedPathCorruptionDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        if not isinstance(path, list):
            path = [path]

        self.transform = transform
        self.path = path
        self._mol_data = None
        self._mol_offsets = None
        self._path_offsets = None
        self._inv = None
        self.epoch = None

        self.set_epoch(0)

    def set_epoch(self, idx):
        if idx == self.epoch:
            return

        self.epoch = idx

        data = np.load(self.path[idx % len(self.path)])

        self._mol_data = torch.tensor(data['mol']).share_memory_()
        self._inv = torch.tensor(data['inv']).share_memory_()

        path_lengths = data['path_len']
        path_offsets = np.zeros(path_lengths.shape[0] + 1, dtype=np.int32)
        np.cumsum(path_lengths, out=path_offsets[1:])
        self._path_offsets = torch.tensor(path_offsets).share_memory_()

        mol_lengths = data['mol_len']
        mol_offsets = np.zeros(mol_lengths.shape[0] + 1, dtype=np.int32)
        np.cumsum(mol_lengths, out=mol_offsets[1:])
        self._mol_offsets = torch.tensor(mol_offsets).share_memory_()

    def __getitem__(self, idx):
        mols = []
        inverses = []

        data_path = io.BytesIO(self._mol_data[self._mol_offsets[idx]:self._mol_offsets[idx + 1]].numpy().tobytes())

        start_idx = self._path_offsets[idx]
        end_idx = self._path_offsets[idx + 1]

        for i in range(start_idx, end_idx):
            mols.append(pickle.load(data_path))
            inverses.append(action.Action.from_array(self._inv[i]))

        if self.transform is not None:
            return self.transform(mols, inverses)
        else:
            return {
                'mol': mols,
                'inverse': inverses
            }

    def __len__(self):
        return len(self._path_offsets) - 1


class PathToSingleDataset(torch.utils.data.Dataset):
    """ Dataset which transforms a path dataset to a dataset which samples
    single path elements (which are used by the joint training).
    """
    def __init__(self, dataset, expected_corruption_steps=5, seed=None, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.expected_corruption_steps = expected_corruption_steps
        self.seed = seed

    def set_epoch(self, epoch):
        if hasattr(self.dataset, 'set_epoch'):
            self.dataset.set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, idx):
        if self.epoch is not None:
            rng = np.random.RandomState(hash((self.seed, self.epoch, idx)) % (2 ** 32))
        else:
            rng = np.random.RandomState(torch.randint(2**32 - 1, size=(8,), device='cpu'))

        num_steps = rng.geometric(1 / (1 + self.expected_corruption_steps)) - 1
        num_steps = rng.randint(num_steps + 1)
        result = self.dataset[idx]

        num_steps = min(num_steps, len(result['mol']) - 1)

        mol = result['mol'][num_steps]
        inv = result['inverse'][num_steps]

        if self.transform is None:
            return {'mol': mol, 'inverse': inv}
        else:
            return self.transform(mol, inv)

    def __len__(self):
        return len(self.dataset)


def _collate(batch):
    return batch


def _transform_serialize(mols, inverses):
    mol_bytes = np.concatenate(
        [np.frombuffer(pickle.dumps(mol, pickle.HIGHEST_PROTOCOL), dtype=np.uint8)
         for mol in mols], axis=0)
    inverse_bytes = np.stack(
        [inv.to_array() for inv in inverses], axis=0)

    return mol_bytes, inverse_bytes, len(mols)


def main():
    import argparse
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default='cached_data.npz')
    parser.add_argument('-p', '--path', type=str, default='../data/zinc/train.txt')
    parser.add_argument('-s', '--seed', type=int, default=42)
    parser.add_argument('-e', '--epoch', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--vocab', type=str, default='zinc')
    parser.add_argument('--limit-elements', type=int, default=None)

    args = parser.parse_args()

    batch_size = 32

    dataset = PathCorruptionDataset(args.path, _transform_serialize, expected_steps=None)
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
    lengths = np.empty(len(dataset), dtype=np.int32)

    data_elements = itertools.chain.from_iterable(dataloader)

    if args.limit_elements is not None:
        data_elements = itertools.islice(data_elements, args.limit_elements)

    progress_iter = tqdm.tqdm(data_elements, total=args.limit_elements or len(dataset))

    for i, (mol, inverse, length) in enumerate(progress_iter):
        output_mol.append(mol)
        output_inv.append(inverse)
        lengths[i] = length
        progress_iter.set_postfix(make_postfix(), refresh=False)

    info_arr = np.array([args.seed, args.epoch])
    mol_arr = np.concatenate(output_mol)
    mol_length = np.array([len(x) for x in output_mol], dtype=np.int32)
    inv_arr = np.concatenate(output_inv, axis=0)

    result = {
        'info': info_arr,
        'mol': mol_arr,
        'mol_len': mol_length,
        'inv': inv_arr,
        'path_len': lengths
    }

    np.savez_compressed(args.output, **result)

if __name__ == '__main__':
    main()
