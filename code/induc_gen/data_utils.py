from __future__ import print_function

from torch.utils.data import Dataset
from random import shuffle
import numpy as np
import six

from .chemutils import get_mol, get_smiles

from functools import lru_cache


@lru_cache()
def get_vocab(isomeric=False, vocab_name='zinc'):
    """ Loads the vocabulary file from package data. """
    import pkg_resources
    import gzip

    with gzip.open(pkg_resources.resource_stream(__name__, 'vocab/{0}_vocab.txt.gz'.format(vocab_name)), mode='r') as f:
        vocab = [line.decode().strip("\r\n ") for line in f.readlines()]

    # if not isomeric: maybe no stereo in vocab?
    #     vocab = [get_smiles(get_mol(s), isomeric=False) for s in vocab]

    vocab = [(s, get_mol(s)) for s in vocab]

    return vocab


class MoleculeDataset(Dataset):
    def __init__(self, real_file, fake_file):
        with open(real_file) as f:
            self.data = [(line.strip("\r\n ").split()[0], 1) for line in f]
        with open(fake_file) as f:
            fake_data = [(line.strip("\r\n ").split()[0], 0) for line in f]
        self.data = self.data[:len(fake_data)]
        self.data.extend(fake_data)
        shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'x': self.data[idx][0], 'label': float(self.data[idx][1])}

if __name__ == "__main__":
    dataset = MoleculeDataset('zinc/train.txt', 'zinc_corrupt.txt')
    print(dataset[10])
