import numpy as np
import torch
import pickle
import contextlib
import gzip
from ..laman import action
from ..laman import laman_edit as edit


@contextlib.contextmanager
def _open_maybe_compressed(path, mode):
    if path.endswith('.gz'):
        with gzip.open(path, mode) as f:
            yield f
    else:
        with open(path, mode) as f:
            yield f


class LamanCorruptionDataset(torch.utils.data.Dataset):
    def __init__(self, path, expected_steps=5, seed=0, transform=None):
        with _open_maybe_compressed(path, 'rb') as f:
            self.data = pickle.load(f)
        self.epoch = 0
        self.expected_steps = expected_steps
        self.seed = 0
        self.transform = transform

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, idx):
        rng = np.random.RandomState(hash((self.seed, self.epoch, idx)) % (2 ** 32))

        number_of_steps = rng.geometric(1 / (1 + self.expected_steps)) - 1
        return_step = rng.randint(number_of_steps + 1)

        graph = self.data[idx]
        # Default action when no corruptions are executed.
        inverse = action.Stop()

        for _ in range(return_step):
            if rng.uniform() < 0.5 and len(graph.nodes) > 3:
                graph, inverse = edit.delete_random_node(graph, rng=rng, return_inverse=True)
            else:
                graph, inverse = edit.insert_random_node(graph, rng=rng, return_inverse=True)

        if self.transform is None:
            return graph, inverse
        else:
            return self.transform(graph, inverse)

    def __len__(self):
        return len(self.data)
