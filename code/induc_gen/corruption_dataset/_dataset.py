import numpy as np

from ..chemutils import get_mol
from .. import molecule_edit as me
from .. import action, vocabulary
import torch


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None, seed=None, vocab=None):
        with open(path, 'r') as f:
            self.data = f.readlines()

        if vocab is None:
            vocab = vocabulary.Vocabulary()
        elif isinstance(vocab, str):
            vocab = vocabulary.Vocabulary(vocab)

        self.vocab = vocab

        self.transform = transform
        self.epoch = None
        self.seed = seed

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_data(self, idx, rng):
        try:
            return get_mol(self.data[idx]), None
        except RuntimeError as e:
            raise RuntimeError("Failed to parse smile {0}. Original error {1}".format(self.data[idx], e))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.epoch is not None:
            rng = np.random.RandomState(hash((self.seed, self.epoch, idx)) % (2 ** 32))
        else:
            rng = np.random.RandomState(torch.randint(2**32 - 1, size=(8,), device='cpu'))

        mol, inverse = self.get_data(idx, rng)

        if self.transform:
            return self.transform(mol, inverse)
        else:
            return {'mol': mol, 'inverse': inverse}


class CorruptionDataset(BaseDataset):
    """ Torch dataset which produces instances of examples from the corruption chain. """
    def __init__(self, path, expected_corruption_steps=5, transform=None, ignore_errors=True,
                 inverse_type=None, seed=None, vocab=None):
        """ Creates a new instance of the dataset.

        Parameters
        ----------
        path: the path from which to load the data.
        expected_corruption_steps: the average number of corruption steps to take.
        transform: optional transformation to be applied.
        ignore_errors: if True, ignores errors from invalid inductive steps and
            returns the last valid corruption.
        inverse_type: optional type of inverse action desired. Rejection sample until hit.
        """
        super(CorruptionDataset, self).__init__(path, transform, seed, vocab)

        self.expected_corruption_steps = expected_corruption_steps
        self.ignore_errors = ignore_errors
        self.inverse_type = inverse_type

    def get_data(self, idx, rng):
        number_of_steps = rng.geometric(1 / (1 + self.expected_corruption_steps)) - 1
        return_step = rng.randint(number_of_steps + 1)

        mol, _ = super(CorruptionDataset, self).get_data(idx, rng)

        # Default action when no corruptions are executed.
        inverse = action.Stop()

        try:
            for _ in range(return_step):
                if rng.uniform() < 0.5 and len(me.get_leaves(mol)) >= 2:
                    mol, inverse = me.delete_random_leaf(mol, rng=rng, return_inverse=True, vocab=self.vocab)
                else:
                    # Insert
                    mol, inverse = me.insert_random_node(mol, self.vocab, rng=rng, return_inverse=True)
        except ValueError as e:
            if not self.ignore_errors:
                raise e

        if self.inverse_type:
            if not isinstance(inverse, self.inverse_type):
                return self.get_data(rng.randint(len(self)))

        return mol, inverse


class SplitCorruptionDataset(BaseDataset):
    """ Corruption dataset with a canonical ordering where deletion are
    executed before insertions.

    The dataset is split so that half the data is produced from a molecule
    in the deletion phase, wherease the other half is produced from a molecule
    in the insertion phase.

    """
    def __init__(self, path, expected_delete_steps=2, expected_insert_steps=3, transform=None, seed=None, vocab=None):
        super(SplitCorruptionDataset, self).__init__(path, transform, seed, vocab)
        self.expected_delete_steps = expected_delete_steps
        self.expected_insert_steps = expected_insert_steps

        self._partition_idx = []
        self._partition = np.empty(len(self), dtype=np.int32)

        self._set_partition(0)

    def set_epoch(self, epoch):
        super(SplitCorruptionDataset, self).set_epoch(epoch)
        self._set_partition(epoch)

    def _set_partition(self, epoch):
        rng = np.random.RandomState(epoch)

        permutation = rng.permutation(len(self))
        self._partition_idx = np.array_split(permutation, 2)

        partition = np.empty(len(self), dtype=np.int32)

        for i, idx in enumerate(self._partition_idx):
            partition[idx] = i

        self._partition = torch.tensor(partition).share_memory_()

    @property
    def partition_index(self):
        """ Returns a list of indices corresponding to each partition. """
        return self._partition_idx

    def get_data(self, idx, rng):
        delete_steps = rng.geometric(1 / (1 + self.expected_delete_steps)) - 1
        insert_steps = rng.geometric(1 / (1 + self.expected_insert_steps)) - 1

        mol = get_mol(self.data[idx])

        # If this is true, we will only execute delete actions.
        return_delete_inverse = self._partition[idx]

        if return_delete_inverse:
            return_step_delete = rng.randint(delete_steps + 1)
        else:
            return_step_delete = delete_steps

        inverse = action.Stop()

        for _ in range(return_step_delete):
            try:
                mol, inverse = me.delete_random_leaf(mol, rng=rng, return_inverse=True, vocab=self.vocab)
            except ValueError:
                pass

        if return_delete_inverse:
            # If we only need to compute deletions, return now.
            return mol, inverse

        return_step = rng.randint(insert_steps + 1)

        inverse = action.Switch()

        for _ in range(return_step):
            try:
                mol, inverse = me.insert_random_node(mol, self.vocab, rng, return_inverse=True)
            except ValueError:
                pass

        return mol, inverse
