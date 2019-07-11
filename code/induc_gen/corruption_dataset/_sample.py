import torch
import itertools


class RoundRobinSampler(torch.utils.data.Sampler):
    """ Utility class which combines several samplers by sampling
    from each one in turn.

    """
    def __init__(self, *samplers):
        self.samplers = samplers

    def __iter__(self):
        zipped = itertools.zip_longest(*self.samplers)
        all_samples = itertools.chain.from_iterable(zipped)

        for sample in all_samples:
            if sample is not None:
                yield sample

    def __len__(self):
        return sum(len(sampler) for sampler in samplers)


class PartitionDatasetSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        samplers = [
            torch.utils.data.SubsetRandomSampler(idx)
            for idx in self.dataset.partition_index]

        batch_samplers = [
            torch.utils.data.BatchSampler(sampler, self.batch_size, drop_last=True)
            for sampler in samplers]

        round_robin_sampler = RoundRobinSampler(*batch_samplers)
        return iter(round_robin_sampler)

    def __len__(self):
        return sum(len(idx) // self.batch_size for idx in self.dataset.partition_index)


class RepeatDataset(torch.utils.data.Dataset):
    """ Simple dataset which repeats a given dataset. """
    def __init__(self, dataset, repeats):
        self.dataset = dataset
        self.repeats = repeats

    def __getitem__(self, idx):
        if idx >= self.repeats * len(self.dataset):
            raise IndexError()

        return self.dataset[idx % len(self.dataset)]

    def __len__(self):
        return self.repeats * len(self.dataset)
