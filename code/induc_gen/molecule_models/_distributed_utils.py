import typing
import torch
import torch.distributed

from functools import partial


class DistributedTrainingInfo(typing.NamedTuple):
    sync_url: str
    world_size: int
    rank: int
    local_rank: int

    def __bool__(self):
        return self.world_size > 1


def train_boostrap_distributed(parameters, train):
    world_size = parameters.get('world_size', 1)

    if world_size == 1:
        # Single-node training, nothing to do.
        parameters['rank'] = 0
        return train(parameters)

    parameters['rank'] = -1
    from torch.multiprocessing import spawn

    spawn(partial(train_distributed, train=train), nprocs=parameters['world_size'], args=(parameters,))


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


def train_distributed(local_rank, parameters, train):
    config = get_distributed_config(parameters, local_rank)
    initialize_distributed(config)
    train(parameters, config)
