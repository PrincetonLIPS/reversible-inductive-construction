import contextlib
import torch


@contextlib.contextmanager
def autograd_range(name):
    """ Creates an autograd range for pytorch autograd profiling
    """
    torch.autograd._push_range(name)
    yield
    torch.autograd._pop_range()
