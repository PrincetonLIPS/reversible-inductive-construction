import torch
import numpy as np
import collections
import numbers
from ..molecule_models import _train_utils


def cast_numpy_rec(x):
    if isinstance(x, np.ndarray):
        return _train_utils.cast_numpy_to_torch(x)
    elif isinstance(x, numbers.Number):
        return x
    elif hasattr(x, '_make'):
        return x._make(map(cast_numpy_rec, x))
    elif isinstance(x, collections.abc.Mapping):
        return {k: cast_numpy_rec(y) for k, y in x}
    elif isinstance(x, collections.abc.Iterable):
        return [cast_numpy_rec(y) for y in x]
    else:
        raise ValueError("Unknown type to cast {0}".format(type(x)))
