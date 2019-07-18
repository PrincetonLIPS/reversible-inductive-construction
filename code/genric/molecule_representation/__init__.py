""" Module implementing functions to compute representations of molecules. """

import typing
from functools import partial
import torch
from ..chemutils import get_mol, get_mol_2D
from .. import chemutils
from ..vocabulary import AtomTuple
import numpy as np

from collections import OrderedDict

from ._types import AtomInfo, BondInfo, ScopedTuple, LeafInfo, RingInfo, GraphInfo
from ._representation import ATOM_FDIM, BOND_FDIM
from ._representation import mol2graph_single, bond_incidence_list, atom_bond_list, bond_features, atom_features
from ._representation import atom_bond_list_segment, bond_incidence_list_segment
from ._aggregration import combine_mol_graph, dict_to_graph_data
