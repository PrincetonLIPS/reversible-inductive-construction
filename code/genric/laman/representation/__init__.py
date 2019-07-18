""" Module for computing representations for Laman graphs that can be fed to
standard message passing models.
"""

from ._data import StructureRep, ScopedIndex, LamanRep
from ._representation import graph_to_rep, EDGE_FDIM, NODE_FDIM
from ._aggregation import combine_graph_reps
from ._action import encode_action, get_action_offsets, decode_action
