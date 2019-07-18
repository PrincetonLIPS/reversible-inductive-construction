""" This module contains classes describing the inductive moves used by the corrupter and denoiser. """

import abc


# TO-DO change action and cont to abstract
class Action(abc.ABC):
    @property
    @abc.abstractmethod
    def action_type(self):
        raise NotImplementedError()


class Continue(Action):
    pass


class Stop(Action):
    @property
    def action_type(self):
        return 0


class Delete(Continue):
    """ Deletion action.
    A deletion is the inverse of one of the two Henneberg moves.
    """
    pass


class Insert(Continue):
    """ Insertion action.
    An insertion is one of the two Henneberg moves.
    """
    pass


class HI(Insert):
    """ Henneberg move type I.

    Attributes:
    node_a: Any node in the graph.
    node_b: Any node in the graph other than node_a.
    """
    def __init__(self, node_a, node_b):
        if node_a == node_b:
            raise ValueError("Nodes must be different.")
        self.node_a = node_a
        self.node_b = node_b

    @property
    def action_type(self):
        return 1


class HII(Insert):
    """ Henneberg move type II.

    Attributes:
    edge: Any edge in the graph specified as a tuple of (node, node).
    solo_node: Any node in the graph not contained in the specified edge.
    """
    def __init__(self, edge, solo_node):
        if solo_node in edge:
            raise ValueError("solo_node must not be in edge.")
        self.edge = edge
        self.solo_node = solo_node

    @property
    def action_type(self):
        return 2


class rev_HI(Delete):
    """ The inverse of Henneberg move type I.

    Attributes:
    node: Any node of the degree 2 in the graph.
    """
    def __init__(self, node):
        self.node = node

    @property
    def action_type(self):
        return 3


class rev_HII(Delete):
    """ The inverse of Henneberg move type II.

    Attributes:
    node: Any node of degree 3 in the graph.
    neighs_to_connect: Two of the node's neighbors specified as a tuple indicating the new edge to add.
        Not all combinations of neighbors will result in a valid Laman graph.
    """
    def __init__(self, node, neighbors_to_connect):
        self.node = node
        self.neighbors_to_connect = neighbors_to_connect

    @property
    def action_type(self):
        return 4
