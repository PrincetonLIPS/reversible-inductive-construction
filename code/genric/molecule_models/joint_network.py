import torch
import torch.nn
from .. import model

from ..model.message_passing import make_action_mpn
from ..model.readout import MoleculeEmbeddingNetwork
from ..model.classification import MultiClassificationNetwork
from .modules import FullyConnectedUnit, PartialLogitNetwork, get_scopes_from_graph

import enum


""" Model for full joint training of action and location """


class JointClassificationNetworkConfiguration:
    def __init__(self, vocab_atom_action_size, vocab_bond_action_size, hidden_size,
                 depth=5, molecule_embedding_size=None, version=1):
        self.vocab_atom_action_size = vocab_atom_action_size
        self.vocab_bond_action_size = vocab_bond_action_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.molecule_embedding_size = molecule_embedding_size or hidden_size
        self.version = version


class JointClassificationNetwork(torch.nn.Module):
    """ Module for joint classification of molecule denoising.

    This module implements the main policy for a jonit classification
    strategy, where we predict a global action to be taken on the molecule
    among all possible actions.

    """
    def __init__(self, batch_size, config: JointClassificationNetworkConfiguration):
        super(JointClassificationNetwork, self).__init__()

        men = MoleculeEmbeddingNetwork(make_action_mpn(config.hidden_size, config.depth))
        self.molecule_embedding_network = men

        task_networks = [
            ('stop', FullyConnectedUnit(men.molecule_embedding_size, lambda emb, graph: emb.embedding)),
            ('delete_leaf_atom',
                PartialLogitNetwork(
                    men.leaf_atom_embedding_size, men.molecule_embedding_size, 1, config.hidden_size,
                    lambda emb, _: emb.leaf_embedding[0], lambda graph: graph.leaf_info.leaf_atom_info.scope)),
            ('delete_leaf_ring',
                PartialLogitNetwork(
                    men.leaf_ring_embedding_size, men.molecule_embedding_size, 1, config.hidden_size,
                    lambda emb, _: emb.leaf_embedding[1], lambda graph: graph.leaf_info.leaf_ring_info.scope)),
            ('insert_atom',
                PartialLogitNetwork(
                    men.message_size, men.molecule_embedding_size, config.vocab_atom_action_size, config.hidden_size,
                    lambda emb, _: emb.atom_message, lambda graph: graph.atom_info.atom_scope)),
            ('insert_bond',
                PartialLogitNetwork(
                    men.message_size, men.molecule_embedding_size, config.vocab_bond_action_size, config.hidden_size,
                    lambda emb, graph: torch.index_select(emb.bond_message, 0, graph.ring_info.ring_bond_idx.long()),
                    lambda graph: graph.ring_info.ring_scope))]

        self.classification_network = MultiClassificationNetwork(
            task_networks, get_scopes_from_graph, batch_size)

    def forward(self, graph):
        with model.autograd_range("molecule_embedding"):
            embedding = self.molecule_embedding_network(graph)

        with model.autograd_range("classification_network"):
            return self.classification_network(embedding, graph)


class SplitClassificationNetwork(torch.nn.Module):
    """ Module for split classification of molecule denoising.

    This module implements the main policy for a split classification
    strategy, where we first predict deletion type operations, and
    then predict insertion type operations.

    """
    class Mode(enum.IntEnum):
        Delete = 0
        Insert = 1

    def __init__(self, batch_size, vocab_atom_action_size, vocab_bond_action_size,
                 hidden_size=256, depth=5):
        super(SplitClassificationNetwork, self).__init__()

        men = MoleculeEmbeddingNetwork(make_action_mpn(hidden_size, depth))
        self.molecule_embedding_network = men

        task_delete_networks = [
            ('switch', FullyConnectedUnit(men.molecule_embedding_size, lambda emb, graph: emb.embedding)),
            ('delete_leaf_atom',
                PartialLogitNetwork(
                    men.leaf_atom_embedding_size, men.molecule_embedding_size, 1, hidden_size,
                    lambda emb, _: emb.leaf_embedding[0], lambda graph: graph.leaf_info.leaf_atom_info.scope)),
            ('delete_leaf_ring',
                PartialLogitNetwork(
                    men.leaf_ring_embedding_size, men.molecule_embedding_size, 1, hidden_size,
                    lambda emb, _: emb.leaf_embedding[1], lambda graph: graph.leaf_info.leaf_ring_info.scope))]

        task_insert_networks = [
            ('stop', FullyConnectedUnit(men.molecule_embedding_size, lambda emb, graph: emb.embedding)),
            ('insert_atom',
                PartialLogitNetwork(
                    men.message_size, men.molecule_embedding_size, vocab_atom_action_size, hidden_size,
                    lambda emb, _: emb.atom_message, lambda graph: graph.atom_info.atom_scope)),
            ('insert_bond',
                PartialLogitNetwork(
                    men.message_size, men.molecule_embedding_size, vocab_bond_action_size, hidden_size,
                    lambda emb, graph: torch.index_select(emb.bond_message, 0, graph.ring_info.ring_bond_idx.long()),
                    lambda graph: graph.ring_info.ring_scope))]

        self.delete_classification_network = MultiClassificationNetwork(
            task_delete_networks, get_scopes_from_graph, batch_size)
        self.insert_classification_network = MultiClassificationNetwork(
            task_insert_networks, get_scopes_from_graph, batch_size)

    def forward(self, graph, mode=Mode.Delete):
        embedding = self.molecule_embedding_network(graph)

        if mode == SplitClassificationNetwork.Mode.Delete:
            return self.delete_classification_network(embedding, graph)
        elif mode == SplitClassificationNetwork.Mode.Insert:
            return self.insert_classification_network(embedding, graph)
        else:
            raise ValueError("Unknown mode.")
