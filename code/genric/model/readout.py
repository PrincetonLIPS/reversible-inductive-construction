import torch
import torch.nn
import typing

from .. import molecule_representation as mr
from .nn import AvgAndMaxPool
from . import autograd_range

from .. import torch_ext

from .message_passing import aggregate_by_incidence


class AtomOutputNetwork(torch.nn.Module):
    """ Network which takes in atom messages, and produces embeddings for the atoms. """
    def __init__(self, message_size, atom_embedding_size=256):
        super(AtomOutputNetwork, self).__init__()
        self.feature_embedding = torch.nn.Sequential(
            torch.nn.Linear(mr.ATOM_FDIM, atom_embedding_size),
            torch.nn.ReLU())

        self.message_size = message_size
        self.output_size = atom_embedding_size

        self.data_to_feature = torch.nn.Sequential(
            torch.nn.Linear(atom_embedding_size + message_size, atom_embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(atom_embedding_size, atom_embedding_size),
            torch.nn.Tanh())

    def forward(self, atom_features, atom_message):
        atom_feat_embedding = self.feature_embedding(atom_features)
        atom_data = torch.cat([atom_feat_embedding, atom_message], dim=1)

        return self.data_to_feature(atom_data)


class MoleculeReadout(torch.nn.Module):
    """ A module which implements the readout for the global molecule embedding. """
    def __init__(self, message_size, embedding_size=256, batch_size=None):
        super(MoleculeReadout, self).__init__()

        self.atom_output_network = AtomOutputNetwork(message_size, embedding_size // 2)
        self.embedding_size = embedding_size

    def forward(self, atom_feature, atom_message, atom_scope):
        # Compute embedding at each atom from message and features.
        atom_output = self.atom_output_network(atom_feature, atom_message)

        avg_pool = torch_ext.segment_avg_pool1d(atom_output, atom_scope)
        max_pool = torch_ext.segment_max_pool1d(atom_output, atom_scope)

        molecule_representations = torch.cat([avg_pool, max_pool], dim=1)

        return molecule_representations


class MoleculeEmbedding(typing.NamedTuple):
    embedding: torch.Tensor
    atom_message: torch.Tensor
    bond_message: torch.Tensor
    leaf_embedding: typing.Tuple[torch.Tensor, torch.Tensor]


class MoleculeEmbeddingNetwork(torch.nn.Module):
    """ Network which computes the necessary embeddings attached to the molecule.

    This network reads out the messages from a message passing network, and
    computes embeddings associated with all aspects of the molecule.
    """
    def __init__(self, message_passing_network):
        super(MoleculeEmbeddingNetwork, self).__init__()
        self.mpn = message_passing_network
        hidden_size = self.mpn.message_size

        self.molecule_readout = MoleculeReadout(hidden_size, hidden_size)

        self.leaf_atom_embedding_network = torch.nn.Sequential(
            torch.nn.Linear(mr.ATOM_FDIM + self.mpn.message_size, hidden_size),
            torch.nn.Tanh())

        self.ring_atom_embedding_network = torch.nn.Sequential(
            torch.nn.Linear(mr.ATOM_FDIM + self.mpn.message_size, hidden_size),
            torch.nn.Tanh())

        self.molecule_embedding_size = self.molecule_readout.embedding_size
        self.leaf_atom_embedding_size = hidden_size
        self.leaf_ring_embedding_size = hidden_size
        self.message_size = self.mpn.message_size

    def forward(self, graph: mr.GraphInfo):
        # Extract all model data
        atom_feature, atom_incidence, atom_scope = graph.atom_info
        bond_feature, bond_incidence, bond_scope = graph.bond_info
        (leaf_atom_idx, lead_atom_scope), (leaf_ring_idx, leaf_ring_scope) = graph.leaf_info

        # Compute message passing along molecules.
        atom_message, bond_message = self.mpn(
            atom_feature, bond_feature, atom_incidence, bond_incidence, atom_scope)

        with autograd_range("molecule_readout"):
            molecule_embedding = self.molecule_readout(atom_feature, atom_message, atom_scope)

        # Select and compute atom embeddings along atom leaves
        atom_message_and_feature = torch.cat((atom_feature, atom_message), dim=1)

        leaf_atom_atom_outputs = torch.index_select(atom_message_and_feature, 0, leaf_atom_idx)
        leaf_atom_embeddings = self.leaf_atom_embedding_network(leaf_atom_atom_outputs)

        # Aggregate and compute atom embeddings along ring leaves.
        leaf_ring_all_embeddings = self.ring_atom_embedding_network(atom_message_and_feature)
        leaf_ring_embeddings = aggregate_by_incidence(leaf_ring_all_embeddings, leaf_ring_idx)

        return MoleculeEmbedding(molecule_embedding, atom_message, bond_message,
                                 (leaf_atom_embeddings, leaf_ring_embeddings))
