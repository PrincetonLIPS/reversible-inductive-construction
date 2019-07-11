import torch
import torch.nn
import collections


from .. import molecule_representation as mr
from . import autograd_range


def aggregate_by_incidence_impl(values: torch.Tensor, incidence: torch.Tensor, scale=None, output_size=None):
    if output_size is None:
        output_size = values.shape[0]

    output = values.new_zeros([output_size] + list(values.shape[1:]))
    values_duplicated = values.index_select(0, incidence[1])

    if scale is not None:
        values_duplicated = values_duplicated.mul_(scale.unsqueeze(1))

    output.index_add_(0, incidence[0], values_duplicated)
    return output


def aggregate_by_incidence(values: torch.Tensor, incidence):
    if isinstance(incidence, torch.Tensor):
        if incidence.layout != torch.sparse_coo:
            return torch.mm(incidence, values)
        else:
            return aggregate_by_incidence_impl(values, incidence._indices(), incidence._values(), incidence.shape[0])
    elif isinstance(incidence, collections.Sequence):
        idx, scale, size = incidence
        return aggregate_by_incidence_impl(values, idx, scale, size[0])
    else:
        raise ValueError("Unknown type of incidence.")


class GeneralizedEmbeddingNetwork(torch.jit.ScriptModule):
    """ Custom configurable message-passing network.

    This class implements the main plumbing for a message passing network on the molecule,
    but exposes points that can be configured to easily create different variants of the networks.
    """
    def __init__(self, depth, edge_embedding_network, message_aggregation_network, batch_size=None):
        """ Creates a new module representing the message passing network.

        Parameters
        ----------
        depth: number of message passing iterations to execute.
        edge_embedding_network: a `torch.nn.Module` representing the model used to
            compute edge (=bond) embeddings from bond features. This model receives
            an array of bond features, and should return the corresponding edge embeddings.
        message_aggregation_network: a `torch.nn.Module` representing the model used to
            compute the message to be used at the next step. This model receives the array
            of messages corresponding to the sum of the propagated messages, and the array
            of initial edge embeddings.
        """
        super(GeneralizedEmbeddingNetwork, self).__init__()

        self.depth = depth
        self.edge_embedding_network = edge_embedding_network
        self.message_aggregation_network = message_aggregation_network

        self.message_size = self.message_aggregation_network.message_size

    __constants__ = ["batch_size", "depth"]

    def _propagate_messages(self, edge_embedding, bond_incidence):
        message = edge_embedding

        for i in range(self.depth - 1):
            message = aggregate_by_incidence(message, bond_incidence)
            message = self.message_aggregation_network(message, edge_embedding)

        return message

    def forward(self, atom_features, bond_features, atom_incidence, bond_incidence, atom_scope):
        # Compute edge embeddings from features. These are also used as initial message.
        edge_embedding = self.edge_embedding_network(bond_features)

        # Compute message passing along edges
        with autograd_range("propagate_messages"):
            bond_message = self._propagate_messages(edge_embedding, bond_incidence)

        # Compute message values at each atom by aggregating from the edges.
        atom_message = aggregate_by_incidence(bond_message, atom_incidence)

        return atom_message, bond_message


class MessageAggregationNetwork(torch.jit.ScriptModule):
    def __init__(self, message_size, nonlinearity=torch.nn.ReLU()):
        super(MessageAggregationNetwork, self).__init__()
        self.dense = torch.nn.Linear(message_size, message_size, bias=False)
        self.nonlinearity = nonlinearity
        self.message_size = message_size

    @torch.jit.script_method
    def forward(self, message, edge_embedding):
        message = self.dense(message)
        return self.nonlinearity(message + edge_embedding)


def make_action_mpn(hidden_size=128, depth=5):
    parameters = {
        'edge_embedding_network': torch.nn.Sequential(
            torch.nn.Linear(mr.ATOM_FDIM + mr.BOND_FDIM, hidden_size),
            torch.nn.ReLU()),
        'message_aggregation_network': MessageAggregationNetwork(hidden_size, nonlinearity=torch.nn.ReLU()),
    }

    return GeneralizedEmbeddingNetwork(depth, **parameters)
