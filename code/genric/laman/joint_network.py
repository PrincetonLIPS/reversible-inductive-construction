import typing
import torch

from .. import torch_ext
from ..model import message_passing, autograd_range, classification

from ..molecule_models.modules import FullyConnectedUnit, PartialLogitNetwork
from .representation import NODE_FDIM, EDGE_FDIM, LamanRep


class LamanEmbeddingNetwork(torch.nn.Module):
    def __init__(self):
        super(LamanEmbeddingNetwork, self).__init__()
    pass


class JointClassificationNetworkConfig:
    def __init__(self, depth, message_size):
        self.depth = depth
        self.message_size = message_size
        self.embedding_size = message_size
        self.hidden_size = message_size


def minmax(a, b):
    return torch.min(a, b), torch.max(a, b)


def readout_h1_features(embedding, graph: LamanRep):
    with autograd_range("h1_feature"):
        message_vertex = embedding.vertex_message

        indices = torch_ext.segment_triu_indices(
            graph.vertex.scope, offset=1, device=message_vertex.device)

        message_a = message_vertex.index_select(0, indices[:, 0])
        message_b = message_vertex.index_select(0, indices[:, 1])
        return torch.cat(minmax(message_a, message_b), dim=-1)


def readout_h2_features(embedding, graph: LamanRep):
    # H2 actions require a location on an edge, which is taken as undirected.
    # Because the messages naturally live on directed edges, we merge them first.
    edge_message_paired = embedding.edge_message.view(
        (2, embedding.edge_message.shape[0] // 2, embedding.edge_message.shape[1]))
    edge_message_undirected = edge_message_paired.mean(dim=0)

    with autograd_range("h2_feature"):
        return torch_ext.segment_cartesian_product(
            embedding.vertex_message, edge_message_undirected,
            graph.vertex.scope, graph.edge.scope // 2)


def readout_rev_h1_features(embedding, graph: LamanRep):
    with autograd_range("rev_h1_feature"):
        return embedding.vertex_message.index_select(0, graph.reverse_h1.index)


def readout_rev_h2_features(embedding, graph: LamanRep):
    with autograd_range("rev_h2_feature"):
        message_vertex = embedding.vertex_message
        message_node = message_vertex.index_select(0, graph.reverse_h2.index[:, 0])
        message_other1 = message_vertex.index_select(0, graph.reverse_h2.index[:, 1])
        message_other2 = message_vertex.index_select(0, graph.reverse_h2.index[:, 2])
        return torch.cat((message_node, *minmax(message_other1, message_other2)), dim=-1)


class LamanGraphReadout(torch.nn.Module):
    def __init__(self, message_size, embedding_size):
        super(LamanGraphReadout, self).__init__()

        self.pre_output = torch.nn.Sequential(
            torch.nn.Linear(message_size, embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_size, embedding_size // 2))

    def forward(self, vertex_message, vertex_scope):
        output = self.pre_output(vertex_message)
        avg_out = torch_ext.segment_avg_pool1d(output, vertex_scope)
        max_out = torch_ext.segment_max_pool1d(output, vertex_scope)

        return torch.cat((avg_out, max_out), dim=-1)


class LamanGraphEmbedding(typing.NamedTuple):
    embedding: torch.Tensor
    vertex_message: torch.Tensor
    edge_message: torch.Tensor


class JointClassificationNetwork(torch.nn.Module):
    def __init__(self, config: JointClassificationNetworkConfig):
        super(JointClassificationNetwork, self).__init__()

        self.mpn = message_passing.GeneralizedEmbeddingNetwork(
            depth=config.depth,
            edge_embedding_network=torch.nn.Sequential(
                torch.nn.Linear(EDGE_FDIM, config.message_size),
                torch.nn.Tanh()),
            message_aggregation_network=message_passing.MessageAggregationNetwork(config.message_size))

        self.graph_readout = LamanGraphReadout(config.message_size, config.embedding_size)

        task_networks = [
            ('stop', FullyConnectedUnit(config.embedding_size, lambda emb, graph: emb.embedding)),
            ('h1', PartialLogitNetwork(
                2 * config.message_size, config.embedding_size, 1, config.hidden_size,
                readout_h1_features, lambda graph: graph.action_scopes.h1)),
            ('h2', PartialLogitNetwork(
                2 * config.message_size, config.embedding_size, 1, config.hidden_size,
                readout_h2_features, lambda graph: graph.action_scopes.h2)),
            ('rev_h1', PartialLogitNetwork(
                config.message_size, config.embedding_size, 1, config.hidden_size,
                readout_rev_h1_features, lambda graph: graph.action_scopes.rev_h1)),
            ('rev_h2', PartialLogitNetwork(
                3 * config.message_size, config.embedding_size, 1, config.hidden_size,
                readout_rev_h2_features, lambda graph: graph.action_scopes.rev_h2))]

        self.classification_network = classification.MultiClassificationNetwork(
            task_networks, lambda graph: graph.action_scopes._asdict())

    def forward(self, graph: LamanRep):
        with autograd_range("message_passing"):
            vertex_message, edge_message = self.mpn(
                graph.vertex.feature, graph.edge.feature,
                graph.vertex.incidence, graph.edge.incidence, graph.vertex.scope)

        with autograd_range("graph_readout"):
            graph_embedding = self.graph_readout(vertex_message, graph.vertex.scope)

        embedding = LamanGraphEmbedding(graph_embedding, vertex_message, edge_message)

        with autograd_range("classification"):
            return self.classification_network(embedding, graph)
