import numpy as np
import networkx as nx

from collections import OrderedDict

NODE_FDIM = 12
EDGE_FDIM = 2 * NODE_FDIM


def minmax(it):
    return min(it), max(it)


def get_edge_incidence_size(graph):
    degrees = np.array(graph.degree)[:, 1]
    return np.sum(degrees * (degrees - 1))


def _get_graph_edges(graph, edges=None):
    if edges is None:
        edges = OrderedDict([(minmax(e), i) for i, e in enumerate(graph.edges)])
    return edges


def get_edge_incidence_list(graph, edges=None):
    edges = _get_graph_edges(graph, edges)
    len_edge_incidence_list = get_edge_incidence_size(graph)

    bond_incidence_idx = np.empty((2, len_edge_incidence_list), dtype=np.int32)
    edge_incidence_values = np.empty(len_edge_incidence_list, dtype=np.float32)

    current_idx = 0

    def _fill_bond_incidence(j, edge, a, value):
        nonlocal current_idx

        for a2 in graph.neighbors(a):
            if a2 in edge:
                continue

            assert current_idx < len_edge_incidence_list
            bond2 = minmax((a, a2))
            bond_incidence_idx[0, current_idx] = j
            bond_incidence_idx[1, current_idx] = 2 * edges[minmax((a, a2))] + int(bond2[0] == a)
            edge_incidence_values[current_idx] = value
            current_idx += 1

    for edge, i in edges.items():
        a1, a2 = edge
        degree = graph.degree[a1] + graph.degree[a2] - 2
        value = 1 / np.sqrt(degree)

        _fill_bond_incidence(2 * i, edge, a1, value)
        _fill_bond_incidence(2 * i + 1, edge, a2, value)

    num_edges = graph.number_of_edges()
    shape = [2 * num_edges, 2 * num_edges]

    return bond_incidence_idx, edge_incidence_values, shape


def get_vertex_incidence_list(graph, edges=None):
    edges = _get_graph_edges(graph, edges)
    vertex_incidence_idx = np.empty((2, 2 * graph.number_of_edges()), dtype=np.int32)
    vertex_incidence_values = np.empty(2 * graph.number_of_edges(), dtype=np.float32)

    current_idx = 0
    offset = 0

    for i, v in enumerate(graph.nodes):
        for j, v2 in enumerate(graph.neighbors(v)):
            edge = minmax((v, v2))
            edge_idx = edges[edge]

            vertex_incidence_idx[0, current_idx] = i
            vertex_incidence_idx[1, current_idx] = 2 * edge_idx + int(edge[0] == v)
            vertex_incidence_values[current_idx] = 1 / np.sqrt(graph.degree[v])
            current_idx += 1

    shape = [graph.number_of_nodes(), 2 * graph.number_of_edges()]
    return vertex_incidence_idx, vertex_incidence_values, shape


def get_vertex_features(graph, feature_type='zero'):
    if feature_type == 'zero':
        return np.zeros((graph.number_of_nodes(), NODE_FDIM), dtype=np.float32)
    elif feature_type == 'degree_fourier':
        degree = np.array(graph.degree)[:, 1]
        degree_rel = degree / np.max(degree)
        degree_rel = np.expand_dims(degree_rel, 1)

        frequency = np.expand_dims(2 * np.pi * 2 ** np.arange(NODE_FDIM), 0)
        return np.cos(degree_rel * frequency).astype(np.float32)
    else:
        raise ValueError("Unknown vertex feature type.")


def get_edge_features(graph, vertex_features):
    result = np.empty((2 * graph.number_of_edges(), EDGE_FDIM), dtype=np.float32)

    for i, edge in enumerate(graph.edges):
        a, b = minmax(edge)
        result[2 * i, :NODE_FDIM] = vertex_features[a]
        result[2 * i, NODE_FDIM:] = vertex_features[b]

        result[2 * i + 1, :NODE_FDIM] = vertex_features[b]
        result[2 * i + 1, NODE_FDIM:] = vertex_features[a]

    return result


def get_reverse_h1_location(graph):
    degrees = np.array(graph.degree)
    return degrees[degrees[:, 1] == 2, 0]


def get_reverse_h2_location(graph):
    degrees = np.array(graph.degree)

    nodes = degrees[degrees[:, 1] == 3, 0]

    result = np.empty((3 * len(nodes), 3), dtype=np.int32)

    for i, node in enumerate(nodes):
        na, nb, nc = graph.neighbors(node)

        result[3 * i, :] = [node, na, nb]
        result[3 * i + 1, :] = [node, na, nc]
        result[3 * i + 2, :] = [node, na, nc]

    return result


def graph_to_rep(graph, edges=None):
    edges = _get_graph_edges(graph)

    vertex_features = get_vertex_features(graph, feature_type='degree_fourier')
    edge_features = get_edge_features(graph, vertex_features)

    vertex_incidence = get_vertex_incidence_list(graph, edges)
    edge_incidence = get_edge_incidence_list(graph, edges)

    reverse_h1_location = get_reverse_h1_location(graph)
    reverse_h2_location = get_reverse_h2_location(graph)

    return {
        'vertex_feature': vertex_features,
        'edge_feature': edge_features,
        'vertex_incidence': vertex_incidence,
        'edge_incidence': edge_incidence,
        'reverse_h1_location': reverse_h1_location,
        'reverse_h2_location': reverse_h2_location
    }
