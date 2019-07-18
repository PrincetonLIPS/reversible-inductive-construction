import numpy as np
from .. import action
from ._representation import minmax, _get_graph_edges, get_reverse_h1_location


def get_action_offsets(graph):
    num_nodes = graph.number_of_nodes()
    degrees = np.array(graph.degree)[:, 1]

    lengths = [
        1,
        num_nodes * (num_nodes - 1) // 2,
        num_nodes * graph.number_of_edges(),
        np.sum(degrees == 2),
        np.sum(degrees == 3) * 3
    ]

    return np.concatenate(([0], np.cumsum(np.array(lengths))))


def encode_action(act: action.Action, graph, edges=None, include_stop=True):
    edges = _get_graph_edges(graph, edges)
    offset = 0

    num_nodes = graph.number_of_nodes()

    if include_stop:
        # If our encoding includes stop, it is at the first index.
        if isinstance(act, action.Stop):
            return 0
        offset += 1

    if isinstance(act, action.HI):
        # HI actions are encoded in the canonical form (a, b) with a < b
        # and then linearized in lexicographic order.
        a, b = minmax((act.node_a, act.node_b))
        return offset + (b - a - 1) + (num_nodes * (num_nodes - 1)) // 2 - (num_nodes - a) * (num_nodes - a - 1) // 2

    offset += num_nodes * (num_nodes - 1) // 2

    if isinstance(act, action.HII):
        # This encoding is over-valid, as it also encodes
        # actions when the solo node is part of the edge, which is illegal.
        edge = minmax(act.edge)
        edge_idx = edges[edge]

        return offset + act.solo_node * len(edges) + edge_idx

    offset += num_nodes * len(edges)

    rev_h1_locations = get_reverse_h1_location(graph)

    if isinstance(act, action.rev_HI):
        return offset + np.searchsorted(rev_h1_locations, act.node)

    offset += len(rev_h1_locations)

    if isinstance(act, action.rev_HII):
        degree = np.array(graph.degree)
        nodes3 = degree[degree[:, 1] == 3, 0]
        node3_idx = np.searchsorted(nodes3, act.node)

        connect = minmax(act.neighbors_to_connect)
        a, b, c = sorted(graph.neighbors(act.node))

        if connect == (a, b):
            neighbor_idx = 0
        elif connect == (a, c):
            neighbor_idx = 1
        elif connect == (b, c):
            neighbor_idx = 2
        else:
            raise ValueError("Error in encoding for action")

        return offset + 3 * node3_idx + neighbor_idx

    raise ValueError("Unknown action type")


def decode_action(value, graph):
    edges = _get_graph_edges(graph)
    offsets = get_action_offsets(graph)
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    if value == 0:
        return action.Stop()

    if value < offsets[2]:
        value -= offsets[1]
        a, b = np.stack(np.triu_indices(num_nodes, 1))[:, value]
        return action.HI(a, b)

    if value < offsets[3]:
        value -= offsets[2]

        node_idx = value // num_edges
        edge_idx = value % num_edges

        solo_node = list(graph.nodes)[node_idx]
        edge = list(graph.edges)[edge_idx]

        if solo_node in edge:
            return None
        return action.HII(edge, solo_node)

    if value < offsets[4]:
        value -= offsets[3]
        rev_h1_locations = get_reverse_h1_location(graph)

        return action.rev_HI(rev_h1_locations[value])

    if value < offsets[5]:
        value -= offsets[4]

        node_idx = value // 3
        edge_idx = value % 3
        degree = np.array(graph.degree)
        node = degree[degree[:, 1] == 3, 0][node_idx]

        a, b, c = sorted(graph.neighbors(node))

        if edge_idx == 0:
            edge = (a, b)
        elif edge_idx == 1:
            edge = (a, c)
        else:
            edge = (b, c)

        return action.rev_HII(node, edge)

    raise ValueError("Invalid action index.")
