from random import random
from copy import deepcopy
import numpy as np
import networkx as nx

import pyximport; pyximport.install()
from . import pebble, action

from IPython import embed


def get_new_node(G):
    return max(list(G.nodes)) + 1


def get_nodes_of_degree(G, degree):
    """ Returns a list of all nodes from graph G of the specified degree.
    """
    nodes = []
    for node, d in G.degree:
        if d == degree:
            nodes.append(node)
    return nodes


def compute_HI(G, act: action.HI, return_inverse=False):
    G = deepcopy(G)
    new_node = get_new_node(G)
    for partner_node in [act.node_a, act.node_b]:
        G.add_edge(partner_node, new_node)

    if return_inverse:
        inverse_action = action.rev_HI(new_node)
        return G, inverse_action
    else:
        return G


def compute_HII(G, act: action.HII, return_inverse=False):
    G = deepcopy(G)
    new_node = get_new_node(G)
    # Delete partner edge
    G.remove_edge(act.edge[0], act.edge[1])
    # Add three new edges
    for node in [act.edge[0], act.edge[1], act.solo_node]:
        G.add_edge(node, new_node)

    if return_inverse:
        inverse_action = action.rev_HII(new_node, act.edge)
        return G, inverse_action
    else:
        return G


def compute_insert(G, act: action.Insert, return_inverse=False):
    """ Computes the result of executing the given insertion action.
    """
    if isinstance(act, action.HI):
        return compute_HI(G, act, return_inverse=return_inverse)
    elif isinstance(act, action.HII):
        return compute_HII(G, act, return_inverse=return_inverse)
    else:
        raise ValueError("Expected action to be instance of either HI or HII")


def compute_action(G, act, return_inverse=False):
    if isinstance(act, action.Insert):
        return compute_insert(G, act, return_inverse=return_inverse)
    elif isinstance(act, action.Delete):
        return compute_delete(G, act, return_inverse=return_inverse)
    else:
        raise ValueError("Expected action to be instance of action.Continue")


def generate_random_HI(G, rng=np.random):
    node_a, node_b = rng.choice(G.nodes, size=2, replace=False)
    return action.HI(node_a, node_b)


def generate_random_HII(G, rng=np.random):
    # Find partner edge
    # We limit number of retries at both steps.
    # If all retries fail, we return None.
    for _ in range(5):
        edge = list(G.edges)[rng.choice(len(G.edges))]
        # Find solo partner node
        for _ in range(5):
            partner_node = rng.choice(G.nodes)
            if partner_node not in edge:
                return action.HII(edge, partner_node)

    return None


def insert_random_node(G, rng=np.random, return_action=False, return_inverse=False, p_HI=0.5):
    """ Inserts a node onto the graph randomly. Action HI is chosen with probability `p_HI`.
    """
    def generate_random_insert():
        if rng.uniform() > p_HI:
            ret_or_none = generate_random_HII(G, rng=rng)
        else:
            ret_or_none = None

        if ret_or_none is None:
            return generate_random_HI(G, rng=rng)
        else:
            return ret_or_none

    act = generate_random_insert()
    results = compute_insert(G, act, return_inverse=return_inverse)

    if return_inverse:
        result_G, inverse = results
    else:
        result_G = results

    result = [result_G]
    if return_action:
        result += [act]
    if return_inverse:
        result += [inverse]

    return result[0] if len(result) == 1 else result


def renumber(G):
    G = deepcopy(G)
    renumber_map = {}
    for idx, ori_node in enumerate(list(G.nodes)):
        if idx != ori_node:
            neighbors = list(G[ori_node])
            new_node = idx
            renumber_map[ori_node] = new_node
            G.remove_node(ori_node)
            for neighbor in neighbors:
                G.add_edge(new_node, neighbor)
        else:
            renumber_map[ori_node] = ori_node
    return G, renumber_map


def compute_rev_HI(G, act: action.rev_HI, return_inverse=False):
    G = deepcopy(G)
    if G.degree[act.node] != 2:
        raise ValueError("Removed node must have degree 2 for rev_HI.")

    if return_inverse:
        node_a, node_b = list(G[act.node])
        inverse_action = action.HI(node_a, node_b)

    G.remove_node(act.node)

    G, renumber_map = renumber(G)
    if return_inverse:
        inverse_action.node_a = renumber_map[inverse_action.node_a]
        inverse_action.node_b = renumber_map[inverse_action.node_b]

    if return_inverse:
        return G, inverse_action
    else:
        return G


def compute_rev_HII(G, act: action.rev_HII, return_inverse=False):
    G = deepcopy(G)
    if G.degree[act.node] != 3:
        raise ValueError("Removed node must have degree 3 for rev_HII.")

    if return_inverse:
        adj_nodes = list(G[act.node])
        for adj_node in adj_nodes:
            if adj_node not in act.neighbors_to_connect:
                solo_node = adj_node
                break
        inverse_action = action.HII(act.neighbors_to_connect, solo_node)

    G.remove_node(act.node)
    G.add_edge(act.neighbors_to_connect[0], act.neighbors_to_connect[1])

    if not is_laman(G):
        raise ValueError("Invalid rev_HII.")

    G, renumber_map = renumber(G)
    if return_inverse:
        inverse_action.solo_node = renumber_map[inverse_action.solo_node]
        for n in range(2):
            inverse_action.edge[n] = renumber_map[inverse_action.edge[n]]
    
    if return_inverse:
        return G, inverse_action
    else:
        return G


def generate_random_rev_HI(G, rng=np.random):
    node_cands = get_nodes_of_degree(G, 2)
    if not node_cands:
        return None
    node = rng.choice(node_cands)
    return action.rev_HI(node)


def generate_random_rev_HII(G, rng=np.random):
    node_cands = get_nodes_of_degree(G, 3)
    if not node_cands:
        return None
    node = rng.choice(node_cands)
    G_copy = deepcopy(G)
    adj_nodes = list(G_copy[node])
    G_copy.remove_node(node)
    # See where the edge should be added
    neigh_connect_options = []
    for i in range(len(adj_nodes)):
        for j in range(i+1, len(adj_nodes)):
            a = adj_nodes[i]
            b = adj_nodes[j]
            if not G_copy.has_edge(a, b):
                G_copy.add_edge(a, b)
                if is_laman(G_copy):
                    neigh_connect_options.append([a, b])
                G_copy.remove_edge(a, b)
    neighs_to_connect = neigh_connect_options[rng.choice(len(neigh_connect_options))]
    return action.rev_HII(node, neighs_to_connect)


def compute_delete(G, act: action.Delete, return_inverse=False):
    """ Computes the result of executing the given deletion action.
    """
    if isinstance(act, action.rev_HI):
        return compute_rev_HI(G, act, return_inverse=return_inverse)
    elif isinstance(act, action.rev_HII):
        return compute_rev_HII(G, act, return_inverse=return_inverse)
    else:
        raise ValueError("Expected action to be instance of either rev_HI or rev_HII")


def delete_random_node(G, rng=np.random, return_action=False, return_inverse=False, p_rev_HI=0.5):
    """ Deletes a random node of degree 2 or 3. Action rev_HI is chosen with probability `p_rev_HI`.
    """
    def generate_random_delete():
        if rng.uniform() < p_rev_HI:
            return generate_random_rev_HI(G, rng=rng)
        else:
            return generate_random_rev_HII(G, rng=rng)
    act = generate_random_delete()
    if act is None:
        if get_nodes_of_degree(G, 2):
            act = generate_random_rev_HI(G, rng=rng)
        else:
            act = generate_random_rev_HII(G, rng=rng)
    results = compute_delete(G, act, return_inverse=return_inverse)

    if return_inverse:
        result_G, inverse = results
    else:
        result_G = results

    result = [result_G]
    if return_action:
        result += [act]
    if return_inverse:
        result += [inverse]

    return result[0] if len(result) == 1 else result


def apply_random_corruptions(G, expected_corruption_steps=5, rng=np.random):
    """ Applies a random sequence of corruptions to G.
    """
    number_of_steps = rng.geometric(1 / (1 + expected_corruption_steps)) - 1
    for _ in range(number_of_steps):
        if rng.uniform() < 0.5 and len(G.nodes) > 3:
            G, _ = delete_random_node(G, rng=rng, return_inverse=True)
        else:
            G, _ = insert_random_node(G, rng=rng, return_inverse=True)
    return G


def is_laman(G):
    """ Returns true if G satisfies the Laman conditions, else false.
    """
    lat = pebble.lattice()

    for x, y in G.edges:
        if not lat.add_bond(x, y):
            return False
    if len(lat.bond) != 2 * len(lat.graph.keys()) - 3:
        return False

    return True
