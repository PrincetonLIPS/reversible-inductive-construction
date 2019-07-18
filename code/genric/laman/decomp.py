import random
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

from genric.laman import data_gen

from IPython import embed


def to_bipartite(G):
    """
    Converts a Laman graph to bipartite form.

    Args:
        G (nx.Graph): The graph to be converted.

    Returns:
        nx.Graph: The graph in bipartite form as type nx.Graph.
    """
    B = nx.Graph()
    # Pin random bond
    bonds = tuple(G.edges)
    pinned_bond = random.choice(bonds)
    # For each unpinned bond, add associated nodes and itself.
    for idx, b in enumerate(bonds):
        if b == pinned_bond:
            continue
        nodes_to_add = []
        for node in b:
            if node in pinned_bond:
                continue
            nodes_to_add.extend(['%i_1' % node, '%i_2' % node])
        B.add_nodes_from(nodes_to_add, bipartite=0)
        bond_name = 'E%i' % idx
        B.add_node(bond_name, bipartite=1)
        for node in nodes_to_add:
            B.add_edge(node, bond_name)
    return B


def get_DoD(G):
    """
    Computes degree of decomposability of a Laman graph.

    Args:
        G (nx.Graph): The graph to be decomposed.

    Returns:
        float: The degree of decomposability.
    """
    B = to_bipartite(G)
    # Find perfect matching (pass in one of the sets in case B is disconnected)
    var_nodes = {n for n, d in B.nodes(data=True) if d['bipartite'] == 0}
    M = nx.bipartite.maximum_matching(B, var_nodes)
    # Build directed graph
    D = nx.DiGraph()
    for node_key in M.keys():
        D.add_edge(node_key, M[node_key])
    M_edges = set(D.edges())
    other_edges = set(B.edges()).difference(M_edges)
    for oe in other_edges:
        if 'E' in oe[0]:
            D.add_edge(oe[0], oe[1])
        else:
            D.add_edge(oe[1], oe[0])
    # Get number of strongly connected components
    strongly_conn_comps = list(nx.strongly_connected_components(D))
    num_comp = len(strongly_conn_comps)
    return num_comp / len(G.nodes())


def dod_vs_size_exp():
    sizes = range(3, 100, 2)
    num_trials = 10
    for p in [0, 0.5, 1.0]:
        dods = []
        for size in sizes:
            dod_sum = 0
            for trial in range(num_trials):
                G = data_gen.RH(size, p)
                dod_sum += get_DoD(G)
            dods.append(dod_sum / num_trials)
            print(size)
        plt.plot(sizes, dods, linestyle='--', marker='^', label='p = %.1f' % p)
    plt.legend(loc='upper right')
    plt.xlabel('Size of graph')
    plt.ylabel('Degree of decomposability')
    plt.show(block=False)
    plt.pause(0.001)
    plt.savefig('dod_vs_size.png')

if __name__ == '__main__':
    dod_vs_size_exp()


    # sizes = range(3, 100, 2)
    # num_size_trials = 15
    # num_instance_trials = 15
    # for p in [0, 0.5, 1.0]:
    #     dod_stds = []
    #     for size in sizes:
    #         std_sum = 0
    #         for trial in range(num_size_trials):
    #             G = data_gen.RH(size, p)
    #             this_graph_dods = []
    #             for i_trial in range(num_instance_trials):
    #                 this_graph_dods.append(get_DoD(G))
    #             std_sum += np.std(this_graph_dods)
    #         dod_stds.append(std_sum / num_size_trials)
    #         print(size)
    #     plt.plot(sizes, dod_stds, linestyle='--', marker='^', label='p = %.1f' % p)
    # plt.legend(loc='upper right')
    # plt.xlabel('Size of graph')
    # plt.ylabel('Standard deviation of DoD')
    # plt.show(block=False)
    # plt.pause(0.001)
    # plt.savefig('std_dod_vs_size.png')

    # embed()
