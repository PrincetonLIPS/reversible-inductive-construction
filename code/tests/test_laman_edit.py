import time
import numpy as np
import pickle as pkl
from copy import deepcopy
import networkx as nx
from genric.laman import laman_edit as le
from genric.laman import data_gen as dg

from IPython import embed


def test_is_laman():
    G = dg.RH(20, 0.5)
    tic = time.time()
    print("Running Laman checker")
    assert le.is_laman(G)
    elapsed = time.time() - tic
    print("Check took %.5f seconds \n" % elapsed)
    G.add_edge(100, 5)
    assert (le.is_laman(G) == False)


def test_deterministic_generation():
    G1 = dg.RH(20, 0.5, rng=np.random.RandomState(42))
    G2 = dg.RH(20, 0.5, rng=np.random.RandomState(42))
    assert nx.is_isomorphic(G1, G2)


def test_laman_after_corruption():
    G = dg.RH(20, 0.5)
    G2 = le.apply_random_corruptions(G, expected_corruption_steps=15)
    assert le.is_laman(G2)


def test_deterministic_corruption():
    G = dg.RH(20, 0.5)
    G2 = le.apply_random_corruptions(G, expected_corruption_steps=15, rng=np.random.RandomState(7))
    G3 = le.apply_random_corruptions(G, expected_corruption_steps=15, rng=np.random.RandomState(7))
    assert nx.is_isomorphic(G2, G3)


def test_repeat_action():
    G = dg.RH(20, 0.5)
    G2 = deepcopy(G)
    actions = []
    for _ in range(24):
        G, act = le.insert_random_node(G, return_action=True)
        actions.append(act)
    for _ in range(12):
        G, act = le.delete_random_node(G, return_action=True)
        actions.append(act)
    for act in actions:
        G2 = le.compute_action(G2, act)
    assert nx.is_isomorphic(G, G2)


def test_inverses():
    tic = time.time()
    G = dg.RH(30, 0.5)
    print('Graph generation took %.5f seconds' % (time.time() - tic))
    for _ in range(24):
        G_target = deepcopy(G)
        G, inv = le.insert_random_node(G, return_inverse=True)
        G_inv = le.compute_action(G, inv)
        assert nx.is_isomorphic(G_inv, G_target)
    for _ in range(12):
        G_target = deepcopy(G)
        G, inv = le.delete_random_node(G, return_inverse=True)
        G_inv = le.compute_action(G, inv)
        assert nx.is_isomorphic(G_inv, G_target)


def test_generate_dataset():
    filename = '/tmp/laman_set.obj'
    data = dg.generate_dataset(20, filename=filename)
    with open(filename, 'rb') as f:
        data_2 = pkl.load(f)
    for G, G2 in zip(data, data_2):
        assert nx.is_isomorphic(G, G2)


def test_renumber():
    G = dg.RH(30, 0.5)
    for _ in range(10):
        G = le.delete_random_node(G)
    for idx, node in enumerate(list(G.nodes)):
        assert idx == node
