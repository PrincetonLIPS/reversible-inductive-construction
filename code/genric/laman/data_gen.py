import pickle
import contextlib
import numpy as np
import networkx as nx

from random import random
from tqdm import trange

from .laman_edit import generate_random_HI, generate_random_HII, compute_HI, compute_HII


@contextlib.contextmanager
def _open_maybe_compressed(path, mode):
    if path.endswith('.gz'):
        with gzip.open(path, mode) as f:
            yield f
    else:
        with open(path, mode) as f:
            yield f


def RH(n, p, rng=np.random):
    """ Generate random graph based on n, the number of vertices, 
    and p, the probability of choosing HI (Algorithm 7).
    """
    # Initialize adjacency matrix to K3
    G = nx.Graph()
    for i in range(3):
        for j in range(i+1, 3):
            G.add_edge(i, j)
    # Connect additional edges via HI or HII
    for i in range(3, n):
        r = rng.uniform()
        if r < p:
            act = generate_random_HI(G, rng=rng)
            G = compute_HI(G, act)
        else:
            act = generate_random_HII(G, rng=rng)
            G = compute_HII(G, act)
    return G


def generate_dataset(num_graphs, size_dist=None, p_dist=None, rng=np.random, filename=None):
    if not size_dist:
        def size_dist(rng):
            return int(np.round(rng.normal(loc=30, scale=5)))
    if not p_dist:
        def p_dist(rng):
            return rng.uniform(low=0.4, high=0.6)
    data = []
    for i in trange(num_graphs):
        this_size = size_dist(rng)
        this_p = p_dist(rng)
        data.append(RH(this_size, this_p, rng=rng))

    if filename is not None:
        with _open_maybe_compressed(filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    return data


_dod_p_dist = {
    'med': lambda rng: rng.uniform(0.4, 0.6),
    'low': lambda rng: rng.uniform(0.0, 0.1),
    'high': lambda rng: rng.uniform(0.9, 1.0)
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',
        default='../data/laman/low_decomp_dataset.pkl.gz',
        help='The output file. If it ends in .gz, will be compressed using gzip')

    parser.add_argument('--num-samples', type=int, default=10000)
    parser.add_argument('--dod', choices=['low', 'med', 'high'], default='low',
        help='The distribution of the degree of decomposability.')

    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        rng = np.random.RandomState(args.seed)
    else:
        rng = np.random


    generate_dataset(args.num_samples, filename=args.output, p_dist=_dod_p_dist[args.dod], rng=rng)


if __name__ == '__main__':
    main()