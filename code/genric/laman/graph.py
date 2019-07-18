import numpy as np
from matplotlib import pyplot as plt

import networkx as nx

from IPython import embed

class Graph(object):

    def __init__(self):
        self.graph = {} # undirected graph (complete) - dict of sets

    def __str__(self):
        return str(self.graph)

    def clear(self):
        self.__init__()

    def permute(self):
        new_dict = {}
        perm = np.random.permutation(len(self.get_nodes()))
        for x in self.graph.keys():
            new_dict[perm[x]] = set([])
            for y in self.graph[x].copy():
                new_dict[perm[x]].add(perm[y])
        self.graph = new_dict

    def add_bond(self, x, y):

        """
        :param x: a site
        :param y: a different site
        """

        # no self-loop bonds:
        if x == y:
            raise ValueError('add_bond must have two different sites')

        # check if bond already exists
        if self.contains_bond(x, y):
            raise ValueError('bond already exists')

        # update complete graph
        if x not in self.graph:
            self.graph[x] = set([y])
        else:
            self.graph[x].add(y)
        if y not in self.graph:
            self.graph[y] = set([x])
        else:
            self.graph[y].add(x)

    def reduce_node_keys(self):
        # ensure no node index gaps
        new_dict = {}
        new_keys = {}
        for idx, x in enumerate(self.graph.keys()):
            new_keys[x] = idx
        for x in self.graph.keys():
            new_dict[new_keys[x]] = set([])
            for y in self.graph[x].copy():
                new_dict[new_keys[x]].add(new_keys[y])
        self.graph = new_dict

    def remove_bond(self, x, y):
        if not self.contains_bond(x, y):
            raise ValueError('bond does not exist')
        self.graph[x].remove(y)
        self.graph[y].remove(x)

        # Remove disconnected nodes
        if not self.graph[x]:
            self.graph.pop(x)
        if not self.graph[y]:
            self.graph.pop(y)

    def contains_bond(self, x, y):
        if x in self.graph:
            if y in self.graph[x]:
                return True
        return False

    def get_nodes(self):
        return list(self.graph.keys())

    def get_degrees(self):
        deg = np.zeros(len(self.get_nodes())).astype(int)  # to-do: optimize this
        for x in self.graph.keys():
            deg[x] = len(self.graph[x])
        return deg

    def get_bonds(self):
        bonds = set([])
        for x in self.graph:
            for y in self.graph[x]:
                b = tuple(sorted([x,y]))
                bonds.add(b)
        return bonds

    def draw(self, with_colors=False, with_labels=False, pos=None):
        nx_obj = nx.Graph()
        for x in self.graph:
            for y in self.graph[x]:
                nx_obj.add_edge(x,y)

        labels_dict = None
        if with_labels:
            labels_dict = {}
            for node in self.get_nodes():
                labels_dict[node] = str(node)

        color_map = None
        if with_colors:
            color_map = []
            for node in self.get_nodes()[:-1]:
                color_map.append('blue')
            # Make most last one green
            color_map.append('green')
        
        if pos:
            # If some positions are provided, keep these fixed
            fixed = self.get_nodes()
        else:
            fixed=None
        pos = nx.spring_layout(nx_obj, pos=pos, fixed=fixed)

        nx.draw(nx_obj, pos=pos, node_color=color_map, with_labels=with_labels)
        plt.ion()
        plt.show()

        return pos
