# -*- coding: utf-8 -*-
import copy
import logging
import random

import networkx as nx

from federated_core.misc import tools

# logger = logging.getLogger('topology')

import os
from collections import namedtuple

import numpy as np

# helper functions
TopologyParams = namedtuple(typename='TopologyParams',
                            field_names=['num_nodes', 'density', 'mutation_prob'])


def probability_row(symmetry_matrix):
    sum_row = np.sum(symmetry_matrix, axis=-1, keepdims=True)
    non_zero_row = sum_row + (sum_row == 0)
    result = symmetry_matrix / non_zero_row

    return result


def topology_file_path(topology_params, num_topology):
    topology_directory = tools.get_topology_directory()
    num_nodes = topology_params.num_nodes
    density = topology_params.density
    mutation_prob = topology_params.mutation_prob

    file_name = 'gossip-{}nodes-{}density-{}muta-{}topology.npy'.format(num_nodes,
                                                                        density,
                                                                        mutation_prob,
                                                                        num_topology)
    file_path = os.path.join(topology_directory, file_name)

    return file_path


# initial adjacency matrix
def random_upper_triangle(num_nodes):
    rows, cols = np.indices(dimensions=(num_nodes, num_nodes))
    flatten_row = np.reshape(rows, newshape=(-1))
    flatten_col = np.reshape(cols, newshape=(-1))

    random_matrix = np.random.uniform(low=0, high=1., size=(num_nodes, num_nodes))
    flatten_value = np.reshape(random_matrix, newshape=(-1))

    random_triangle = [[r, c, v] for r, c, v in zip(flatten_row, flatten_col, flatten_value) if r < c]  # upper triangle
    return random_triangle


# 1. minimal spanning tree
def get_vertex_index_in_subgraphs(vertex, subgraphs):
    for index, subgraph in enumerate(subgraphs):
        if vertex in subgraph:
            return index
    return -1


def kruskal_minimal_spanning_tree(random_triangle, subgraphs):
    random_triangle = sorted(random_triangle, key=lambda x: x[-1], reverse=True)

    edge_set = set()
    while len(subgraphs) > 1:
        num_remain_edges = len(random_triangle)
        for edge_index in range(num_remain_edges - 1, -1, -1):
            row, col, _ = random_triangle[edge_index]
            random_triangle.pop(edge_index)

            row_index_in_subgraph = get_vertex_index_in_subgraphs(row, subgraphs)
            col_index_in_subgraph = get_vertex_index_in_subgraphs(col, subgraphs)
            assert row_index_in_subgraph != -1 and col_index_in_subgraph != -1

            if row_index_in_subgraph == col_index_in_subgraph:
                continue

            edge_set.add((row, col))

            if row_index_in_subgraph > col_index_in_subgraph:
                subgraph1 = subgraphs.pop(row_index_in_subgraph)
                subgraph2 = subgraphs.pop(col_index_in_subgraph)

            else:
                subgraph1 = subgraphs.pop(col_index_in_subgraph)
                subgraph2 = subgraphs.pop(row_index_in_subgraph)

            new_subgraph = subgraph1 | subgraph2
            subgraphs.append(new_subgraph)

            if len(subgraphs) <= 1:
                break

    return edge_set


# 2.1 mutation
def get_edge_index(random_triangle, edge):
    for idx, (r, c, _) in enumerate(random_triangle):
        if edge[0] == r and edge[1] == c:
            return idx


def mst_mutation_edges(random_triangle, mst_edges, num_mutation):
    mutation_edges = list()
    for _ in range(num_mutation):
        mutation_idx = random.randint(0, len(mst_edges) - 1)
        mutation_edge = mst_edges[mutation_idx]
        mutation_edges.append(mutation_edge)

    for mutation_edge in mutation_edges:
        edge_idx = get_edge_index(random_triangle, mutation_edge)
        random_triangle[edge_idx][-1] = random.random()

    return random_triangle


# 2. pad to connected edges
def check_density(num_nodes, mst_edges, density):
    all_edges = set([(r, c) for r, c, _ in random_upper_triangle(num_nodes)])
    num_pad_edges = int(density * num_nodes * (num_nodes - 1) / 2) - len(mst_edges)

    if num_pad_edges == 0:
        return mst_edges

    if num_pad_edges < 0:
        raise ValueError('Provided density is too small!')

    remain_edges = list(all_edges - mst_edges)
    edges_index = np.random.choice(a=range(len(remain_edges)), size=num_pad_edges, replace=False)
    pad_edges = [remain_edges[i] for i in edges_index]

    density_edges = mst_edges | set(pad_edges)

    return density_edges


# 3. complete graph
def symmetrical_matrix(num_nodes, density_edges):
    adjacency_matrix = np.zeros(shape=(num_nodes, num_nodes))

    row_col_index = ([rc[0] for rc in density_edges],
                     [rc[1] for rc in density_edges])
    adjacency_matrix[row_col_index] = 1.

    diag_index = np.diag_indices_from(adjacency_matrix)
    adjacency_matrix[diag_index] = 1.

    transpose_row_col_index = ([rc[1] for rc in density_edges],
                               [row_col[0] for row_col in density_edges])
    adjacency_matrix[transpose_row_col_index] = 1.

    return adjacency_matrix


def sequential_topology_generator(topology_params):
    num_nodes = topology_params.num_nodes
    density = topology_params.density
    mutation_prob = topology_params.mutation_prob

    random_triangle = random_upper_triangle(num_nodes)
    while True:
        dummy_triangle = copy.deepcopy(random_triangle)
        subgraphs = [set([i]) for i in range(num_nodes)]
        mst_edges = kruskal_minimal_spanning_tree(dummy_triangle, subgraphs)

        density_edges = check_density(num_nodes, mst_edges, density)

        symmetrical_graph = symmetrical_matrix(num_nodes, density_edges)

        probability_graph = probability_row(symmetrical_graph)

        yield probability_graph

        mst_edges = list(mst_edges)
        num_mutation = int(num_nodes * mutation_prob)
        random_triangle = mst_mutation_edges(random_triangle, mst_edges, num_mutation)


# 4. API
def generate_sequential_topology(topology_params, num_topology):
    topology_iter = sequential_topology_generator(topology_params)

    topologies = list()
    while len(topologies) < num_topology + 1:
        print('\rGenerating process: {}->{}'.format(len(topologies), num_topology), end='')

        topology_candidate = next(topology_iter)

        nx_topology = nx.from_numpy_array(topology_candidate)
        if nx.number_connected_components(nx_topology) == 1:
            topologies.append(topology_candidate)
    print()

    file_path = topology_file_path(topology_params, num_topology)
    tools.careful_file_path(file_path)
    np.save(file_path, np.array(topologies))


def test():
    '''
    20 clients, 0.5 density
    50 clients, 0.5 density
    100clients, 0.5 density
    '''
    example_topology_params = TopologyParams(num_nodes=20, density=0.5, mutation_prob=0.1)
    example_num_topology = 3000
    generate_sequential_topology(example_topology_params, example_num_topology)


if __name__ == '__main__':
    test()
