# -*- coding: utf-8 -*-
import os

import numpy as np

from federated_core.misc import tools


def topology_file_path(num_nodes):
    topology_directory = tools.get_topology_directory()
    file_name = 'paramterserver-{}nodes.npy'.format(num_nodes)
    file_path = os.path.join(topology_directory, file_name)

    return file_path


def generate_star_topology(num_nodes):
    ps_matrix = np.zeros(shape=(num_nodes + 1, num_nodes + 1), dtype=np.float32)

    ps_matrix[:, num_nodes] = 1.
    ps_matrix[num_nodes, :] = 1.
    ps_matrix[num_nodes, num_nodes] = 0.

    file_path = topology_file_path(num_nodes)
    tools.careful_file_path(file_path)

    np.save(file_path, ps_matrix)


def test():
    num_nodes = 10
    generate_star_topology(num_nodes)


if __name__ == '__main__':
    test()
