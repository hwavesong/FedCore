# -*- coding: utf-8 -*-
import numpy as np

from federated_core.environment.topology.type import gossip
from federated_core.environment.topology.type import parameter_server


class TopologyGenerator(object):
    def __init__(self):
        pass

    def build_all_reduce(self):
        raise NotImplementedError

    def load_all_reduce(self):
        raise NotImplementedError

    def build_gossip(self, num_topology, density, mutation_prob, num_nodes):
        try:
            topology_params = gossip.TopologyParams(num_nodes=num_nodes,
                                                    density=density,
                                                    mutation_prob=mutation_prob)
            gossip.generate_sequential_topology(topology_params, num_topology)
        except Exception as e:
            pass
            # logger.error('FAILED! THE REASON IS' + e)

    def load_gossip(self, num_topology, density, mutation_prob, num_nodes):
        topology_params = gossip.TopologyParams(num_nodes=num_nodes,
                                                density=density,
                                                mutation_prob=mutation_prob)
        file_path = gossip.topology_file_path(topology_params, num_topology)
        return np.load(file_path)

    def build_parameter_server(self, num_nodes):
        parameter_server.generate_star_topology(num_nodes)

    def load_parameter_server(self, num_nodes):
        file_path = parameter_server.topology_file_path(num_nodes)
        return np.load(file_path)


def test():
    np.random.seed(42)

    num_nodes = 10

    tg = TopologyGenerator()

    tg.build_parameter_server(num_nodes)
    matrix = tg.load_parameter_server(num_nodes)
    print(matrix)


if __name__ == '__main__':
    test()
