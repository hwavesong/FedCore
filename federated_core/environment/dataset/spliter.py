# -*- coding: utf-8 -*-
import os

import numpy as np

from federated_core.environment.dataset.policy import label
from federated_core.environment.dataset.policy import quantity
from federated_core.misc import tools


def label_distribution_is_null(label_distribution):
    if label_distribution:
        raise AttributeError


def quantity_distribution_is_null(quantity_distribution):
    if quantity_distribution:
        raise AttributeError


def label_distribution_is_full(label_distribution):
    if not label_distribution:
        raise AttributeError


def quantity_distribution_is_full(quantity_distribution):
    if not quantity_distribution:
        raise AttributeError


def get_index_path(host_id):
    host_id_directory = tools.get_host_id_directory(host_id)
    index_path = os.path.join(host_id_directory, 'train_index.npy')
    return index_path


def dump_index(node_record_index):
    # logger.info('Dump train datasets...')

    for host_id, record_index in enumerate(node_record_index):
        index_path = get_index_path(host_id)
        tools.careful_file_path(index_path)
        np.save(index_path, record_index)
        # logger.info('Dump {}.'.format(dataset_path))


class NodeDatasetIndex():
    def __init__(self):
        self.label_distribution = None
        self.quantity_distribution = None

    def assign_labels(self, labels):
        self.labels = labels

    def assign_num_nodes(self, num_nodes):
        self.num_nodes = num_nodes

    def enable_label_mean(self):
        label_distribution_is_null(self.label_distribution)
        self.label_distribution = 'MEAN'

    def enable_label_skewness(self):  # warning::return number
        label_distribution_is_null(self.label_distribution)
        self.label_distribution = 'SKEW'

    def enable_quantity_mean(self):
        quantity_distribution_is_null(self.quantity_distribution)
        self.quantity_distribution = 'MEAN'

    def enable_quantity_normal(self):  # warning::return number
        quantity_distribution_is_null(self.quantity_distribution)
        self.quantity_distribution = 'NORM'

    def simulate_label(self, **kwargs):
        if self.label_distribution == 'MEAN':
            return label.iid_distribution(**kwargs)
        elif self.label_distribution == 'SKEW':
            return label.non_iid_distribution(**kwargs)
        else:
            raise NotImplementedError

    def simulate_quantity(self, **kwargs):
        if self.quantity_distribution == 'MEAN':
            return quantity.mean_quantity(**kwargs)
        elif self.quantity_distribution == 'NORM':
            return quantity.normal_quantity(**kwargs)
        else:
            raise NotImplementedError

    def simulate_node_datasets(self, _mean=None, _deviation=100):
        label_distribution_is_full(self.label_distribution)
        quantity_distribution_is_full(self.quantity_distribution)
        # logger.info('Simulate training datasets distirbution.')

        record_index = np.arange(0, len(self.labels))  # todo::shuffle
        record_index = self.simulate_label(record_index=record_index, ys=self.labels)
        node_record_index = self.simulate_quantity(record_index=record_index, num_nodes=self.num_nodes, _mean=_mean, _deviation=_deviation)

        dump_index(node_record_index)

    def load_host_train_index(self, host_id):
        index_path = get_index_path(host_id)
        np_index = np.load(index_path)
        return np_index


def test():
    record_index = np.arange(10).repeat(repeats=5, axis=0)
    np.random.shuffle(record_index)

    ndi = NodeDatasetIndex()
    ndi.assign_labels(record_index)
    ndi.assign_num_nodes(5)
    # ndi.enable_label_mean()
    ndi.enable_label_skewness()
    # ndi.enable_quantity_mean()
    ndi.enable_quantity_normal()
    ndi.simulate_node_datasets()
    host_0_index = ndi.load_host_train_index(0)
    print(host_0_index)


if __name__ == '__main__':
    test()
