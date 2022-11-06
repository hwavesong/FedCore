# -*- coding: utf-8 -*-
import math
import time

import numpy as np


def mean_quantity(**kwargs):
    record_index = kwargs['record_index']
    num_nodes = kwargs['num_nodes']
    mean = kwargs['_mean'] if '_mean' in kwargs else None

    num_records = len(record_index)

    if mean is None:  # distribute in a mean manner.
        mean = math.floor(num_records / num_nodes)

    # when the provided records are not sufficient.
    expected_num_record = mean * num_nodes
    if expected_num_record > num_records:
        raise ValueError('Input mean is too large. Please use a smaller value.')

    # assign record index for clients.
    record_index_for_clients = np.array_split(record_index, indices_or_sections=num_nodes)

    return record_index_for_clients


def normalized_distribution_size(deviation, mean, num_nodes):
    raunchy_distribution = np.random.normal(loc=mean, scale=deviation, size=num_nodes)
    # 2 * sigam
    filtered_2_sigma_distribution = raunchy_distribution[np.abs(raunchy_distribution) - mean < 2 * deviation]

    return filtered_2_sigma_distribution


def normal_quantity(**kwargs):
    record_index = kwargs['record_index']
    num_nodes = kwargs['num_nodes']
    mean = kwargs['_mean'] if '_mean' in kwargs else None
    deviation = kwargs['_deviation'] if '_deviation' in kwargs else 100

    num_records = len(record_index)

    if mean is None:  # distribute in a mean manner.
        mean = math.ceil(num_records / num_nodes)

    wall_time = time.time()
    cumsumed_distribution_size = None
    while True:
        distribution_size = normalized_distribution_size(deviation, mean,
                                                         num_nodes)
        distribution_size = distribution_size.astype(np.int32)
        # info=                    'The mean of generated dataset is {:.2f}, and the deviation is {:.2f}.'.format(
        #         np.mean(distribution_size), np.std(distribution_size))
        # logger.info(info)

        cumsumed_distribution_size = np.cumsum(distribution_size)

        if cumsumed_distribution_size[-1] < num_records:
            break

        if time.time() - wall_time > 60:
            raise ValueError('Bad luck.')

    distribution_size_index = np.insert(cumsumed_distribution_size, 0, [0])

    # assign record index for clients.
    nodes_record_index = list()
    for idx, start in enumerate(distribution_size_index[:-2]):
        end = distribution_size_index[idx + 1]
        nodes_record_index.append(record_index[start:end])

    return nodes_record_index
