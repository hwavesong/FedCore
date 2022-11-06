# -*- coding: utf-8 -*-
import logging
import multiprocessing
import os
import sys

from federated_core.misc.globalvar import FederatedVar

LOGGING_FORMAT_STR = '%(asctime)s : %(levelname)s : %(module)s : %(message)s'


def pin_process_to_core(processes):
    if sys.platform == 'linux':
        num_cores = multiprocessing.cpu_count()
        os.sched_setaffinity(0, [len(processes) % num_cores])
        for idx, participant in enumerate(processes):
            run_on_core = idx % num_cores
            os.sched_setaffinity(participant.pid, [run_on_core])


def cleanup_directoy():
    fv = FederatedVar()
    if os.path.exists(fv.root_directory):
        os.removedirs(fv.root_directory)


def get_topology_directory():
    fv = FederatedVar()
    topology_directory = os.path.join(fv.root_directory, 'topology')
    return topology_directory


def get_host_id_directory(host_id):
    fv = FederatedVar()
    host_directory = os.path.join(fv.root_directory, 'hosts')
    host_id_directory = os.path.join(host_directory, 'host_{}'.format(host_id))
    return host_id_directory


def careful_file_path(file_path):
    parent_directory = os.path.dirname(file_path)
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory)

    if os.path.exists(file_path):
        os.remove(file_path)


def get_logger():
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    return logger


def enable_write_to_console(logger):
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt=LOGGING_FORMAT_STR)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)


def enable_write_to_file(logger, host_id):
    host_id_directory = get_host_id_directory(host_id)
    log_path = os.path.join(host_id_directory, 'run.log')

    file_handler = logging.FileHandler(log_path, mode='w')
    formatter = logging.Formatter(fmt=LOGGING_FORMAT_STR)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
