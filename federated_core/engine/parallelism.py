# -*- coding: utf-8 -*-
import multiprocessing

from federated_core.federation.vertex.host import Host
from federated_core.misc import tools


def role_fn(host_id, func_libs, linker, execution_plan):
    host = Host(host_id, func_libs, linker)
    host.act(execution_plan)


def run_parallel(host_ids, func_libs, linkers, execution_plans):
    participants = list()

    for host_id, linker, execution_plan, in zip(host_ids, linkers, execution_plans):
        participant = multiprocessing.Process(target=role_fn, args=(host_id, func_libs, linker, execution_plan))
        participant.start()
        participants.append(participant)

    tools.pin_process_to_core(participants)

    for participant in participants:
        participant.join()
