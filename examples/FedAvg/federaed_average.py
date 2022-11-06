# -*- coding: utf-8 -*-
# @Time    : 14:54 2022/11/2 
# @Author  : Haohao Song
# @Email   : songhaohao2021@stu.xmu.edu.cn
# @File    : core.py
import tensorflow as tf

from federated_core.compatibility.algorithms import average_parameters
from federated_core.engine import parallelism
from federated_core.environment.dataset.spliter import NodeDatasetIndex
from federated_core.environment.setting.builder import HostBuilder
from federated_core.environment.topology.generator import TopologyGenerator
from federated_core.federation.edge import queuer
from federated_core.misc.enums import ExecutionPlanTemplate
from federated_core.misc.enums import Role

'''
1. 创建拓扑
2. 切分dataset
3. 创建host setting
并行执行引擎:
    4. 编写并发执行的run子函数
    5. 编写并发执行的主函数
'''


def generate_topology(num_nodes):
    tg = TopologyGenerator()
    tg.build_parameter_server(num_nodes)
    topology_matrix = tg.load_parameter_server(num_nodes)


def split_dataset(num_nodes):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # np.random.shuffle(y_train)

    ndi = NodeDatasetIndex()
    ndi.assign_num_nodes(num_nodes)
    ndi.assign_labels(y_train)
    ndi.enable_label_mean()
    # ndi.enable_label_skewness()
    ndi.enable_quantity_mean()
    # ndi.enable_quantity_normal()
    ndi.simulate_node_datasets()


def build_host(num_nodes):
    tg = TopologyGenerator()
    topology_matrix = tg.load_parameter_server(num_nodes)

    for host_id in range(num_nodes):
        hb = HostBuilder(host_id)
        hb.build_neighbors(topology_matrix)
        hb.build_role(Role.client)
        hb.save_setting()

    hb = HostBuilder(num_nodes)
    hb.build_neighbors(topology_matrix)
    hb.build_role(Role.server)
    hb.save_setting()


def run(num_nodes, has_server):
    def build_host_ids():
        if has_server:
            return [i for i in range(num_nodes + 1)]
        else:
            return [i for i in range(num_nodes)]

    def build_func_libs():
        func_libs = {
            'train_dataset'   : 'self_contained_dnn',  # load_train_dataset,
            'test_dataset'    : 'self_contained_dnn',  # load_test_dataset,
            'model'           : 'self_contained_dnn',  # get_model,
            'loss'            : 'self_contained_dnn',  # get_loss,
            'optimizer'       : 'self_contained_dnn',  # get_optimizer,
            'metric_loss'     : 'self_contained_dnn',  # get_metric_loss,
            'metric_acc'      : 'self_contained_dnn',  # get_metric_acc,
            'train_step'      : 'self_contained_dnn',  # get_train_step,
            'test_step'       : 'self_contained_dnn',  # get_test_step,
            'aggregation_func': average_parameters,
        }
        return func_libs

    def build_linkers():
        node_inboxes = queuer.node_inbox(num_nodes + 1)

        linkers = list()
        for host_id in range(num_nodes):
            linker = queuer.LocalQueue(host_id, node_inboxes)
            linkers.append(linker)

        if has_server:
            linker = queuer.LocalQueue(num_nodes, node_inboxes)
            linkers.append(linker)

        return linkers

    def build_execution_plans():
        execution_plans = ExecutionPlanTemplate.client_train * 5
        execution_plans = [[[c, {}] for c in execution_plans] for _ in range(num_nodes)]

        if has_server:
            server_execution_plan = ExecutionPlanTemplate.server_init + ExecutionPlanTemplate.server_sync_train * 5
            server_execution_plan.pop(-1)
            server_execution_plan = [[s, {'iteration': 3}] for s in server_execution_plan]

            execution_plans.append(server_execution_plan)

        return execution_plans

    host_ids = build_host_ids()
    func_libs = build_func_libs()
    linkers = build_linkers()
    execution_plans = build_execution_plans()

    parallelism.run_parallel(host_ids, func_libs, linkers, execution_plans)


def main():
    num_nodes = 2
    # generate_topology(num_nodes)
    # split_dataset(num_nodes)
    # build_host(num_nodes)
    run(num_nodes, True)


if __name__ == '__main__':
    main()
