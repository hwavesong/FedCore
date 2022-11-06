

FederatedCore
============

FederatedCore: a research-oriented federal learning framework.

Features
=============
#. Compatibility. FederatedCore can work seamlessly with mainstream deep learning frameworks, e.g., PyTorch and Tensorflow.
#. Modular. The code of the algorithm module can be used individually.
#. Easy to use. Retrofit existing code to data parallelism with no more than 100 lines code.


QuickStart
=============

Install m3u8_To_MP4 via pip
---------------------------------------

.. code-block:: python

   # via pypi.org
   python -m pip install federatedcore

   # first clone project, and install.
   git clone https://github.com/songs18/FederatedCore.git
   python -m pip install ./FederatedCore


Implement FedAvg in fewer than 100 lines.
---------------------------------------
/examples/FedAvg/federated_average.py


.. code-block:: python

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
        generate_topology(num_nodes)
        split_dataset(num_nodes)
        build_host(num_nodes)
        run(num_nodes, True)


    if __name__ == '__main__':
        main()




