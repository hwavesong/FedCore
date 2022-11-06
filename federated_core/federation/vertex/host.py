# -*- coding: utf-8 -*-
import importlib
import time

from federated_core.compatibility.framework import one_framework
from federated_core.environment.dataset.spliter import NodeDatasetIndex
from federated_core.environment.setting.builder import HostBuilder
from federated_core.federation.vertex import primitives
from federated_core.misc import tools
from federated_core.misc.enums import Command


class Host(object):
    def __init__(self, host_id, func_libs, linker):
        self.logger = tools.get_logger()

        self.host_id = host_id
        tools.enable_write_to_file(self.logger, self.host_id)

        hb = HostBuilder(host_id)
        content = hb.load_setting()
        self.logger.info(content)
        self.neighbor_ids = content['neighbors']
        self.role = content['role']

        test_dataset_module = importlib.import_module(func_libs['test_dataset'])
        load_test_dataset_func = getattr(test_dataset_module, 'load_test_dataset')
        self.test_dataset = load_test_dataset_func()

        model_module = importlib.import_module(func_libs['model'])
        model_func = getattr(model_module, 'get_model')
        self.net_func = model_func(self.test_dataset)

        loss_func_module = importlib.import_module(func_libs['loss'])
        get_loss_func = getattr(loss_func_module, 'get_loss')
        self.loss_func = get_loss_func()

        metric_loss_module = importlib.import_module(func_libs['metric_loss'])
        self.get_metric_loss = getattr(metric_loss_module, 'get_metric_loss')
        self.metric_loss = None

        metric_acc_module = importlib.import_module(func_libs['metric_acc'])
        self.get_metric_acc = getattr(metric_acc_module, 'get_metric_acc')
        self.metric_acc = None

        test_step_module = importlib.import_module(func_libs['test_step'])
        get_test_step_func = getattr(test_step_module, 'get_test_step')
        self.test_step_func = get_test_step_func()

        self.aggregation_func = func_libs['aggregation_func']

        self.linker = linker

        self.tailor_attributes(func_libs)

    def tailor_attributes(self, func_libs):
        if self.role != 'server':
            ndi = NodeDatasetIndex()
            train_index = ndi.load_host_train_index(self.host_id)
            train_dataset_module = importlib.import_module(func_libs['train_dataset'])
            load_train_dataset_func = getattr(train_dataset_module, 'load_train_dataset')
            self.train_dataset = load_train_dataset_func(train_index)
            self.train_dataset_iter = iter(self.train_dataset)

            optimizer_module = importlib.import_module(func_libs['optimizer'])
            get_optimizer_func = getattr(optimizer_module, 'get_optimizer')
            self.optimizer_func = get_optimizer_func()

            train_step_module = importlib.import_module(func_libs['train_step'])
            get_train_step_func = getattr(train_step_module, 'get_train_step')
            self.train_step_func = get_train_step_func()

        # todo::扩展灵活性
        self.net_func = one_framework.inject_weight_gradient_func(self.net_func)

    def act(self, execution_plan=None):
        self.logger.info('execution plan length is {}.'.format(len(execution_plan)))
        self.logger.info('execution plan is {}.'.format(execution_plan))

        messages = None
        content = {}
        for command_idx, command in enumerate(execution_plan):
            self.logger.info('{} -- {} -- {}'.format(self.host_id, command_idx, command))

            command_type = command[0]
            command_content = command[1]

            if command_type is Command.receive_many:
                messages = primitives.receive_many(self.linker, len(self.neighbor_ids))

            if command_type is Command.shutdown:
                primitives.shutdown()

            if command_type is Command.train:
                self.metric_loss = self.get_metric_loss()
                self.metric_acc = self.get_metric_acc()
                self.train_dataset_iter, train_result = primitives.train(messages,
                                                                         self.train_step_func, self.train_dataset, self.train_dataset_iter,
                                                                         self.net_func, self.loss_func, self.optimizer_func,
                                                                         self.metric_loss, self.metric_acc)
                content.update(train_result)

            if command_type is Command.eval:
                self.metric_loss = self.get_metric_loss()
                self.metric_acc = self.get_metric_acc()
                eval_result = primitives.evaluate(self.test_dataset, self.test_step_func, self.net_func, self.loss_func, self.metric_loss, self.metric_acc)
                content.update(eval_result)

            if command_type is Command.initialize:
                init_result = primitives.initialize(self.net_func)
                content.update(init_result)

            if command_type is Command.sync_aggregate:
                aggregate_result = primitives.sync_aggregate(messages, self.aggregation_func)
                content.update(aggregate_result)

            if command_type is Command.send_many:
                if self.role == 'server':
                    content.update(command_content)

                body = {'sender'   : self.host_id,
                        'role'     : self.role,
                        'type'     : 'unused',
                        'send_time': time.time(),
                        'content'  : content}
                self.linker.send(body, self.neighbor_ids)
                self.logger.info('send to {}'.format(self.neighbor_ids))

                messages = None
                content = {}
