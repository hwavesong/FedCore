# -*- coding: utf-8 -*-
import sys


def receive_many(comm_handler, num_receive):
    messages = comm_handler.receive(num_receive)
    # logging.debug('Receive messages from {} nodes.'.format(len(self.neighbor_ids)))

    return messages


def shutdown():
    sys.exit(0)


def initialize(net_func):
    content = dict()
    content['weight'] = net_func.get_weights()
    return content


def train(messages, train_step_func, train_dataset, train_dataset_iter, net_func, loss_func, optimizer_func, train_loss, train_metric):
    messages = messages[0]
    parameters = messages['content']['weight']
    net_func.set_weights(parameters)

    iteration = messages['content']['iteration']  # todo::增加epoch的指定

    train_loss.reset_state()
    train_metric.reset_state()

    for iteration_idx in range(iteration):
        try:
            x, y = next(train_dataset_iter)
        except StopIteration:
            train_dataset_iter = iter(train_dataset)
            x, y = next(train_dataset_iter)

        train_step_func(x, y, net_func, loss_func, optimizer_func, train_loss, train_metric)
        # iteration_idx += 1

        # if iteration_idx % 10 == 0:
        #     self.logger.debug('Iterations - {: >5}, loss: {:.3f}'.format(iteration_idx, self.train_loss))
    content = dict()
    content['weight'] = net_func.get_weights()
    content['train_loss'] = train_loss.result()
    content['train_metric'] = train_metric.result()

    return train_dataset_iter, content


def evaluate(test_dataset, test_step_func, net_func, loss_func, test_loss, test_metric):
    test_loss.reset_state()
    test_metric.reset_state()

    for x, y in test_dataset:
        test_step_func(x, y, net_func, loss_func, test_loss, test_metric, None)

    # logging.info('test_acc:{:.2f}'.format(self.test_metric))

    content = dict()
    content['test_loss'] = test_loss.result()
    content['test_metric'] = test_metric.result()

    return content


def sync_aggregate(messages, aggregation_func):
    # todo::check all the messages
    # print('=' * 10, aggregation_idx, '=' * 10)
    # aggregation_idx += 1

    parameters, losses, metrics = list(), list(), list()
    for message in messages:
        # assert message['type'] == Message.upload
        # print(message)

        parameters.append(message['content']['weight'])
        # losses.append(message['content']['loss'])
        # metrics.append(message['content']['metric'])

        # logging.info('Receive messages from {} / {} at {}.'.format(message['role'], message['sender'], message['send_time']))
        print('Receive messages from {} / {} at {}.'.format(message['role'], message['sender'], message['send_time']))

    avg_parameters = aggregation_func(parameters)

    content = dict()
    content['weight'] = avg_parameters
    # todo::实现server端的评估函数
    # self.optimizer_func.zero_grad()
    # self.net.set_gradients(aggred_gradients)
    # self.optimizer.step()

    # self.logger.debug('{} : loss is {}'.format(action_idx, losses))
    # self.logger.info('{} : loss is {}'.format(action_idx,', '.join(['{:.2f}'.format(l) for l in losses])))
    # logging.info(' : loss is {}'.format(', '.join(['{:.2f}'.format(l) for l in losses])))
    # print(' : loss is {}'.format(', '.join(['{:.2f}'.format(l) for l in losses])))

    # self.logger.debug('{} : metric is {}'.format(action_idx, metrics))
    # self.logger.info('{} : metric is {}'.format(action_idx,', '.join(['{:.2f}%'.format(m*100) for m in metrics])))

    # mean_loss = np.mean(losses)

    return content
