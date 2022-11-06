# -*- coding: utf-8 -*-
import tensorflow as tf


def mean(parameter):
    return tf.reduce_mean(parameter, axis=0)


def get_weights(self):
    return {tensor.name: tensor.read_value() for tensor in self.trainable_variables}


def set_weights(self, weights):
    for tensor in self.trainable_variables:
        tensor.assign(weights[tensor.name])


def get_gradients(self):
    # fixme::做差得到正确的参数
    result = dict()
    # for g, t in zip(gradients, self.net_func.trainable_variables):
    #     result[t.name] = g
    return result


def set_gradients(self, gradients):
    self.optimizer_func.apply_gradients(zip(gradients, self.net_func.trainable_variables))


def apply_gradients(self):
    pass
