# -*- coding: utf-8 -*-
import types

from federated_core.misc.enums import Framework


def detect_framework(instance):
    str_type = str(instance.__class__.__base__)

    if 'tensorflow' in str_type:
        return Framework.tensorflow
    if 'torch' in str_type:
        return Framework.torch

    raise NotImplementedError


def mean(parameter):
    paramter_framework = detect_framework(parameter[0])

    if paramter_framework is Framework.torch:
        from federated_core.compatibility.framework.torch import mean
        return mean(parameter)

    if paramter_framework is Framework.tensorflow:
        from federated_core.compatibility.framework.tensorflow import mean
        return mean(parameter)


def inject_weight_gradient_func(net_func):
    if detect_framework(net_func) is Framework.torch:
        from federated_core.compatibility.framework.torch import get_weights, set_weights, get_gradients, apply_gradients

        net_func.get_weights = types.MethodType(get_weights, net_func)
        net_func.set_weights = types.MethodType(set_weights, net_func)

        net_func.get_gradients = types.MethodType(get_gradients, net_func)
        net_func.set_gradients = types.MethodType(apply_gradients, net_func)

    if detect_framework(net_func) is Framework.tensorflow:
        from federated_core.compatibility.framework.tensorflow import get_weights, set_weights, get_gradients, apply_gradients

        net_func.get_weights = types.MethodType(get_weights, net_func)
        net_func.set_weights = types.MethodType(set_weights, net_func)

        net_func.get_gradients = types.MethodType(get_gradients, net_func)
        net_func.set_gradients = types.MethodType(apply_gradients, net_func)

    return net_func
