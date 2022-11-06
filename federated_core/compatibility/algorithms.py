# -*- coding: utf-8 -*-
from federated_core.compatibility.framework import one_framework


def average_parameters(parameters):
    result = dict()
    for name in parameters[0].keys():
        result[name] = list()
        for parameter in parameters:
            result[name].append(parameter[name])
        result[name] = one_framework.mean(result[name])
    return result
