# -*- coding: utf-8 -*-
import numpy as np


def iid_distribution(**kwargs):
    record_index = kwargs['record_index']
    np.random.shuffle(record_index)

    return record_index


def non_iid_distribution(**kwargs):
    ys = kwargs['ys']
    record_index = kwargs['record_index']

    sort_index = np.argsort(ys, axis=0)
    record_index = record_index[sort_index]

    return record_index
