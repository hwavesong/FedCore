# -*- coding: utf-8 -*-
import threading


class FederatedVar(object):
    _instance_lock = threading.Lock()

    def __init__(self):
        self.root_directory = './federated_io/'

    def __new__(cls, *args, **kwargs):
        if not hasattr(FederatedVar, "_instance"):
            with FederatedVar._instance_lock:
                if not hasattr(FederatedVar, "_instance"):
                    FederatedVar._instance = object.__new__(cls)
        return FederatedVar._instance
