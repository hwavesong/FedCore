# -*- coding: utf-8 -*-
from enum import Enum


class Framework(Enum):
    tensorflow = 'tensorflow'
    torch = 'torch'


class Command(Enum):
    shutdown = 0

    prepare = 1
    train = 2
    eval = 3

    initialize = 4

    sync_aggregate = 5
    async_aggregate = 6  # todo::implement

    receive_one = 7
    receive_many = 8
    receive_all = 9

    send_one = 10
    send_many = 11
    send_all = 12


class ExecutionPlanTemplate():  # sync ps tempalte
    server_init = [Command.initialize, Command.send_many]
    server_sync_train = [Command.receive_many, Command.sync_aggregate, Command.send_many]
    client_train = [Command.receive_many, Command.train, Command.eval, Command.send_many]


class Message(Enum):
    upload = 'upload'
    download = 'download'


class Role(Enum):
    client = 0
    server = 1
