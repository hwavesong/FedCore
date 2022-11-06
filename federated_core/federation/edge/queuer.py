# -*- coding: utf-8 -*-
from multiprocessing import Queue


def node_inbox(num_nodes):
    inboxes = [Queue() for _ in range(num_nodes)]
    return inboxes


class LocalQueue(object):
    def __init__(self, host_id, inbox):
        self.host_id = host_id
        self.inbox = inbox

    def send(self, message, destinations):
        for neighbor in destinations:
            self.inbox[neighbor].put(message)

    def receive(self, num_messages):
        recevied = [self.inbox[self.host_id].get() for _ in range(num_messages)]
        return recevied
