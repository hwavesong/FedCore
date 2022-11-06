# -*- coding: utf-8 -*-
import json
import os

from federated_core.misc import tools


class HostBuilder(object):
    def __init__(self, host_id):
        self.host_id = host_id
        self.content = {'host_id': self.host_id}

    def build_neighbors(self, topology_matrix):
        self.content['neighbors'] = [idx for idx, item in enumerate(topology_matrix[self.host_id]) if item > 0]

    def build_role(self, role):
        self.content['role'] = role.name

    def build_self_attribute(self, key, val):  # used for training.iteration, and etc.
        self.content[key] = val

    def get_setting_path(self):
        host_id_directory = tools.get_host_id_directory(self.host_id)
        setting_path = os.path.join(host_id_directory, 'setting.json')
        return setting_path

    def check_name_in_content(self, name):
        if name not in self.content.keys():
            raise ValueError('Not set {}.'.format(name))

    def save_setting(self):
        self.check_name_in_content('neighbors')
        self.check_name_in_content('role')

        setting_path = self.get_setting_path()
        tools.careful_file_path(setting_path)
        with open(setting_path, 'w', encoding='utf8') as fw:
            json.dump(self.content, fw)

    def load_setting(self):
        setting_path = self.get_setting_path()
        with open(setting_path, 'r', encoding='utf8') as fr:
            content = json.load(fr)

        return content
