# -*- coding: utf-8 -*-

import torch
import time


class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name=str(type(self))  # model name

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        prefix = 'checkpoints/'
        if name is None:
            name = prefix + self.model_name + '_'
            name = time.strftime(name + '%m%d_%H:%M:%S.pth')
        else:
            name = prefix + self.model_name + '_' + str(name)+ '.pth'
        torch.save(self.state_dict(), name)
        return name
