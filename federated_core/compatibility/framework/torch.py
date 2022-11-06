# -*- coding: utf-8 -*-
import torch


def mean(parameter):
    return torch.mean(torch.stack([w.cpu() for w in parameter], dim=0), dim=0)


def get_weights(self):
    return {k: v.cpu() for k, v in self.state_dict().items() if
            'weight' in k or 'bias' in k}


def set_weights(self, weights):
    self.load_state_dict(weights, strict=True)


def get_gradients(self):
    # fixme::做差得到正确的梯度
    grads = []
    for p in self.parameters():
        grad = None if p.grad is None else p.grad.data
        grads.append(grad.cpu())
    return grads


def apply_gradients(self, gradients):
    for g, p in zip(gradients, self.named_parameters()):
        if g is not None:
            p[1].grad = g.to(p[1].device)
    self.optimizer_func.step()
