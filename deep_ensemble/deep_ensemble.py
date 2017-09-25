#!/usr/bin/env python
from __future__ import print_function
from itertools import count

import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import itertools


def create_net(sizes):
    dims = list(zip(sizes, sizes[1:]))
    layers = list(itertools.chain.from_iterable(((nn.Linear(in_dim, out_dim), nn.ReLU())
                                                 for in_dim, out_dim in dims[:-1])))
    last_layer = nn.Linear(*dims[-1])
    layers.append(last_layer)
    net = nn.Sequential(*layers)
    mean = net[0]
    var = nn.Softplus(net[1]) + autograd.Variable(torch.FloatTensor(1e-6))
    return net


if __name__ is "__main__":
    sizes = [1,20,20,2]
    learning_rate = 1e-4
    net = create_net(sizes)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    loss_fn = nn.KLDivLoss()

