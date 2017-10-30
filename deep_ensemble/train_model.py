#!/usr/bin/env python
from itertools import count

import logging
import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import itertools
import numpy as np
import matplotlib.pyplot as plt


def create_base_net(sizes):
    dims = list(zip(sizes, sizes[1:]))
    layers = list(itertools.chain.from_iterable(((nn.Linear(in_dim, out_dim), nn.ReLU())
                                                 for in_dim, out_dim in dims[:-1])))
    reduction_layer = nn.Linear(*dims[-1])
    layers.append(reduction_layer)
    net = nn.Sequential(*layers)
    return net


class ProbabilisticNet(nn.Module):
    def __init__(self, sizes):
        super(ProbabilisticNet, self).__init__()
        self.inner_net = create_base_net(sizes+[2])

    def forward(self, *input):
        net = self.inner_net.forward(*input)
        mean = net[0]
        var = nn.Softplus(net[1]) + torch.autograd.Variable(torch.FloatTensor(1e-6))
        return mean, var

    def nll(self, x, y):
        mean, var = self(x)
        a = torch.log(var)/2
        b = 0.5/var

        def loss_per_point(x1, y1):
            return a - b*(y-mean)**2

        return torch.sum([loss_per_point(x1,y1) for x1, y1 in zip(x,y)])


if __name__ == '__main__':
    sizes = [1,20,20,1]
    learning_rate = 1e-4
    max_iter = 10000
    batchsize=30
    net = create_base_net(sizes)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    xs = np.expand_dims(np.linspace(-5, 5, num=100, dtype=np.float32), -1)
    ts = np.cos(xs)
    losses = np.zeros(max_iter)
    for i in range(0,max_iter):
        indices = np.random.choice(np.arange(len(xs)), size=batchsize)
        x = Variable(torch.from_numpy(xs[indices]))
        t = Variable(torch.from_numpy(ts[indices]))
        t_pred = net(x)
        loss = loss_fn(t_pred, t)
        losses[i] = loss.data[0]
        optimizer.zero_grad()
        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()
        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()
        print('iteration {} with loss {}'.format(i,loss.data[0]))
        torch.save(net, './naive_trained_model.trch')
    plt.figure()
    plt.plot(losses)

    plt.figure()
    xs = np.expand_dims(np.linspace(-10, 10, num=100, dtype=np.float32), -1)
    ts = np.cos(xs)
    pred_ts = net(Variable(torch.from_numpy(xs)))
    plt.plot(xs, pred_ts.data.numpy())
    plt.plot(xs, ts)
    plt.show()

