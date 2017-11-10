#!/usr/bin/env python

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

    def forward(self, input):
        net = self.inner_net.forward(input)
        mean = net[:, 0]
        var = F.softplus(net[:, 1]) + torch.autograd.Variable(torch.ones(1)*1e-6)
        return torch.t(torch.stack((mean, var)))

    @staticmethod
    def nll(x, y):
        means = x[:, 0]
        vars = x[:, 1]
        a = torch.log(vars)/2
        b = 0.5/vars
        return torch.sum(a+b*torch.pow(y-means, 2))


def base_model(sizes):
    net = create_base_net(sizes)
    loss_fn = nn.MSELoss()
    return loss_fn, net


def proba_model(sizes):
    net = ProbabilisticNet(sizes)
    loss_fn = net.nll
    return loss_fn, net


def train_net(sizes, learning_rate, eps, alpha, max_iter, use_adv):
    loss_fn, net = proba_model(sizes)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    batch_size = 800
    losses = np.zeros(max_iter)
    for i in range(0, max_iter):
        optimizer.zero_grad()
        indices = np.random.choice(np.arange(len(xs)), size=batch_size)
        x = Variable(torch.from_numpy(xs[indices]+np.random.randn(batch_size)), requires_grad=True)
        t = Variable(torch.from_numpy(ts[indices]))
        pred = net(x)
        loss = alpha*loss_fn(pred, t)
        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward(retain_graph=True)
        if use_adv:
            adv_sign = torch.sign(x.grad)
            x_adv = x + eps * adv_sign
            x_adv.volatile = False
            pred_adv = net(x_adv)
            loss_adv = (1-alpha)*loss_fn(pred_adv, t)
            loss_adv.backward()
        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()
        joint_loss = loss.data[0] + loss_adv.data[0]
        losses[i] += joint_loss
        if i % 100 == 0:
            print('iteration {} with loss {}'.format(i, joint_loss))
    return net


def ensemble_mean_var(ensemble, x):
    en_mean = np.zeros(len(x))
    en_var = np.zeros(len(x))
    for model in ensemble:
        pred_ts = model(x)
        mean = pred_ts[:, 0].data.numpy()
        var = pred_ts[:, 1].data.numpy()
        en_mean += mean
        en_var += var + mean ** 2
    en_mean /= len(ensemble)
    en_var /= len(ensemble)
    en_var -= en_mean ** 2
    return en_mean, en_var


if __name__ == '__main__':
    xs = np.expand_dims(np.linspace(-5, 5, num=800, dtype=np.float32), -1)
    ts = np.cos(xs)

    learning_rate = 1e-3
    sizes = [1, 16, 16, 1]
    eps = 1e-2
    alpha=0.5
    max_iter = 800

    # number of nets in the ensemble
    K = 1

    trained_nets = [train_net(sizes, learning_rate, eps, alpha, max_iter, use_adv=True) for _ in range(0,K)]
    #plt.figure()
    #plt.plot(losses)
    if True:
        plt.figure()

        xs_np = np.expand_dims(np.linspace(-10, 10, num=100, dtype=np.float32), -1)
        xs = Variable(torch.from_numpy(xs_np))
        ts = np.cos(xs_np)

        pred_ts = ensemble_mean_var(trained_nets, xs)
        means, vars = pred_ts
        plt.errorbar(x=xs_np.reshape(-1,1), y=means.reshape(-1,1), yerr=vars.reshape(-1,1))
        plt.plot(xs_np.reshape(-1,1), ts.reshape(-1,1))
        plt.show()

