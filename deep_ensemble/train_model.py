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
        inner_prediction = self.inner_net.forward(input)
        mean = inner_prediction[:, 0]
        var = F.softplus(inner_prediction[:, 1]) + torch.autograd.Variable(torch.ones(len(mean))*1e-6)
        return mean, var

    @staticmethod
    def nll(means, vars, y):
        a = torch.log(vars)
        b = 1.0/vars
        return torch.sum(a+b*torch.pow(y.squeeze()-means, 2))


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
    batch_size = 16
    losses = np.zeros(max_iter)
    for i in range(0, max_iter):
        optimizer.zero_grad()
        indices = np.random.choice(np.arange(len(xs)), replace=False, size=batch_size)
        x = Variable(torch.from_numpy(xs[indices]), requires_grad=True)
        t = Variable(torch.from_numpy(ts[indices])+0.2*torch.rand(len(indices),1))
        loss = loss_fn(*net(x), t)
        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        adv_sign = torch.sign(x.grad)
        x_adv = x + eps * adv_sign
        x_adv.volatile = False

        pred_adv = net(x_adv)
        loss_adv = loss_fn(*pred_adv, t)
        loss_adv.backward()

        #optimizer.zero_grad()
        #x_joint = torch.cat((x, x_adv))
        #pred_joint = net(x_joint)
        #loss_joint = loss_fn(*pred_joint, torch.cat((t,t)))
        #loss_joint.backward()
        #losses[i] += loss_joint.data.numpy()[0]
        optimizer.step()
        if i % 100 == 0:
            print('iteration {} with loss {}'.format(i, loss.data.numpy()[0]))
    return net


def ensemble_mean_var(ensemble, x):
    en_mean = np.zeros(len(x))
    en_var = np.zeros(len(x))
    for model in ensemble:
        mean, var = model(x)
        mean = mean.data.numpy()
        var = var.data.numpy()
        en_mean += mean
        en_var += var + mean**2
    en_mean /= len(ensemble)
    en_var /= len(ensemble)
    en_var -= en_mean**2
    return en_mean, en_var


if __name__ == '__main__':
    xs = np.expand_dims(np.linspace(-5, 5, num=100, dtype=np.float32), -1)
    ts = np.cos(xs)

    learning_rate = 1e-3
    sizes = [1, 16, 16]
    eps = 1e-3
    alpha=0.5
    max_iter = 1000

    # number of nets in the ensemble
    K = 5

    trained_nets = [train_net(sizes, learning_rate, eps, alpha, max_iter, use_adv=True) for _ in range(0,K)]
    #plt.figure()
    #plt.plot(losses)
    if True:
        plt.figure()

        xs_np = np.expand_dims(np.linspace(-10, 10, num=200, dtype=np.float32), -1)
        xs = Variable(torch.from_numpy(xs_np))
        ts = np.cos(xs_np)

        pred_ts = ensemble_mean_var(trained_nets, xs)
        means, vars = pred_ts
        plt.errorbar(x=xs_np.reshape(-1,1), y=means.reshape(-1,1), yerr=3*np.sqrt(vars.reshape(-1,1)))
        plt.plot(xs_np.reshape(-1,1), ts.reshape(-1,1))
        plt.show()

