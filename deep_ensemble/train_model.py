#!/usr/bin/env python

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

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
        var = F.softplus(inner_prediction[:, 1]) + torch.ones(len(mean), requires_grad=True)*1e-6
        return mean, var

    @staticmethod
    def nll(means, vars, y):
        a = torch.log(vars)
        b = 1.0/vars
        return torch.sum(a+b*torch.pow(y.squeeze()-means, 2))


def base_model(sizes: int):
    net = create_base_net(sizes)
    loss_fn = nn.MSELoss()
    return loss_fn, net


def probabilistic_model(sizes: int):
    net = ProbabilisticNet(sizes)
    loss_fn = net.nll
    return loss_fn, net


def train_net(x_train, t_train, sizes, learning_rate, eps, alpha, max_iter, use_adv):
    loss_fn, net = probabilistic_model(sizes)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    batch_size = 16
    for i in range(0, max_iter):
        optimizer.zero_grad()
        indices = np.random.choice(np.arange(len(x_train)), replace=False, size=batch_size)
        x = torch.tensor(x_train[indices], requires_grad=True)
        t = torch.tensor(t_train[indices])+0.2*torch.rand(len(indices),1)
        loss = loss_fn(*net(x), t)
        loss.backward()

        if use_adv:
            with torch.no_grad():
                adv_sign = torch.sign(x.grad)
                x_adv = x + eps * adv_sign

            loss_adv = loss_fn(*net(x_adv), t)
            loss_adv.backward()

        optimizer.step()
        if i % 100 == 0:
            print(f'iteration {i} with loss {loss.detach().numpy()}')
    return net


def ensemble_mean_var(ensemble, x):
    en_mean = np.zeros(len(x))
    en_var = np.zeros(len(x))
    for model in ensemble:
        with torch.no_grad():
            mean, var = model(x)
        mean = mean.numpy()
        var = var.numpy()
        en_mean += mean
        en_var += var + mean**2
    en_mean /= len(ensemble)
    en_var /= len(ensemble)
    en_var -= en_mean**2
    return en_mean, en_var


if __name__ == '__main__':

    learning_rate = 1e-3
    sizes = [1, 16, 16]
    eps = 1e-3
    alpha = 0.5
    max_iter = 1000

    # number of nets in the ensemble
    K = 5

    x_train = np.expand_dims(np.linspace(-10, 10, num=400, dtype=np.float32), -1)
    t_train = np.cos(x_train)
    trained_nets = [train_net(x_train, t_train, sizes, learning_rate, eps, alpha, max_iter, use_adv=True) for _ in range(0, K)]

    if True:
        plt.figure()

        xs_np = np.expand_dims(np.linspace(-20, 20, num=200, dtype=np.float32), -1)
        xs = torch.tensor(xs_np)
        ts = np.cos(xs_np)

        pred_ts = ensemble_mean_var(trained_nets, xs)
        means, vars = pred_ts
        plt.errorbar(x=xs_np.reshape(-1,1), y=means.reshape(-1,1), yerr=3*np.sqrt(vars.reshape(-1,1)))
        plt.plot(xs_np.reshape(-1,1), ts.reshape(-1,1))
        plt.show()
