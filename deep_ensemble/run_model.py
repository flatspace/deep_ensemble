import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable


def load_model(path):
    return torch.load(path)


if __name__ =='__main__':
    model = load_model('./naive_trained_model.trch')
    xs = np.expand_dims(np.linspace(-5, 5, num=100, dtype=np.float32), -1)
    ts = np.cos(xs)
    pred_ts = model(Variable(torch.from_numpy(xs)).transpose(1,-1))
    plt.figure()
    plt.plot(xs, pred_ts.data.numpy())
    plt.plot(xs, ts)
    plt.show()
