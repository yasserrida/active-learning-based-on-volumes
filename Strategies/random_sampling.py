import numpy as np
from .strategy import Strategy


class RandomSampling(Strategy):
    def __init__(self, x_train, y_train, idxs_labeled, net, handler, args):
        super(RandomSampling, self).__init__(
            x_train, y_train, idxs_labeled, net, handler, args)

    def query(self, n):
        return np.random.choice(np.where(self.idxs_labeled == 0)[0], n)
