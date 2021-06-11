import numpy as np
from .strategy import Strategy


class MarginSampling(Strategy):
    def __init__(self, x_train, y_train, idxs_labeled, net, handler, args):
        super(MarginSampling, self).__init__(
            x_train, y_train, idxs_labeled, net, handler, args)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_labeled]
        probs = self.predict_prob(
            self.x_train[idxs_unlabeled], self.y_train[idxs_unlabeled])
        probs_sorted, idxs = probs.sort(descending=True)
        U = probs_sorted[:, 0] - probs_sorted[:, 1]
        return idxs_unlabeled[U.sort()[1][:n]]
