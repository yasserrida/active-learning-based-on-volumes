import numpy as np
from Helpers.strategie import Strategie


class RandomSampling(Strategie):
    def __init__(self, x_train, y_train, idxs_labeled, net, handler, args):
        super(RandomSampling, self).__init__(
            x_train, y_train, idxs_labeled, net, handler, args)

    # ======== Selectionner des point aléatoirement
    def query(self, n):
        return np.random.choice(np.where(self.idxs_labeled == 0)[0], n)
