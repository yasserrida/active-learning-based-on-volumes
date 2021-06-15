import numpy as np
import torch
from Helpers.strategie import Strategie


class VAL(Strategie):
    def __init__(self, x_train, y_train, idxs_labeled, net, handler, args):
        super(VAL, self).__init__(
            x_train, y_train, idxs_labeled, net, handler, args)

    # ======== Selectionner les point avec le score de confiance le plus elevé
    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_labeled]
        probs = self.predict_prob(
            self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        U = probs.max(1)[0]
        return idxs_unlabeled[U.sort()[1][:n]]
