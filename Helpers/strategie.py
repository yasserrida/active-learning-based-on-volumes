import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


class Strategie:
    def __init__(self, x_train, y_train, idxs_labeled, classifier, handler, args):
        self.x_train = x_train
        self.y_train = y_train
        self.idxs_labeled = idxs_labeled
        self.classifier = classifier
        self.handler = handler
        self.args = args
        self.n_pool = len(y_train)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def query(self, n):
        pass

    def update(self, idxs_labeled):
        self.idxs_labeled = idxs_labeled

    def _train(self, loader_tr, optimizer):
        self.clf.train()
        # ========== Boucler par le nombre de batch
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device)
            # ========== mettre les gradients à zéro
            optimizer.zero_grad()
            out, e1 = self.clf(x)
            # ========== Définir la fonction de perte
            loss = F.cross_entropy(out, y)
            # ========== accumule les gradients
            loss.backward()
            # ========== metre à jour les paramètres
            optimizer.step()

    def train(self):
        n_epoch = self.args['n_epoch']
        self.clf = self.classifier().to(self.device)
        # ========== Stochastic Gradient Descent Optimizer
        optimizer = optim.SGD(self.clf.parameters(), **
                              self.args['optimizer_args'])
        # ========== Index de train
        idxs_train = np.arange(self.n_pool)[self.idxs_labeled]
        # ========== paralléliser le processus de chargement des données avec le batching automatique
        loader_tr = DataLoader(self.handler(self.x_train[idxs_train], self.y_train[idxs_train],
                               transform=self.args['transform']), shuffle=True, ** self.args['loader_tr_args'])
        # ========== Boucler par le nombre d'epoche
        for epoch in range(1, n_epoch+1):
            self._train(loader_tr, optimizer)

    def predict(self, x_train, y_train):
        # ========== paralléliser le processus de chargement des données avec le batching automatique
        loader_te = DataLoader(self.handler(
            x_train, y_train, transform=self.args['transform']), shuffle=False, ** self.args['loader_te_args'])
        # ========== Met le module en mode évaluation.
        self.clf.eval()
        # ========== Initialiser avec des zéros
        P = torch.zeros(len(y_train), dtype=y_train.dtype)
        # ========== Gestionnaire de contexte qui a désactivé le calcul du gradient
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                # ====== la valeur maximale de tous les éléments
                pred = out.max(1)[1]
                P[idxs] = pred.cpu()
        return P

    def predict_prob(self, x_train, y_train):
        # ========== paralléliser le processus de chargement des données avec le batching automatique
        loader_te = DataLoader(self.handler(
            x_train, y_train, transform=self.args['transform']), shuffle=False, ** self.args['loader_te_args'])
        # ========== Met le module en mode évaluation.
        self.clf.eval()
        # ========== Initialiser avec des zéros
        probs = torch.zeros([len(y_train), len(np.unique(y_train))])
        # ========== Gestionnaire de contexte qui a désactivé le calcul du gradient
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                # ====== Applique la fonction Softmax
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        return probs

    def get_embedding(self, X, Y):
        # ========== paralléliser le processus de chargement des données avec le batching automatique
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                               shuffle=False, **self.args['loader_te_args'])
        # ========== Met le module en mode évaluation.
        self.clf.eval()
        # ========== Initialiser avec des zéros
        embedding = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        # ========== Gestionnaire de contexte qui a désactivé le calcul du gradient.
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                # ======== encastrement
                out, e1 = self.clf(x)
                embedding[idxs] = e1.cpu()
        return embedding
