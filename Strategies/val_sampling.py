import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


class VAL:
    def __init__(self, x_train, y_train, idxs_labeled, classifier, handler, args):
        self.x_train = x_train
        self.y_train = y_train
        self.idxs_labeled = idxs_labeled
        self.classifier = classifier
        self.handler = handler
        self.args = args
        self.n_pool = len(y_train)
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def query(self, n):
        pass

    def update(self, idxs_labeled):
        self.idxs_labeled = idxs_labeled

    def _train(self, epoch, loader_tr, optimizer):
        self.clf.train()
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            out, e1 = self.clf(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

    def train(self):
        n_epoch = self.args['n_epoch']
        self.clf = self.classifier().to(self.device)
        optimizer = optim.SGD(self.clf.parameters(), **
                              self.args['optimizer_args'])
        idxs_train = np.arange(self.n_pool)[self.idxs_labeled]
        loader_tr = DataLoader(self.handler(self.x_train[idxs_train], self.y_train[idxs_train],
                               transform=self.args['transform']), shuffle=True, **self.args['loader_tr_args'])
        for epoch in range(1, n_epoch+1):
            self._train(epoch, loader_tr, optimizer)

    def predict(self, x_train, y_train):
        loader_te = DataLoader(self.handler(
            x_train, y_train, transform=self.args['transform']), shuffle=False, **self.args['loader_te_args'])
        self.clf.eval()
        P = torch.zeros(len(y_train), dtype=y_train.dtype)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                pred = out.max(1)[1]
                P[idxs] = pred.cpu()

        return P

    def predict_prob(self, x_train, y_train):
        loader_te = DataLoader(self.handler(
            x_train, y_train, transform=self.args['transform']), shuffle=False, **self.args['loader_te_args'])
        self.clf.eval()
        probs = torch.zeros([len(y_train), len(np.unique(y_train))])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        return probs
