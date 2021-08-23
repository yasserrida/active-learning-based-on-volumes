from Helpers.dataset import get_dataset, get_handler
from Helpers.model import get_classifier
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET = 'MNIST'
NB_QUERY = 500
args = {'n_epoch': 10, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        'loader_tr_args': {'batch_size': 64, 'num_workers': 1},
        'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
        'optimizer_args': {'lr': 0.01, 'momentum': 0.5}}
NB_INITIAL_ETIQUITE = 1000
NB_QUERY = 500


def get_dataset_local():
    x_train, y_train = get_dataset(name=DATASET)
    x_train = x_train[:40000]
    y_train = y_train[:40000]
    return x_train, y_train


def label_initial_data(n_pool):
    idxs_labeled = np.zeros(n_pool, dtype=bool)
    index_temp = np.arange(n_pool)
    np.random.shuffle(index_temp)
    idxs_labeled[index_temp[:NB_INITIAL_ETIQUITE]] = True
    return idxs_labeled


def get_kernel_matrix(x_train, y_train, handler):
    loader_te = DataLoader(handler(
        x_train, y_train, transform=args['transform']), shuffle=False, ** args['loader_te_args'])
    return loader_te


if __name__ == '__main__':
    # -------------------- Init
    x_train, y_train = get_dataset_local()
    idxs_labeled = label_initial_data(len(y_train))
    handler = get_handler(DATASET)
    classifier = get_classifier(DATASET)
    classifier.eval()

    # --------------------- K Kernel Matrix of dataset
    k = get_kernel_matrix(x_train, y_train, handler)

    # --------------------- X* Sparse matrix
    x_etoil = torch.zeros(len(y_train), dtype=y_train.dtype)
    t = 0

    with torch.no_grad():
        while(t < len(y_train)/2):
            for x, y, idxs in k:
                x, y = x.to(device), y.to(device)
                # -------- Calculate the conﬁdence score of x_i
                out, e1 = classifier(x)
                conf = out.max(1)[1]

                # -------- add data point with the hieghest score to x_etoile
                x_etoil[idxs] = conf.cpu()

                # -------- Update t
                t = t + 1

    classifier.train()
    # -------- initilise U by passive sampling in x_etoile
    U = []
    j = 0
    while(j):
        # --------  Divide the local space to K parts by the model
        B = []

        # --------  Update U_prime
        U_prime = []

        # --------  Calculate the loss function of U and U_prime
        out, e1 = classifier(x_train[j])
        L = F.cross_entropy(out, y)
        out, e1 = classifier(x_train[j+1])
        L_prim = F.cross_entropy(out, y)
        if(L - L_prim == 0):
            break
        j += 1

    # --------  Query the labels of U and store them in matrix y

    # --------  Train the classiﬁcation model h on ( U, y )
    classifier.train()
    # -------- Predict X on h
    # -------- Return error rate on X
