from Strategies import RandomSampling, MarginSampling, KMeansSampling, VAL
from Helpers.model import get_net
from Helpers.dataset import get_dataset, get_handler
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch

NB_INITIAL_ETIQUITE = 1000
NB_QUERY = 500
NB_ITERATIONS = 5
DATASET = 'MNIST'
ARGS_POOL = {'MNIST':
             {'n_epoch': 10, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
              'loader_tr_args': {'batch_size': 64, 'num_workers': 1},
              'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
              'optimizer_args': {'lr': 0.01, 'momentum': 0.5}},
             'FashionMNIST':
             {'n_epoch': 10, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
              'loader_tr_args': {'batch_size': 64, 'num_workers': 1},
              'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
              'optimizer_args': {'lr': 0.01, 'momentum': 0.5}},
             'SVHN':
             {'n_epoch': 20, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
              'loader_tr_args': {'batch_size': 64, 'num_workers': 1},
              'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
              'optimizer_args': {'lr': 0.01, 'momentum': 0.5}},
             }


def label_initial_data():
    # ========== générer un pool étiqueté initial
    index_etiquite = np.zeros(n_pool, dtype=bool)
    index_temp = np.arange(n_pool)
    np.random.shuffle(index_temp)
    index_etiquite[index_temp[:NB_INITIAL_ETIQUITE]] = True
    return index_etiquite


def main_function(strategy, index_etiquite):
    # ========= Trainer le modéle
    strategy.train()
    # ========= Tester le modéle
    P = strategy.predict(x_test, y_test)
    # ========= Calculer la precision Initial
    presions = np.zeros(NB_ITERATIONS + 1)
    presions[0] = 1.0 * (y_test == P).sum().item() / len(y_test)
    print('\nStratégie : ' + type(strategy).__name__)
    print('\tItération 0' + ' ............................. ' +
          'Précision: {}'.format(presions[0]))

    # ========= Repeter pour le nombre d'iteration
    for rd in range(1, NB_ITERATIONS + 1):
        q_idxs = strategy.query(NB_QUERY)
        index_etiquite[q_idxs] = True
        # ========= mise a jour la dataset
        strategy.update(index_etiquite)
        strategy.train()
        # ========= Calculer la precision
        P = strategy.predict(x_test, y_test)
        presions[rd] = 1.0 * (y_test == P).sum().item() / len(y_test)
        print('\tItération {}'.format(
            rd) + ' ............................. ' + 'Précision: {}'.format(presions[rd]))
    return presions


if __name__ == '__main__':
    result = []
    # ======= Définir le seed
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.enabled = False

    # ========== Diviser la dataset
    x_train, y_train, x_test, y_test = get_dataset(DATASET)
    x_train = x_train[:40000]
    y_train = y_train[:40000]

    # ======= Initialisation
    n_pool = len(y_train)
    print("\nDataSet : " + DATASET)
    print('\tNombre de pool étiqueté: {}'.format(NB_INITIAL_ETIQUITE))
    print('\tNombre de pool non étiqueté: {}'.format(
        n_pool - NB_INITIAL_ETIQUITE))
    print('\tNombre de pool de test: {}'.format(len(y_test)))

    # ========= RandomSampling
    strategy = RandomSampling(x_train, y_train, label_initial_data(), get_net(
        DATASET), get_handler(DATASET), ARGS_POOL[DATASET])
    result.append(main_function(strategy, label_initial_data()))

    # ========= MarginSampling
    strategy = MarginSampling(x_train, y_train, label_initial_data(), get_net(
        DATASET), get_handler(DATASET), ARGS_POOL[DATASET])
    result.append(main_function(strategy, label_initial_data()))

    # ========= KMeansSampling
    strategy = KMeansSampling(x_train, y_train, label_initial_data(), get_net(
        DATASET), get_handler(DATASET), ARGS_POOL[DATASET])
    result.append(main_function(strategy, label_initial_data()))

    labels = ['Échantillonnage aléatoire', 'Échantillonnage de marge',
              'Échantillonnage K-means', 'Échantillonnage VAL']
    colors = ['blue', 'red', 'green', 'pink', 'black', 'orange']
    for i in range(0, len(result)):
        plt.plot(range(0, len(result[i])), result[i], color=colors[i],
                 alpha=0.6, label=labels[i], linestyle='-', marker='o')
    plt.xlabel("Iterations")
    plt.ylabel('Précision')
    plt.title('Comparaison entre les différents stratégies | ' + str(NB_ITERATIONS) + ' itération | pool étiquité ' +
              str(NB_INITIAL_ETIQUITE) + ' | pool non étiquité ' + str(n_pool - NB_INITIAL_ETIQUITE))
    plt.grid()
    plt.legend()
    plt.show()
