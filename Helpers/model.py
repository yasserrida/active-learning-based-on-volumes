import torch.nn as nn
import torch.nn.functional as F


def get_classifier(name):
    if name == 'MNIST':
        return Net1
    elif name == 'FashionMNIST':
        return Net1
    elif name == 'SVHN':
        return Net2


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        # ======= Applique une convolution 2D
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # ======= Mettre à zéro au hasard des canaux entiers
        self.conv2_drop = nn.Dropout2d()
        # ======= Applique une transformation linéaire
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # ======= Applique une transformation non linéaire (Pooling functions)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        e1 = F.relu(self.fc1(x))
        # ======== Dropout function
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        # ======= Applique une convolution 2D
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        # ======= Mettre à zéro au hasard des canaux entiers
        self.conv3_drop = nn.Dropout2d()
        # ======= Applique une transformation linéaire
        self.fc1 = nn.Linear(1152, 400)
        self.fc2 = nn.Linear(400, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        # ======= Applique une transformation non linéaire (Pooling functions)
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 1152)
        x = F.relu(self.fc1(x))
        e1 = F.relu(self.fc2(x))
        # ======== Dropout function
        x = F.dropout(e1, training=self.training)
        x = self.fc3(x)
        return x, e1

    def get_embedding_dim(self):
        return 50
