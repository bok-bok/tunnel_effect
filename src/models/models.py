from abc import ABCMeta, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, normalize
from torchvision import models

from models.resnet_models_GN_WS import resnet34


class ResNet(nn.Module, metaclass=ABCMeta):
    def __init__(self, device="cpu", weights_path=None):
        """ResNet34 model for CIFAR-10

        Args:
            weights_path (str): Path to pretrained weights
            device (str): Device to load weights onto
        """
        super().__init__()
        self.init_model()
        self.resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )
        # self.resnet.bn1 = nn.BatchNorm2d(64)
        self.resnet.maxpool = nn.Identity()

        self.resnet.fc = nn.Linear(512, 10)
        if weights_path:
            if "swav" in weights_path:
                print("loading swav model")
                state = torch.load(weights_path)["state_dict"]
                for k in list(state.keys()):
                    if "backbone" in k:
                        state[k.replace("backbone.", "")] = state[k]
                    del state[k]
                self.resnet.load_state_dict(state, strict=False)
            else:
                print("loading pretrained model")
                self.resnet.load_state_dict(torch.load(weights_path, map_location=device))
        self.resnet.to(device)
        self.resnet.eval()

    @abstractmethod
    def init_model(self):
        # self.resnet = models.resnet34(weights=None)
        pass

    def forward(self, x):
        return self.resnet(x)


class ResNet18(ResNet):
    def init_model(self):
        self.resnet = models.resnet18(weights=None)


class ResNet34(ResNet):
    def init_model(self):
        self.resnet = models.resnet34(weights=None)


class ResNet34_GN(ResNet):
    def init_model(self):
        self.resnet = resnet34()


class MLP(nn.Module):
    def __init__(self, device="mps", weights_path=None):
        super(MLP, self).__init__()
        layers = []
        input_size = 32 * 32 * 3  # CIFAR-10 images are 32x32 pixels with 3 color channels

        layers = []
        layers.append(nn.Linear(input_size, 1028))
        layers.append(nn.ReLU())

        for _ in range(11):  # 11 layers in total (1 to 12)
            layers.append(nn.Linear(1028, 1028))
            layers.append(nn.ReLU())

        # 13
        layers.append(nn.Linear(1028, 10))  # 10 output classes for CIFAR-10

        # layer init
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

        self.layers = nn.Sequential(*layers)

        if weights_path:
            self.load_state_dict(torch.load(weights_path, map_location=device))

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.layers(x)
        return x


class IterativeKNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.nn_model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.data = None
        self.labels = None
        self.train_data = None
        self.predicted_labels = None
        self.target_labels = None

    def update(self, new_data, new_labels):
        new_data = np.array(new_data)
        new_labels = np.array(new_labels)
        if self.data is None:
            self.data = new_data
            self.labels = new_labels
        else:
            self.data = np.concatenate((self.data, new_data))
            self.labels = np.concatenate((self.labels, new_labels))

    def train(self):
        self.mean = self.data.mean(axis=0)
        self.std = self.data.std(axis=0)
        X_train = (self.data - self.mean) / self.std
        X_train = normalize(X_train, axis=1)

        # Normalization to unit length
        self.nn_model.fit(X_train, self.labels)

    def predict(self, X, target_labels):
        X = (X - self.mean) / self.std
        X = normalize(X, axis=1)

        labels = self.nn_model.predict(X)
        if self.target_labels is None:
            self.target_labels = target_labels
        else:
            self.target_labels = np.concatenate((self.target_labels, target_labels))
        if self.predicted_labels is None:
            self.predicted_labels = labels
        else:
            self.predicted_labels = np.concatenate((self.predicted_labels, labels))

    def get_accuracy(self):
        return np.sum(self.target_labels == self.predicted_labels) / len(self.target_labels)
