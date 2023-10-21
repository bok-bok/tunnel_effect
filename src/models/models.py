from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from torchvision import models

from models.resnet_models_GN_WS import resnet34


class MobileV3(nn.Module):
    def __init__(self, device="mps", weights_path=None):
        """ResNet34 model for CIFAR-10

        Args:
            weights_path (str): Path to pretrained weights
            device (str): Device to load weights onto
        """
        super().__init__()
        self.model = models.mobilenet_v3_large(pretrained=False)
        in_features = self.model._modules["classifier"][-1].in_features
        self.model._modules["classifier"][-1] = nn.Linear(in_features, 10, bias=True)
        if weights_path:
            print("loading model")
            self.model.load_state_dict(torch.load(weights_path, map_location=device))
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # self.model.to(device)
        self.model.eval()

    def forward(self, x):
        return self.model(x)


class ResNet(nn.Module, metaclass=ABCMeta):
    def __init__(self, device="mps", weights_path=None):
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
