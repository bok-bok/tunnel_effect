import os
import ssl

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import ResNet34_Weights, resnet18

from data_loader import get_data_loader
from models.models import MLP, MLP_5, MobileV3, ResNet18, ResNet34
from utils import get_analyzer, get_model

ssl._create_default_https_context = ssl._create_unverified_context
from pydantic import BaseModel

from utils.analyzer import ConvAnalyzer, MLPAnalyzer, ResNetAnalyzer

# use pydantic create a data data_name class that can be only cifar10 or imagent
# code here
if __name__ == "__main__":
    # config
    model_name = f"resnet34"
    pretrained = True
    data_name = "cifar10"
    batch_size = 512

    input_size = 10000
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # use model_name and pretrained to get model
    model = get_model(model_name, pretrained)
    model.to(device)

    # dummy input help analyzer to get the shape of output
    if data_name == "cifar10":
        dummy_input = torch.randn(1, 3, 32, 32).to(device)
    else:
        dummy_input = torch.randn(1, 3, 224, 224).to(device)

    train_dataloader, test_dataloader = get_data_loader(data_name, batch_size=batch_size)
    _, input_loader = get_data_loader(data_name, batch_size=input_size)

    input_data = next(iter(input_loader))[0].to(device)

    analyzer = get_analyzer(model, model_name, dummy_input)
    # analyzer.download_qr_sigs_values(input_data)
    analyzer.download_representations(input_data)
    # analyzer.download_online_variances(test_dataloader, input_size)
    # analyzer.download_means(input_data)

    # analyzer.download_online_means(test_dataloader, input_size)
    # analyzer.download_values(test_dataloader, input_data, input_size)
    # analyzer.download_singular_values(input_data)
    # analyzer.download_online_variances(test_dataloader, input_size)
    # analyzer.download_accuarcy(train_dataloader, test_dataloader)
