import argparse

import torchvision
from torch.utils.data import DataLoader, Subset

from analyzer import get_analyzer
from data_loader import (
    get_cifar100_input_data,
    get_cifar_input_data,
    get_CIFAR_transforms,
    get_data_loader,
)
from utils import get_model


def parser():
    parser = argparse.ArgumentParser(description="Get GPU numbers")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--data", type=str, required=True, help="Data name")
    parser.add_argument("--OOD", action="store_true", help="Out of distribution")
    parser.add_argument("--random", action="store_true", help="Random projection")
    args = parser.parse_args()
    return args.model, args.data, args.OOD, args.random


if __name__ == "__main__":
    model_name, data_name, OOD, RANDOM = parser()
    if RANDOM:
        pretrained = False
    else:
        pretrained = True

    if OOD:
        input_data = get_cifar100_input_data()
    else:
        input_data = get_cifar_input_data()
    print(input_data.shape)
    model_name = "resnet34"
    data_name = "cifar10"
    input_data = get_cifar_input_data()
    pretrained = False
    OOD = False

    model = get_model(model_name, data_name, pretrained=pretrained)

    analyzer = get_analyzer(model, model_name, data_name)

    analyzer.download_singular_values(input_data, OOD, pretrained)
