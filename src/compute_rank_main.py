import argparse
import time

import torchvision
from torch.utils.data import DataLoader, Subset

from analyzer import get_analyzer
from data_loader import (
    get_balanced_imagenet100_input_data,
    get_balanced_imagenet_input_data,
    get_balanced_places_input_data,
    get_cifar100_input_data,
    get_cifar_input_data,
    get_CIFAR_transforms,
    get_data_loader,
    get_NINCO_input_data,
)
from utils import get_model


def parser():
    parser = argparse.ArgumentParser(description="Get GPU numbers")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--data", type=str, required=True, help="Data name")
    parser.add_argument("--input_size", type=int, default=15000, help="Input size")
    parser.add_argument("--random", action="store_true", help="Random projection")
    parser.add_argument("--resolution", type=int, default=224, help="Image resolution")
    args = parser.parse_args()
    return args.model, args.data, args.random, args.input_size, args.resolution


if __name__ == "__main__":
    start = time.time()
    model_name, data_name, RANDOM, input_size, resolution = parser()
    if RANDOM:
        pretrained = False
    else:
        pretrained = True

    # load data
    if "cifar" in data_name:
        pretraind_data_name = "cifar10"
    elif "imagenet100" == data_name:
        pretraind_data_name = "imagenet100"
    else:
        pretraind_data_name = "imagenet"

    if data_name == "cifar10":
        input_data = get_cifar_input_data()
    elif data_name == "cifar100":
        input_data = get_cifar100_input_data()
    elif data_name == "imagenet100":
        print("Loading imagenet100 data")
        input_data = get_balanced_imagenet100_input_data(
            resolution=resolution, sample_size=input_size
        )
    elif data_name == "imagenet":
        print("Loading imagenet data")
        input_data = get_balanced_imagenet_input_data(input_size)
    elif data_name == "places":
        print("Loading Places data")
        input_data = get_balanced_places_input_data(input_size)
    elif data_name == "ninco":
        input_data = get_NINCO_input_data()

    input_data, label = input_data

    GAP = False
    main_device = "cpu"
    svd_device = "cuda:0"

    model = get_model(model_name, pretraind_data_name, pretrained=pretrained)

    analyzer = get_analyzer(model, model_name, pretraind_data_name)
    analyzer.add_gpus(main_device, svd_device)
    analyzer.download_singular_values(input_data, data_name, GAP, pretrained)
    end = time.time()
    print(f"Total time: {end-start}")
