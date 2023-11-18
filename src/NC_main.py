import argparse
import ssl
import time

import torch
from PIL import Image

from analyzer import get_analyzer
from data_loader import (
    get_balanced_imagenet100_input_data,
    get_balanced_imagenet_input_data,
    get_cifar_input_data,
    get_imagenet_input_data,
)
from utils import get_model

ssl._create_default_https_context = ssl._create_unverified_context

# use pydantic create a data data_name class that can be only cifar10 or imagent
# code here


def parser():
    parser = argparse.ArgumentParser(description="Get GPU numbers")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--pretrained_data", type=str, required=True, help="Pretrained data")
    parser.add_argument("--type", type=str, required=True, help="Type of data")
    parser.add_argument("--input_size", type=int, required=True, help="Input size of model")

    args = parser.parse_args()
    return args.model, args.pretrained_data, args.type, args.input_size


if __name__ == "__main__":
    model_name, data_name, metrics_type, input_size = parser()
    main_device = "cpu"
    classifier_device = "cuda:0"

    weight_path = f"weights/{model_name}.pth"
    # use model_name and pretrained to get model
    model = get_model(model_name, data_name, True, weight_path)
    model.to(main_device)

    # input_data = get_cifar_input_data()
    if data_name == "imagenet":
        # input_data = get_balanced_imagenet_input_data(input_size)
        input_data = get_imagenet_input_data(input_size)
    elif data_name == "cifar10":
        print("loading cifar10")
        input_data = get_cifar_input_data(input_size)
    elif data_name == "imagenet100":
        if "32" in model_name:
            resolution = 32
        elif "64" in model_name:
            resolution = 64
        elif "128" in model_name:
            resolution = 128
        elif "224" in model_name:
            resolution = 224
        input_data = get_balanced_imagenet100_input_data(resolution, input_size)

    input_data, input_label = input_data
    print(input_data.shape)
    # start = time.time()
    analyzer = get_analyzer(model, model_name, data_name)
    analyzer.add_gpus(main_device, classifier_device)
    if metrics_type == "NC_4":
        analyzer.download_NC4(input_data, input_label)
    elif metrics_type == "NC_2":
        print("NC_2")
        analyzer.download_NC2(input_data, input_label)
    elif metrics_type == "NC_1":
        analyzer.download_NC1(input_data, input_label)
