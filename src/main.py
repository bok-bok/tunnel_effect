import argparse
import ssl
import time

import torch

from analyzer import get_analyzer
from data_loader import (
    get_balanced_imagenet_input_data,
    get_cifar_input_data,
    get_data_loader,
)
from utils import get_model

ssl._create_default_https_context = ssl._create_unverified_context

# use pydantic create a data data_name class that can be only cifar10 or imagent
# code here


def parser():
    parser = argparse.ArgumentParser(description="Get GPU numbers")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--data", type=str, required=True, help="Data name")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--gpu1", type=int, required=True, help="First GPU number to use")
    parser.add_argument("--gpu2", type=int, required=True, help="Second GPU number to use")

    args = parser.parse_args()
    gpu1 = f"cuda:{args.gpu1}"
    gpu2 = f"cuda:{args.gpu2}"
    return args.model, args.data, args.batch_size, gpu1, gpu2


if __name__ == "__main__":
    model_name, data_name, batch_size, main_device, classifier_device = parser()
    pretrained = True

    weight_path = f"weights/{model_name}.pth"

    # use model_name and pretrained to get model

    model = get_model(model_name, data_name, pretrained, weight_path)
    model.to(main_device)

    train_dataloader, test_dataloader = get_data_loader(data_name, batch_size=batch_size)

    start = time.time()
    analyzer = get_analyzer(model, model_name, data_name)
    analyzer.add_gpus(main_device, classifier_device)
    analyzer.download_accuarcy(train_dataloader, test_dataloader)
    # analyzer.inspect_layers_dim()
    end = time.time()
    print(f"total time  : {end - start}")
