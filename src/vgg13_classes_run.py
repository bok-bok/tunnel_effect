import argparse
import ssl
import time

import torch
from PIL import Image

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
class_num_to_samples_size = {10: 1000, 50: 200, 100: 100, 1000: 10}


def parser():
    parser = argparse.ArgumentParser(description="Get GPU numbers")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--pretrained_data", type=str, required=True, help="Pretrained data")
    parser.add_argument("--data", type=str, required=True, help="Data name")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--gpu1", type=int, required=True, help="First GPU number to use")
    parser.add_argument("--gpu2", type=int, required=True, help="Second GPU number to use")

    args = parser.parse_args()
    gpu1 = f"cuda:{args.gpu1}"
    gpu2 = f"cuda:{args.gpu2}"
    return args.model, args.pretrained_data, args.data, args.batch_size, gpu1, gpu2


if __name__ == "__main__":
    model_name, pretrained_data, data_name, batch_size, main_device, classifier_device = parser()
    pretrained = True
    resolution = 224

    if "10" in model_name:
        class_num = 10
    elif "50" in model_name:
        class_num = 50
    elif "100" in model_name:
        class_num = 100
    elif "1000" in model_name:
        class_num = 1000
    else:
        raise ValueError("class_num not found")

    train_sample_per_class = class_num_to_samples_size[class_num]
    test_sample_per_class = 50
    if data_name != "imagenet":
        class_num = None

    weight_path = f"weights/{model_name}.pth"
    # use model_name and pretrained to get model
    model = get_model(model_name, data_name, pretrained, weight_path)
    # main_device = "cpu"
    model.to(main_device)
    print(class_num)

    train_dataloader, test_dataloader = get_data_loader(
        data_name,
        train_sample_per_class,
        test_sample_per_class,
        class_num=class_num,
        batch_size=batch_size,
        resolution=resolution,
    )

    # start = time.time()
    analyzer = get_analyzer(model, model_name, data_name)
    analyzer.add_gpus(main_device, classifier_device)
    analyzer.download_accuarcy(train_dataloader, test_dataloader, pretrained_data, resolution)
    # analyzer.inspect_layers_dim(dummy_input)
    # end = time.time()
    # print(f"total time  : {end - start}")
