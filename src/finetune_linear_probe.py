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
    get_yousuf_imagenet100,
)
from utils import get_model

ssl._create_default_https_context = ssl._create_unverified_context

# use pydantic create a data data_name class that can be only cifar10 or imagent
# code here


def parser():
    parser = argparse.ArgumentParser(description="Get GPU numbers")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--pretrained_data", type=str, required=True, help="Pretrained data")
    parser.add_argument("--data", type=str, required=True, help="Data name")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--gpu1", type=int, required=True, help="First GPU number to use")
    parser.add_argument("--gpu2", type=int, required=True, help="Second GPU number to use")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")

    args = parser.parse_args()
    gpu1 = f"cuda:{args.gpu1}"
    gpu2 = f"cuda:{args.gpu2}"
    return args.model, args.pretrained_data, args.data, args.batch_size, gpu1, gpu2, args.lr


if __name__ == "__main__":
    model_name, pretrained_data, data_name, batch_size, main_device, classifier_device, lr = parser()
    pretrained = True
    resolution = 224

    # get class_num and resolution
    if "imagenet100" in model_name:
        class_num = 100
        if "32" in model_name:
            resolution = 32
        elif "64" in model_name:
            resolution = 64
        elif "128" in model_name:
            resolution = 128
        elif "224" in model_name or "down_up" in model_name:
            resolution = 224

        else:
            raise ValueError("resolution not found")

    if "cifar" in data_name:
        resolution = 32
        class_num = 10

    weight_path = f"weights/{model_name}.pth"

    if "resnet34_original" in model_name:
        epochs = model_name.split("_")[-1]
        weight_path = f"weights/resnet34/resnet34_{epochs}.pth"

    # use model_name and pretrained to get model
    model = get_model(model_name, data_name, pretrained, weight_path)
    # main_device = "cpu"
    model.to(main_device)
    if data_name == "yousuf_imagenet100":
        if "vit" in model_name:
            checkpoint = torch.load(f"weights/vit/{resolution}.pth")
        args = checkpoint["args"]
        train_dataloader, test_dataloader = get_yousuf_imagenet100(args, batch_size)
        data_name = "imagenet100"
        pretrained_data = "imagenet100"
    else:
        train_dataloader, test_dataloader = get_data_loader(
            data_name, class_num=class_num, batch_size=batch_size, resolution=resolution
        )

    # start = time.time()
    analyzer = get_analyzer(model, model_name, data_name)
    analyzer.add_gpus(main_device, classifier_device)
    analyzer.set_lr(lr=lr)
    acc = analyzer.check_last_layer_acc(train_dataloader, test_dataloader, resolution)
    print(f"lr: {lr}, acc: {acc}")
    # analyzer.download_accuarcy(train_dataloader, test_dataloader, pretrained_data, resolution, GAP=True)

    # analyzer.inspect_layers_dim(dummy_input)
    # end = time.time()
    # print(f"total time  : {end - start}")
