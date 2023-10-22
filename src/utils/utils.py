import argparse

import matplotlib.pyplot as plt
import torch
from torchvision.models import ResNet50_Weights, resnet50

from models.models import MLP, MobileV3, ResNet18, ResNet34, ResNet34_GN


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet34")
    parser.add_argument("--pretrained", action="store_true")

    args = parser.parse_args()
    return args


def get_model(model_name: str, dataset: str, pretrained: bool = True, weights_path: str = None):
    if "cifar" in dataset:
        return get_cifar_model(model_name, pretrained, weights_path)
    else:
        return get_imagenet_model(model_name, weights_path)


def get_imagenet_model(model_name, weights_path=None):
    if "resnet50" in model_name:
        if "swav" in model_name:
            print("loading resnet50_swav")
            return torch.hub.load("facebookresearch/swav:main", "resnet50")
        else:
            print("loading resnet50")
            return resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)


def get_cifar_model(model_name, pretrained=True, weights_path=None):
    if "resnet18" in model_name:
        if pretrained:
            if "swav" in model_name:
                model = ResNet18(weights_path="weights/resnet18_swav.ckpt")
            else:
                model = ResNet18(weights_path="weights/resnet18.pth")
        else:
            model = ResNet18()
    elif "resnet50" in model_name:
        if pretrained:
            print("loading swav model")
            model = torch.hub.load("facebookresearch/swav:main", "resnet50")
        else:
            model = resnet50()

    elif "resnet34" in model_name:
        if "GN" in model_name:
            print("loading GN model")
            model = ResNet34_GN(weights_path=weights_path)
        else:
            if pretrained:
                if weights_path is not None:
                    print(f"loading {weights_path}")
                    model = ResNet34(weights_path=weights_path)
                else:
                    model = ResNet34(weights_path="weights/resnet34_0.pth")
            else:
                model = ResNet34()
    elif "mlp" in model_name:
        if pretrained:
            if weights_path is not None:
                print(f"loading {weights_path}")
                model = MLP(weights_path=weights_path)
            else:
                model = MLP(weights_path="weights/mlp_0.pth")

        else:
            model = MLP()
    else:
        raise ValueError("model name not supported")
    return model


def get_size(data):
    if isinstance(data, torch.Tensor):
        total_bytes = data.element_size() * data.numel()
        total_gigabytes = total_bytes / (1024 * 1024 * 1024)
    else:
        total_gigabytes = data / (1024 * 1024 * 1024)
    total_gigabytes = round(total_gigabytes, 2)
    return total_gigabytes
