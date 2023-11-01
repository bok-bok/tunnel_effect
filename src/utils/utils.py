import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.models
from torchvision.models import (
    ConvNeXt_Base_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    Swin_B_Weights,
    convnext_base,
    resnet18,
    resnet34,
    resnet50,
    swin_b,
)

from models.models import MLP, ResNet18, ResNet34, ResNet34_GN


def mean_center(X):
    mean = torch.mean(X, dim=1, keepdim=True)
    X_prime = X - mean
    return X_prime


def compute_X_reduced(X, b):
    # F X N
    X = mean_center(X)

    # b X F
    G = torch.randn(b, X.size(0))
    G = G / torch.norm(G, dim=0, keepdim=True)  # Normalize columns to unit length

    # b X N
    X_reduced = torch.mm(G, X)
    del G
    del X
    return X_reduced


def random_projection_method(X, b, cov=False):
    X_reduced = compute_X_reduced(X, b).to("cuda")

    variance = torch.var(X_reduced)

    if not cov:
        s = torch.linalg.svdvals(X_reduced)
        del X_reduced

        return s.detach().cpu(), variance.detach().cpu()
    else:
        cov_mat = torch.cov(X_reduced, correction=1)
        print(f"cov_mat shape: {cov_mat.size()}")
        del X_reduced
        s = torch.linalg.svdvals(cov_mat)
        return s.detach().cpu(), variance.detach().cpu()


def vectorize_global_avg_pooling(x, normalize=False):
    output = F.avg_pool2d(x, 2, stride=2)
    # output = F.avg_pool2d(x, 4, stried=4)
    output = output.view(output.size(0), -1)
    if normalize:
        output = F.normalize(output, p=2, dim=1)
    return output


def vectorize_global_max_pooling(x, normalize=False):
    output = F.max_pool2d(x, 2, stride=2)
    # output = F.avg_pool2d(x, 4, stried=4)
    output = output.view(output.size(0), -1)
    if normalize:
        output = F.normalize(output, p=2, dim=1)
    return output


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
        return get_imagenet_model(model_name)


def get_imagenet_model(model_name: str):
    if "resnet18" in model_name:
        return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif "resnet34" in model_name:
        return resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    elif "resnet50" in model_name:
        if "swav" in model_name:
            print("loading resnet50 swav model")
            return torch.hub.load("facebookresearch/swav:main", "resnet50")
        else:
            print("loading resnet50 model")
            return resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    elif "convnext" in model_name.lower():
        return convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
    elif "swin" in model_name.lower():
        return swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)


def get_cifar_model(model_name, pretrained=True, weights_path=None):
    print(f"Loading {model_name} - cifar")
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
    elif "resnet50" in model_name:
        model = resnet50()
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.Identity()

        return model
    else:
        raise ValueError("model name not supported")
    return model


def get_size(data):
    if isinstance(data, torch.Tensor):
        total_bytes = data.element_size() * data.numel()
        total_gigabytes = total_bytes / (1024 * 1024 * 1024)
    else:
        # size of float32
        total_gigabytes = data / (1024 * 1024 * 1024)
    total_gigabytes = round(total_gigabytes, 2)
    return total_gigabytes
