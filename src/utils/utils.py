import argparse

import matplotlib.pyplot as plt
import torch
from torchvision.models import MobileNet_V3_Large_Weights, ResNet18_Weights, resnet50

from models.models import MLP, MobileV3, ResNet18, ResNet34
from utils.analyzer import MLPAnalyzer, ResNetAnalyzer


def get_analyzer(model, model_name: str, dummy_input):
    if "mlp" in model_name:
        return MLPAnalyzer(model, model_name, dummy_input)

    elif "resnet" in model_name:
        return ResNetAnalyzer(model, model_name, dummy_input)
    else:
        raise ValueError("model name not supported")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet34")
    parser.add_argument("--pretrained", action="store_true")

    args = parser.parse_args()
    return args


def get_model(model_name: str, pretrained: bool = True, weights_path: str = None):
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
