import argparse
import sys

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
    ViT_B_16_Weights,
    convnext_base,
    resnet18,
    resnet34,
    resnet50,
    swin_b,
    vit_b_16,
)

from models.models import MLP, ResNet18, ResNet34, ResNet34_GN, convnextv2_fcmae, mae

sys.path.append("models/CLIP")
sys.path.append("models/CLIP/clip")
from models.CLIP import clip
from models.CLIP.clip import CLIP, ResidualAttentionBlock


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


def vectorize_global_avg_pooling(x, patch_size=2, normalize=False):
    # check dim of x
    # print(x.size())
    if len(x.size()) == 4:
        output = F.avg_pool2d(x, patch_size, stride=patch_size)
    elif len(x.size()) == 3:
        # print(x.size())
        patch_size = 2
        x = x[:, 1:]
        x_transposed = x.transpose(1, 2)
        output = F.avg_pool1d(x_transposed, patch_size, stride=patch_size)
        output = output.transpose(1, 2)

        # output = F.avg_pool1d(x, patch_size, stride=patch_size)
        # print(output.size())
    output = output.reshape(output.size(0), -1)
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
        return get_imagenet_model(model_name, pretrained)


def get_imagenet_model(model_name: str, pretrained=True):
    if "resnet18" in model_name:
        if pretrained:
            return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            print("loading random init resnet18 model")
            return resnet18()
    elif "resnet34" in model_name:
        if pretrained:
            return resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        else:
            print("loading random init resnet34 model")
            return resnet34()
    elif "resnet50" in model_name:
        if "swav" in model_name:
            path = "weights/swav_800ep_pretrain.pth.tar"
            print(f"loading resnet50 swav model {path}")
            model = resnet50()
            state_dict = torch.load(path, map_location="cpu")
            # remove prefixe "module."
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            for k, v in model.state_dict().items():
                if k not in list(state_dict):
                    print('key "{}" could not be found in provided state dict'.format(k))
                elif state_dict[k].shape != v.shape:
                    print(
                        'key "{}" is of different shape in model and provided state dict'.format(k)
                    )
                    state_dict[k] = v
            model.load_state_dict(state_dict, strict=False)
            return model
            # return torch.hub.load("facebookresearch/swav:main", "resnet50")
        else:
            if pretrained:
                print("loading resnet50 model")
                return resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            else:
                print(f"loading random init resnet50 model")
                return resnet50()
    elif "mae" in model_name.lower():
        return mae(pretrained=pretrained)
    elif "convnextv2" in model_name.lower():
        print("loading convnextv2 model")
        return convnextv2_fcmae(pretrained=pretrained)
    elif "clip" in model_name.lower():
        if pretrained:
            print("loading imagenet1k pretrained clip model")
            return clip.load("ViT-B/32", jit=False)
        else:
            print("loading random init clip model")
            return clip.load("ViT-B/32", jit=False, pretrained=False)

    elif "convnext" in model_name.lower():
        if pretrained:
            print("loading imagenet1k pretrained convnext model")
            return convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        else:
            print("loading random init convnext model")
            return convnext_base()
    elif "vit" in model_name.lower():
        return vit_b_16(ViT_B_16_Weights.IMAGENET1K_V1)
    elif "swin" in model_name.lower():
        if pretrained:
            return swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
        else:
            print("loading random init swin model")
            return swin_b()
    elif "dinov2" in model_name.lower():
        return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")


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
                    model = ResNet34(weights_path="weights/resnet34.pth")
            else:
                print("loading random model")
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
