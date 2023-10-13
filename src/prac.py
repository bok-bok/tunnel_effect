import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import randomized_svd
from torch import nn

# from solo.backbones import resnet18
from torchvision.models import resnet18

from data_loader import get_data_loader
from utils import get_analyzer


def mean_center(X):
    mean = torch.mean(X, dim=1, keepdim=True)
    X_prime = X - mean
    return X_prime


def qr_svd_method(X):
    X = mean_center(X)
    Q, R = torch.linalg.qr(X_prime.T)
    U, S, V = torch.linalg.svd(R, full_matrices=False)

    S_squared = S**2
    S_normalized = S_squared / torch.sum(S_squared)
    return S_normalized.detach()


def direct_svd_method(X_prime):
    X = mean_center(X)
    U, S, V = torch.svd(X_prime)
    S_squared = S**2
    S_normalized = S_squared / torch.sum(S_squared)
    return S_normalized


def random_projection_method(X, b):
    X = mean_center(X)
    G = torch.randn(b, X.size(0))
    G = G / torch.norm(G, dim=1, keepdim=True)  # Normalize columns to unit length
    X_reduced = torch.mm(G, X)
    U, S, V = torch.linalg.svd(X_reduced, full_matrices=False)
    S_squared = S**2
    S_normalized = S_squared / torch.sum(S_squared)
    return S_normalized


data_name = "cifar10"
train_dataloader, test_dataloader = get_data_loader(data_name, batch_size=500)

# vgg19 = torchvision.models.vgg19(pretrained=False, num_classes=10)

model = torchvision.models.resnet34(pretrained=False)
model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
model.maxpool = nn.Identity()
outputs = []

input_data = next(iter(test_dataloader))[0]


def hook_fn(module, input, output):
    output = output.view(output.size(0), -1)
    outputs.append(output)


hooks = []


def register_hooks(model):
    for layer in model.children():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            hook = layer.register_forward_hook(hook_fn)
            hooks.append(hook)
        # If the layer has children, recursively add hooks to those too
        if len(list(layer.children())) > 0:
            register_hooks(layer)


def register_hooks_resnet(model):
    layers = []
    layers.append(model.conv1)
    for block_name in ["layer1", "layer2", "layer3", "layer4"]:
        block = getattr(model, block_name)
        for layer in block:
            layers.extend([layer.conv1, layer.conv2])
    for layer in layers:
        layer.register_forward_hook(hook_fn)


# input_data = torch.randn(10000, 65536)
# input_data = input_data.T
# X_prime = mean_center(input_data)
# S = qr_svd_method(X_prime)
# print(S[:10])


# register_hooks_resnet(model)
print("laodign")
representations = torch.load("values/representations/resnet34.pt")
# sigs = torch.load("values/sigs/resnet34.pt")

qr_sigs = []
for re in representations:
    re = re.T
    print(re.size())
    if re.size(0) > 8000:
        re = re[:8000]
    X_prime = mean_center(re)
    sig = qr_svd_method(X_prime)
    print(sig.size())
    qr_sigs.append(sig)

torch.save(qr_sigs, "values/qr_sigs/resnet34.pt")

# for idx, output in enumerate(outputs):
#     # output = output.flatten()
#     sig = sigs[idx]
#     normalized_sig = sig / torch.sum(sig)
#     qr_svd = qr_svd_method(output)
#     print(f"size comparision : {qr_svd.size()} vs {normalized_sig.size()}")
#     print(f"qr_svd: {qr_svd[:10]}")
#     print(f"normalized_sig: {normalized_sig[:10]}")

#     print(f"layer {idx} shape: {output.shape}")
