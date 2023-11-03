import math
import os
import time
from collections import OrderedDict

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # Import ticker module for custom formatting
import numpy as np
import torch
import torch as pt
import torch.nn.functional as F
import torchvision
from flowtorch.analysis import SVD
from sklearn import random_projection
from timm.models.layers import trunc_normal_
from torch import nn

# from solo.backbones import resnet18
from torchvision.models import resnet18, resnet34
from tqdm import tqdm

from data_loader import get_balanced_imagenet_input_data, get_data_loader
from models.models import IterativeKNN, convnextv2_fcmae
from utils import get_model
from utils.utils import compute_X_reduced, get_size, random_projection_method

os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"


def get_ranks(sigs, threshold):
    ranks = []
    for sig in sigs:
        count = (sig > threshold).sum().item()
        ranks.append(count)
    return ranks


def get_dynamic_ranks(model_name, sigs):
    model_name = model_name.split("_")[0]
    dims = torch.load(f"values/dimensions/{model_name}.pt")
    ranks = []
    for i, sig in enumerate(sigs):
        # eps = torch.finfo(torch.float32).eps * len(sig)
        eps = torch.finfo(torch.float32).eps * dims[i]
        # eps = torch.finfo(torch.float32).eps * 300

        threshold = torch.max(sig) * eps
        count = (sig > threshold).sum().item()
        ranks.append(count)
    return ranks


def compuate_singular_values(representation):
    cov = torch.cov(representation)
    print(cov.shape)
    S = torch.linalg.svdvals(cov)
    print(S.shape)
    return S


def compute_max_abs_error(s_reduced, s_original):
    # s_original = s_original / torch.sum(s_original)
    s_reduced_padded = torch.cat((s_reduced, torch.zeros(len(s_original) - len(s_reduced))))
    max_abs_error = (s_reduced_padded - s_original).abs().max()
    return max_abs_error.item()


def compute_mean_square_error(s_reduced, s_original):
    s_reduced_padded = torch.cat((s_reduced, torch.zeros(len(s_original) - len(s_reduced))))
    mean_sqaure_error = torch.mean((s_reduced_padded - s_original) ** 2)
    return mean_sqaure_error.item()


def representations_to_full_singular_values(model_name):
    def helper(model_name, file):
        load_path = f"values/representations/{model_name}/{file}.pt"
        print(f"processing {file}")

        re = torch.load(load_path).detach().cpu()
        print(f"representation shape: {re.shape}")
        cov_mat = torch.cov(re, correction=1)
        print(f"cov_mat shape: {cov_mat.shape}")
        S = torch.linalg.svdvals(cov_mat)
        # save
        save_path = f"values/full_sigs/{model_name}/{file}.pt"
        torch.save(S, save_path)

    files = os.listdir(f"values/representations/{model_name}")
    files.sort(key=lambda x: int(x.split("_")[0]), reverse=True)
    files = files[17:]
    print(files)
    for file in files:
        file = file.split(".")[0]
        helper(model_name, file)


def compute_error_original_vs_random_projection(model_name, files):
    def helper(model_name, file):
        projection_dims = [512, 1024, 2048, 4096, 8192, 16384]
        representation = torch.load(f"values/representations/{model_name}/{file}.pt").detach().cpu()
        features_size = representation.shape[0]

        target_singular_values = (
            torch.load(f"values/full_sigs/{model_name}/{file}.pt").detach().cpu()
        )
        target_singular_values = target_singular_values / torch.sum(target_singular_values)
        original_MAE = {}
        original_MSE = {}
        for dim in projection_dims:
            if dim >= features_size:
                break
            print(f"start computing original error for {dim} dimensions")
            cur_representation = representation[:dim]
            print(cur_representation.shape)
            cov_mat = torch.cov(cur_representation, correction=1)
            print(cov_mat.shape)
            S_reduced = torch.linalg.svdvals(cov_mat)
            S_reduced = S_reduced / torch.sum(S_reduced)

            MAE = compute_max_abs_error(S_reduced, target_singular_values)
            MSE = compute_mean_square_error(S_reduced, target_singular_values)
            print(f"MAE: {MAE}")
            print(f"MSE: {MSE}")
            original_MAE[dim] = MAE
            original_MSE[dim] = MSE
            print()
        torch.save(original_MAE, f"values/errors/{model_name}/original_MAE_{file}.pt")
        torch.save(original_MSE, f"values/errors/{model_name}/original_MSE_{file}.pt")

        projection_MAE = {}
        projection_MSE = {}
        for i in range(3):
            print(f"{i+1}th iteration ======================")
            for b in projection_dims:
                if b >= features_size:
                    break
                print(f"Computing singular values for {b} projection dimensions")
                S_reduced = random_projection_method(representation, b)
                MAE = compute_max_abs_error(S_reduced, target_singular_values)
                MSE = compute_mean_square_error(S_reduced, target_singular_values)
                if b not in projection_MAE:
                    projection_MAE[b] = [MAE]
                    projection_MSE[b] = [MSE]
                else:
                    projection_MAE[b].append(MSE)
                    projection_MSE[b].append(MSE)

                print()
        torch.save(projection_MAE, f"values/errors/{model_name}/projection_MAE_{file}.pt")
        torch.save(projection_MSE, f"values/errors/{model_name}/projection_MSE_{file}.pt")

    for file in files:
        helper(model_name, file)


def compute_save_singular_values(model_name, layer):
    representation = torch.load(f"values/representations/{model_name}/{layer}.pt").T
    print(f"representation shape: {representation.shape}")
    representation.to("cuda")
    cov_mat = torch.cov(representation, correction=1)
    print(f"cov_mat shape: {cov_mat.shape}")
    S = torch.linalg.svdvals(cov_mat)
    download_path = f"values/singular_values/{model_name}"
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    torch.save(S, f"{download_path}/{layer}.pt")


def create_data():
    x = pt.linspace(-10, 10, 100)
    t = pt.linspace(0, 20, 200)
    Xm, Tm = pt.meshgrid(x, t)

    data_matrix = 5.0 / pt.cosh(0.5 * Xm) * pt.tanh(0.5 * Xm) * pt.exp(1.5j * Tm)  # primary mode
    data_matrix += 0.5 * pt.sin(2.0 * Xm) * pt.exp(2.0j * Tm)  # secondary mode
    data_matrix += 0.5 * pt.normal(mean=pt.zeros_like(Xm), std=pt.ones_like(Xm))
    return data_matrix


def median_threshold(sig, m, n):
    # m feature
    # n sample

    beta = m / n
    print(beta)
    med = torch.median(sig)
    return (0.56 * (beta**3) - 0.95 * (beta**2) + 1.82 * (beta) + 1.43) * med


def sig_histogram(sig, layer):
    sig = sig[:1000]
    plt.bar(range(len(sig)), sig)
    print(sig[0])
    plt.xlabel(r"$i$")
    plt.ylabel(r"$\sigma_i$")
    plt.savefig(f"sig_histogram_{layer}.png")


def mean_center(X):
    mean = torch.mean(X, dim=1, keepdim=True)
    X_prime = X - mean
    return X_prime


def sparse_random_projection_matrix(original_dim, target_dim):
    # Generate sparse random projection matrix
    prob = 1 / (2 * torch.sqrt(torch.tensor([target_dim], dtype=torch.float)))
    s = (torch.rand(target_dim, original_dim) < prob).float()
    signs = torch.sign(torch.randn(target_dim, original_dim))
    G = torch.sqrt(torch.tensor([target_dim], dtype=torch.float)) * s * signs
    return G.T


def compute_sparesed_X_reduced(X, b):
    X = mean_center(X)  # F x N
    G = sparse_random_projection_matrix(b, X.size(0))  # b x F
    print(G.shape)
    G = G / torch.norm(G, dim=0, keepdim=True)  # Normalize columns to unit length
    X_reduced = torch.mm(G, X)  # b x N
    del G
    del X
    return X_reduced


class AggregateSpatialInformation(nn.Module):
    def __init__(self, target_size):
        super(AggregateSpatialInformation, self).__init__()
        self.target_size = target_size

    def forward(self, x):
        # Calculate the pooling size based on the target size
        _, _, h, w = x.size()
        pooling_size = int(h * w / self.target_size)
        return F.avg_pool2d(x, pooling_size, stride=pooling_size)


def vectorize_global_avg_pooling(x):
    output = F.avg_pool2d(x, 2, stride=2)
    flatten_output = output.view(output.size(0), -1)
    normalized_output = F.normalize(flatten_output, p=2, dim=1)
    return normalized_output


def vectorize_global_max_pooling(x):
    output = F.max_pool2d(x, 2, stride=2)
    flatten_output = output.view(output.size(0), -1)
    normalized_output = F.normalize(flatten_output, p=2, dim=1)
    return normalized_output


def get_dir(feature_type, normalize, knn):
    return f"values/cifar10/{'knn_ood_acc' if knn else 'ood_acc'}/{feature_type}/model{'_norm' if normalize else ''}"


def test_model(model):
    _, test_loader = get_data_loader("imagenet", batch_size=32)
    # get acc on test:
    model.eval()
    model.to("cuda:0")
    correct = 0
    total = 0
    for data, target in tqdm(test_loader):
        data, target = data.to("cuda:0"), target.to("cuda:0")
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(data)
    print(f"test acc: {correct/total}")


def extract_conv2d_layers(model):
    conv_layers = []

    for name, module in model.named_children():
        if isinstance(module, torch.nn.Conv2d):
            conv_layers.append(module)
        else:
            # Dive deeper into nested children
            for sub_name, sub_module in module.named_children():
                if isinstance(sub_module, torch.nn.Conv2d):
                    conv_layers.append(sub_module)

    return conv_layers


if __name__ == "__main__":
    # model = convnextv2_fcmae()
    # layers = []
    # for name, module in model.named_modules():
    #     if isinstance(module, nn.Conv2d):
    #         layers.append(module)
    # print(layers)
    # # conv_layers = extract_conv2d_layers(model)
    # # print(conv_layers)
    dinov2_1 = torch.load("values/imagenet/ood_acc/places/dinov2/0.pt")
    dinov2_2 = torch.load("values/imagenet/ood_acc/places/dinov2/1.pt")
    print(dinov2_1)
    print(dinov2_2)
