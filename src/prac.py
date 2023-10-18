import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # Import ticker module for custom formatting
import numpy as np
import scienceplots
import torch
import torchvision
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import randomized_svd
from torch import nn

# from solo.backbones import resnet18
from torchvision.models import resnet18

from data_loader import get_data_loader
from utils import get_analyzer, get_model

os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"


def mean_center(X):
    mean = torch.mean(X, dim=1, keepdim=True)
    X_prime = X - mean
    return X_prime


def random_projection_method(X, b):
    X = mean_center(X)
    G = torch.randn(b, X.size(0))
    G = G / torch.norm(G, dim=0, keepdim=True)  # Normalize columns to unit length
    X_reduced = torch.mm(G, X)
    S = torch.linalg.svdvals(X_reduced)
    S_squared = S**2
    S_normalized = S_squared / torch.sum(S_squared)
    return S_normalized.detach()


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


# representations_to_full_singular_values("resnet34_0")


# # compute_error_original_vs_random_projection("resnet34_0", "16_16384.pt")
model_name = "resnet34_0"
target_dim = "16384"
original = True
files = os.listdir(f"values/representations/{model_name}")
files = [file.split(".")[0] for file in files if target_dim in file]
files.sort(key=lambda x: int(x.split("_")[0]), reverse=True)
# print(files)
compute_error_original_vs_random_projection(model_name, files)
