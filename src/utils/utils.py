import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from scipy.spatial.distance import cosine
from timm.models.vision_transformer import vit_base_patch16_224
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

from models.models import (
    MLP,
    ResNet18,
    ResNet34,
    ResNet34_GN,
    convnextv2_fcmae,
    get_resnet34_imagenet100,
    get_vgg11_by_class_num,
    get_vgg11_by_sample_num,
    get_vgg13_imagenet100,
    mae,
)

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
    # G = G.to(torch.float16)/
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


def vectorize_global_avg_pooling(x, patch_size=2, normalize=False, device="cuda"):
    # check dim of x
    if len(x.size()) == 4:
        if x.size(2) == 1 and x.size(3) == 1:
            print("cannot reduce spatial dim")
            output = x
            pass
        else:
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
    # print(output.size())
    output = output.reshape(output.size(0), -1)
    if normalize and output.size(0) > 1:
        sample_size, dim = output.size()
        batch_norm = nn.BatchNorm1d(dim, affine=True)
        batch_norm.to(device)
        batch_norm.weight.data.fill_(1.0)
        batch_norm.bias.data.fill_(0.0)
        batch_norm.weight.requires_grad = False
        batch_norm.bias.requires_grad = False

        output = batch_norm(output)
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
    elif "samples" in model_name:
        if "1000" in model_name:
            sample_per_class = 1000
        elif "100" in model_name:
            sample_per_class = 100
        elif "200" in model_name:
            sample_per_class = 200
        elif "500" in model_name:
            sample_per_class = 500
        return get_vgg11_by_sample_num(sample_per_class, pretrained)

    elif "class" in model_name:
        if "1000" in model_name:
            class_num = 1000
        elif "100" in model_name:
            class_num = 100
        elif "10" in model_name:
            class_num = 10
        elif "50" in model_name:
            class_num = 50
        return get_vgg11_by_class_num(class_num=class_num, pretrained=True)
    elif "vgg13_imagenet100" in model_name:
        print("loading vgg imagenet100 model")
        return get_vgg13_imagenet100(model_name, pretrained)
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
        model_name = "vit_base_patch16_224.mae"
        print(f"loading {model_name}")
        model = timm.create_model(model_name, pretrained=True)
        return model
    elif "convnextv2" in model_name.lower():
        model_name = "convnextv2_base.fcmae"
        print(f"loading {model_name}")
        model = timm.create_model(model_name, pretrained=True)
        return model
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
            model_name = "convnext_base.fb_in1k"
            model = timm.create_model(model_name, pretrained=True)
            return model
        else:
            model = timm.create_model(model_name, pretrained=False)
            return model
    elif "vit" in model_name.lower():
        model_name = "vit_base_patch16_224.augreg_in1k"
        model = timm.create_model(model_name, pretrained=True)
        return model
    elif "swin" in model_name.lower():
        if pretrained:
            return swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
        else:
            print("loading random init swin model")
            return swin_b()
    elif "dinov2" in model_name.lower():
        return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    elif "mugs" in model_name.lower():
        print("loading mugs model")
        model = vit_base_patch16_224()
        if not pretrained:
            print("loading random init mugs model")
            return model
        path = "weights/vit_base_backbone_400ep.pth"
        load_pretrained_weights(model, path, "state_dict", "vit_b_16", 16)
        return model
    elif "dino" in model_name.lower():
        model_name = "vit_base_patch16_224.dino"
        print(f"loading {model_name}")
        model = timm.create_model(model_name, pretrained=True)
        return model


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


def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # print(state_dict.keys())
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        # remove `encoder.` prefix induced by MAE
        state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(
            "Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg)
        )
    else:
        print("There is no reference weights available for this model => We use random weights.")


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# def computeNC1(labels, embeddings):
#     # Calculate class means (mu_c) and global mean (mu_G) accounting for class imbalance
#     unique_classes, class_counts = labels.unique(return_counts=True)
#     mu_c = torch.stack([embeddings[:, labels == c].mean(dim=1) for c in unique_classes])
#     mu_G = (mu_c * class_counts.unsqueeze(1)).sum(dim=0) / class_counts.sum()

#     # Calculate within-class variance
#     within_class_var = torch.tensor(0.0)
#     for c in unique_classes:
#         class_embeddings = embeddings[:, labels == c].T
#         class_mean = mu_c[unique_classes == c].squeeze(0)
#         within_class_var += ((class_embeddings - class_mean) ** 2).sum()

#     # Normalize the within-class variance by the number of training examples
#     normalized_within_class_var = within_class_var / (embeddings.shape[1] - unique_classes.size(0))

#     # Calculate between-class variance
#     between_class_var = ((mu_c - mu_G) ** 2).sum() * class_counts
#     normalized_between_class_var = between_class_var.sum() / (unique_classes.size(0) - 1)

#     # Calculate the ratio of within-class variance to between-class variance
#     nc1 = normalized_within_class_var / normalized_between_class_var

#     return normalized_within_class_var.item(), normalized_between_class_var.item(), nc1.item()


def average_pairwise_cosine_distance(X):
    print(f"X shape: {X.size()}")
    if len(X.shape) == 4:
        # if swin transformer, match the shape of (N, C, H, W)
        if X.shape[1] == X.shape[2]:
            X = X.permute(0, 3, 1, 2)

        N, C, H, W = X.shape
    elif len(X.shape) == 3:
        # remove positional embedding
        X = X[:, 1:]
        N, C, H = X.shape
        W = 1
    epsilon = 1e-10

    # Reshape and normalize
    # N X C X (H * W)
    X_reshaped = X.reshape(N, C, -1)

    norms = np.linalg.norm(X_reshaped, axis=2, keepdims=True) + epsilon
    X_normalized = X_reshaped / norms  # Normalize each channel

    total_distance = 0.0

    for n in range(N):
        # Compute cosine similarity matrix for each feature map
        cosine_similarity = np.dot(X_normalized[n], X_normalized[n].T)
        # Convert cosine similarity to cosine distance
        cosine_distance = 1 - cosine_similarity

        # Sum only the upper triangular part, excluding the diagonal
        # skip dividing by 2, since summing upper triangular part do same thing.
        total_distance += np.sum(np.triu(cosine_distance, k=1))

    # Calculate the average distance
    average_distance = total_distance / (N * C * C)
    return average_distance


def computeNC1(labels, embeddings):
    unique_classes, class_counts = labels.unique(return_counts=True)
    mu_c = torch.stack([embeddings[:, labels == c].mean(dim=1) for c in unique_classes])
    mu_G = (mu_c * class_counts.unsqueeze(1)).sum(dim=0) / class_counts.sum()

    within_class_var = torch.tensor(0.0)
    for c in unique_classes:
        class_embeddings = embeddings[:, labels == c]
        class_mean = mu_c[unique_classes == c].squeeze(0)
        # within_class_var += ((class_embeddings - class_mean) ** 2).sum()
        within_class_var += (
            (class_embeddings.T - class_mean) ** 2
        ).sum()  # transpose to make it work

    normalized_within_class_var = within_class_var / embeddings.shape[1]

    between_class_var = ((mu_c - mu_G) ** 2).sum(dim=1) * class_counts
    normalized_between_class_var = between_class_var.sum() / embeddings.shape[1]

    nc1 = normalized_within_class_var / normalized_between_class_var

    return normalized_within_class_var.item(), normalized_between_class_var.item(), nc1.item()


def stable_rank(A, maxIter=1000):
    # shape of A : (D, N)
    # Ensure the matrix A is in single precision
    A = A.float()

    # Compute the sum of squares of each row in A
    B = torch.sum(A * A, dim=1)

    # Power iteration to approximate the largest singular value
    s = power_iter(A, maxIter)

    F_norm_squared = torch.sum(B)

    # Calculate the stable rank using the Frobenius norm squared and the approximated spectral norm squared
    r = F_norm_squared / s
    return r


def power_iter(A, maxIter, tol=1e-4):
    # Random initialization of x, ensuring single precision
    x = torch.randn((A.shape[1],), dtype=torch.float32, device=A.device)

    for j in range(maxIter):
        # Matrix-vector multiplication
        v = A.T @ (A @ x)

        # Normalizing the vector v
        x_new = v / torch.norm(v)

        # Check for convergence
        if torch.norm(x_new - x) < tol:
            break

        x = x_new

    # Return the spectral norm approximation
    return torch.norm(v) / torch.norm(x)


# Example usage:
# Assuming `labels` and `embeddings` are defined and loaded properly as torch tensors
# labels = torch.tensor([...])
# embeddings = torch.tensor([...])
# within_class_var, between_class_var, nc1 = computeNC1(labels, embeddings)
# print(within_class_var, between_class_var, nc1)
