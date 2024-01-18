import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import torch
from scipy.interpolate import interp1d
from scipy.stats import spearmanr

plt.style.use("science")

plt.rc("text", usetex=False)  # disable LaTeX font
plt.rc("font", size=17, weight="bold")
os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"

plt.rc("lines", linewidth=3)

F = 25
ID_COLOR = "#377EB8"
OOD_COLOR = "#A65628"
OOD_2_COLOR = "#984EA3"
RANK_COLOR = "#E41A1C"
CB_color_cycle = [
    "#377EB8",
    "#FF7F00",
    "#4DAF4A",
    "#F781BF",
    "#A65628",
    "#984EA3",
    "#999999",
    "#E41A1C",
    "#DEDE00",
]


def find_tunnel_start(accuracies, threshold=0.95):
    final_acc = max(accuracies)
    # find when accuracy reach 95% final accuracy
    for idx, acc in enumerate(accuracies):
        if acc > final_acc * threshold:
            if accuracies[idx] == final_acc:
                return None
            return idx + 1


def plot_error_dimension(model_name, dim, original=False, download=False):
    errors = torch.load(f"values/errors/{model_name}/projection_{dim}.pt")
    errors_mean = {}
    errors_std = {}
    for k, v in errors.items():
        errors_mean[k] = np.mean(v)
        errors_std[k] = np.std(v)

    plt.figure(figsize=(10, 8))
    # plot original method error
    if original:
        original_errors: dict = torch.load(f"values/errors/{model_name}/original_{dim}.pt")
        plt.plot(list(original_errors.keys()), list(original_errors.values()), label="Original Method")
    # plot projection method error
    plt.plot(list(errors_mean.keys()), list(errors_mean.values()), label="Projection Method")

    plt.fill_between(
        list(errors_mean.keys()),
        np.array(list(errors_mean.values())) - np.array(list(errors_std.values())),
        np.array(list(errors_mean.values())) + np.array(list(errors_std.values())),
        alpha=0.2,
    )

    plt.xscale("log")
    xticks = list(errors.keys())

    plt.xticks(xticks, labels=[str(x) for x in xticks])

    plt.xlabel("Dimensionality")
    plt.ylabel("Maximum Absolute Error")
    plt.legend(loc="best", frameon=True, framealpha=0.6, ncol=2)
    plt.grid()
    if download:
        plt.savefig(f"{model_name}_MAE_dimension", dpi=300)


def load_files(model_name, target_value_type):
    try:
        values_path = f"values/{target_value_type}"
    except:
        print(f"{target_value_type} not found")
        return
    files = os.listdir(values_path)
    model_files = [f"values/{target_value_type}/{file}" for file in files if model_name in file]
    return model_files


def spearman_test(data_name, id, ood, ood_2, model_name):
    id_acc, _ = get_mean_std(data_name, id, model_name)
    ood_acc, _ = get_mean_std(data_name, ood, model_name)
    cor_1, p_1 = spearmanr(id_acc, ood_acc)

    if ood_2:
        ood_acc_2, _ = get_mean_std(data_name, ood_2, model_name)
        cor_2, p_2 = spearmanr(id_acc, ood_acc_2)

    result = {
        ood: {"correlation": cor_1, "p_value": p_1},
        ood_2: {"correlation": cor_2, "p_value": p_2} if ood_2 else None,
    }
    return result


def get_mean_std(data_name, specific_name, model_name, normalized=False):
    # set OOD or ID
    OOD = False if data_name == specific_name else True

    # add specific name to directory if OOD
    directory = f"values/{data_name}/{f'ood_acc/{specific_name}' if OOD else 'acc'}/{model_name}"
    files = os.listdir(directory)
    values = [torch.load(f"{directory}/{file}") for file in files]
    means = []
    stds = []
    new_values = []
    if normalized:
        for arg in values:
            new_args = []
            for val in arg:
                new_args.append(val / max(arg))
            new_values.append(new_args)
        # values = [val / max(arg) for val in arg for arg in values]
    else:
        new_values = values
    for i in range(len(values[0])):
        mean = np.mean([arg[i] for arg in new_values])
        std = np.std([arg[i] for arg in new_values])
        means.append(mean)
        stds.append(std)
    means = np.array(means)
    stds = np.array(stds)
    return means, stds


model_name_convert = {
    "resnet18": "ResNet18",
    "resnet34": "ResNet34",
    "resnet50": "ResNet50",
    "resnet50_swav": "ResNet50 Swav",
    "vit": "Vision Transformer",
    "vit_torch": "Vision Transformer",
    "mugs": "Mugs",
    "dino": "DINO",
    "mae": "MAE",
    "convnext": "ConvNext",
    "swin": "Swin Transformer",
    "dinov2": "DINOv2",
    "convnextv2": "ConvNextV2",
    "mlp": "MLP",
    "resnet34_imagenet100_32": "ResNet34 - Resolution 32",
    "resnet34_imagenet100_64": "ResNet34 - Resolution 64",
    "resnet34_imagenet100_128": "ResNet34 - Resolution 128",
    "resnet34_imagenet100_224": "ResNet34 - Resolution 224",
    "resnet34_imagenet100_224_wide": "ResNet34 - Resolution 224 - Wide",
    "resnet18_imagenet100_32": "ResNet18 - Resolution 32",
    "resnet18_imagenet100_64": "ResNet18 - Resolution 64",
    "resnet18_imagenet100_128": "ResNet18 - Resolution 128",
    "resnet18_imagenet100_224": "ResNet18 - Resolution 224",
    "resnet18_imagenet100_aug_32": "ResNet18(Augmentation) - Resolution 32",
    "resnet18_imagenet100_aug_64": "ResNet18(Augmentation) - Resolution 64",
    "resnet18_imagenet100_aug_128": "ResNet18(Augmentation) - Resolution 128",
    "resnet18_imagenet100_aug_224": "ResNet18(Augmentation) - Resolution 224",
    "resnet34_original_60": "ResNet34 - Original - 60 Epochs",
    "resnet34_original_90": "ResNet34 - Original - 90 Epochs",
    "resnet34_original_120": "ResNet34 - Original - 120 Epochs",
    "resnet34_original_150": "ResNet34 - Original - 150 Epochs",
    "resnet34_original_final": "ResNet34 - Original - Final",
    "vgg11_imagenet100_32": "VGG11_32",
    "vgg11_imagenet100_64": "VGG11_64",
    "vgg11_imagenet100_128": "VGG11_128",
    "vgg11_imagenet100_224": "VGG11_224",
    "vgg11_imagenet_class_10": "VGG11 - 10 Classes",
    "vgg11_imagenet_class_50": "VGG11 - 50 Classes",
    "vgg11_imagenet_class_100": "VGG11 - 100 Classes",
    "resnet18_imagenet100_no_residual_32": "ResNet18 - No Residual - 32",
    "resnet18_imagenet100_no_residual_64": "ResNet18 - No Residual - 64",
    "resnet18_imagenet100_no_residual_128": "ResNet18 - No Residual - 128",
    "resnet18_cifar10": "ResNet18 ",
    "avit_tiny_patch8_imagenet100_224": "AViT-Tiny - 224",
    "vit_tiny_patch8_imagenet100_64": "ViT-Tiny - 64",
}


data_name_convert = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "imagenet": "ImageNet",
    "imagenet100": "ImageNet",
    "places": "Places",
    "ninco": "NINCO",
}


def plot_resolution_ranks():
    sing_32 = torch.load("values/imagenet100/singular_values/imagenet100/vgg13_imagenet100_32.pt")
    sing_64 = torch.load("values/imagenet100/singular_values/imagenet100/vgg13_imagenet100_64.pt")
    sing_128 = torch.load("values/imagenet100/singular_values/imagenet100/vgg13_imagenet100_128.pt")
    rank_32 = get_rankme_ranks(sing_32)
    rank_64 = get_rankme_ranks(sing_64)
    rank_128 = get_rankme_ranks(sing_128)

    plt.figure(figsize=(10, 8))

    plt.plot(range(1, len(rank_32) + 1), rank_32, label=r"Resolution $32 \times 32$", color="#377EB8")
    plt.plot(range(1, len(rank_32) + 1), rank_64, label=r"Resolution $64 \times 64$", color="#FF7F00")
    plt.plot(range(1, len(rank_32) + 1), rank_128, label=r"Resolution $128 \times 128$", color="#4DAF4A")
    plt.xticks(range(1, len(rank_32) + 1), labels=[str(x) for x in range(1, len(rank_32) + 1)])
    plt.xlim(left=1, right=len(rank_32))

    plt.ylabel("Effective Rank", size=F, fontweight="bold")
    plt.xlabel("Layer", size=F, fontweight="bold")

    plt.legend(loc="best", frameon=True, framealpha=0.6)
    plt.tick_params(axis="both", which="major", labelsize=F)
    plt.tick_params(axis="both", which="minor", labelsize=F)
    plt.grid()

    plt.savefig("resolution_ranks.png", dpi=300)


def plot_NC1_all():
    # supervised models
    plt.figure(figsize=(10, 8))
    supervised_models = ["resnet50", "convnext", "vit", "swin"]

    sup_interp_n1 = []
    for model in supervised_models:
        n1 = torch.load(f"values/imagenet/NC1/{model}.pt")
        n1 = [val / n1[0] for val in n1]
        print(f"{model} : {n1[-1]}")
        normalized_layers = np.linspace(0, 1, len(n1))

        # Create an interpolation function
        interp_func = interp1d(normalized_layers, n1, kind="linear")

        # Interpolate at regular intervals
        interp_points = np.linspace(0, 1, 100)  # Change 100 to the desired resolution
        sup_interp_n1.append(interp_func(interp_points))
    sup_mean_n1 = np.mean(sup_interp_n1, axis=0)
    sup_std_n1 = np.std(sup_interp_n1, axis=0)
    print(f"Supervised mean : {sup_mean_n1[-1]}")
    print(f"Supervised std : {sup_std_n1[-1]}")
    plt.plot(interp_points, sup_mean_n1, label="ImageNet (Supervised)", color=ID_COLOR)
    plt.fill_between(
        interp_points, sup_mean_n1 - sup_std_n1, sup_mean_n1 + sup_std_n1, alpha=0.2, color=ID_COLOR
    )
    print()

    ssl_models = ["dino", "mae", "mugs", "resnet50_swav"]
    ssl_interp_n1 = []
    for model in ssl_models:
        n1 = torch.load(f"values/imagenet/NC1/{model}.pt")
        n1 = [val / n1[0] for val in n1]

        print(f"{model} : {n1[-1]}")
        normalized_layers = np.linspace(0, 1, len(n1))

        # Create an interpolation function
        interp_func = interp1d(normalized_layers, n1, kind="linear")

        # Interpolate at regular intervals
        interp_points = np.linspace(0, 1, 100)
        ssl_interp_n1.append(interp_func(interp_points))

    ssl_mean_n1 = np.mean(ssl_interp_n1, axis=0)
    ssl_std_n1 = np.std(ssl_interp_n1, axis=0)

    print(f"SSL mean : {ssl_mean_n1[-1]}")
    print(f"SSL std : {ssl_std_n1[-1]}")
    plt.plot(interp_points, ssl_mean_n1, label="ImageNet (Self-Supervised)", color=OOD_COLOR)
    plt.fill_between(
        interp_points,
        ssl_mean_n1 - ssl_std_n1,
        ssl_mean_n1 + ssl_std_n1,
        alpha=0.2,
        color=OOD_COLOR,
    )

    models = ["mlp", "resnet34"]
    for i in range(len(models)):
        model = models[i]
        n1 = torch.load(f"values/cifar10/NC1/{model}.pt")
        n1 = [val / n1[0] for val in n1]
        print(f"{model} : {n1[-1]}")
        normalized_layers = np.linspace(0, 1, len(n1))
        plt.plot(
            normalized_layers,
            n1,
            label=f"CIFAR-10 ({model_name_convert[model]})",
            color=CB_color_cycle[i + 5],
        )

    set_plot(plt, n1, xlabel="Relative Layer Depth", ylabel="Normalized NC1 Scores", normalize=True)

    plt.savefig("NC1_all_norm.png", dpi=300)

    # ssl


def plot_NC2_all():
    # supervised models
    plt.figure(figsize=(10, 8))
    models = ["resnet50", "convnext", "dino", "vit", "mae", "mugs", "resnet50_swav"]
    # models = ["convnext"]

    for model in models:
        n2 = torch.load(f"values/imagenet/NC2/{model}.pt")
        normalized_layers = np.linspace(0, 1, len(n2))

        # Create an interpolation function
        interp_func = interp1d(normalized_layers, n2, kind="linear")

        # Interpolate at regular intervals
        interp_points = np.linspace(0, 1, 100)  # Change 100 to the desired resolution
        plt.plot(interp_points, interp_func(interp_points), label=model_name_convert[model])
    set_plot(plt, n2, xlabel="Relative Layer Depth", ylabel="Feature Cosine Distance", normalize=True)
    plt.savefig("NC2_all_norm.png", dpi=300)


def plot_NC4_resolution():
    nc4_32 = torch.load("values/imagenet100/NC4/vgg13_imagenet100_32.pt")
    nc4_64 = torch.load("values/imagenet100/NC4/vgg13_imagenet100_64.pt")
    nc4_128 = torch.load("values/imagenet100/NC4/vgg13_imagenet100_128.pt")
    nc4_224 = torch.load("values/imagenet100/NC4/vgg13_imagenet100_224.pt")

    nc4_32 = [x * 100 for x in nc4_32]
    nc4_64 = [x * 100 for x in nc4_64]
    nc4_128 = [x * 100 for x in nc4_128]
    nc4_224 = [x * 100 for x in nc4_224]

    plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(nc4_32) + 1), nc4_32, label=r"Resolution $32 \times 32$", color="#377EB8")
    plt.plot(range(1, len(nc4_32) + 1), nc4_64, label=r"Resolution $64 \times 64$", color="#FF7F00")
    plt.plot(range(1, len(nc4_32) + 1), nc4_128, label=r"Resolution $128 \times 128$", color="#4DAF4A")
    plt.plot(range(1, len(nc4_32) + 1), nc4_224, label=r"Resolution $224 \times 224$", color="#984EA3")

    plt.xticks(range(1, len(nc4_32) + 1), labels=[str(x) for x in range(1, len(nc4_32) + 1)])
    plt.ylabel("Prediction Agreement [%]", size=F, fontweight="bold")
    plt.xlabel("Layer", size=F, fontweight="bold")
    plt.grid()
    plt.xlim(left=1, right=len(nc4_32))

    plt.legend(loc="best", frameon=True, framealpha=0.6)
    plt.tick_params(axis="both", which="major", labelsize=F)
    plt.tick_params(axis="both", which="minor", labelsize=F)
    plt.savefig("NC4_resolution.png", dpi=300)


def plot_NC1(pretrained_data, model_name):
    nc1 = torch.load(f"values/{pretrained_data}/NC1/{model_name}.pt")
    save_path = f"plots/{pretrained_data}/NC1"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.figure(figsize=(10, 8))
    plt.title(f"{model_name_convert[model_name]} - {pretrained_data}")
    plt.plot(range(1, len(nc1) + 1), nc1, color=ID_COLOR)
    set_plot(plt, nc1, xlabel="Layer", ylabel="NC1")
    plt.ylim(bottom=0)

    plt.savefig(f"{save_path}/{model_name}", dpi=300)


def plot_NC4_train_test(pretrained_data, model_name):
    train_nc4 = torch.load(f"values/{pretrained_data}/NC4/{model_name}.pt")
    test_nc4 = torch.load(f"values/{pretrained_data}/NC4_test/{model_name}.pt")
    save_path = f"plots/{pretrained_data}/NC4_train_test"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_nc4) + 1), train_nc4, label="Train", color=ID_COLOR)
    plt.plot(range(1, len(test_nc4) + 1), test_nc4, label="Test", color=OOD_COLOR)
    set_plot(plt, train_nc4, xlabel="Layer", ylabel="Prediction Agreement [%]")
    plt.title(f"{model_name_convert[model_name]} - {pretrained_data}")
    plt.savefig(f"{save_path}/{model_name}", dpi=300)


def plot_NC_all(pretrained_data, NC_type):
    save_dir = f"plots/{pretrained_data}/{NC_type}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(10, 8))
    if pretrained_data == "imagenet":
        model_names = [
            "resnet50",
            "resnet50_swav",
            "vit",
            "mugs",
            "dino",
            "mae",
            "convnext",
            "swin",
        ]
        labels = model_names
    elif pretrained_data == "imagenet100":
        model_names = [
            "vgg13_imagenet100_32",
            "vgg13_imagenet100_64",
            "vgg13_imagenet100_128",
            "vgg13_imagenet100_224",
        ]
        labels = [
            r"Resolution $32 \times 32$",
            r"Resolution $64 \times 64$",
            r"Resolution $128 \times 128$",
            r"Resolution $224 \times 224$",
        ]
    elif pretrained_data == "cifar10":
        model_names = ["resnet18", "resnet34"]
        labels = model_names

    for i in range(len(model_names)):
        nc = torch.load(f"values/{pretrained_data}/{NC_type}/{model_names[i]}.pt")
        if NC_type == "NC1":
            nc = [val / nc[0] for val in nc]
            print(f"{model_names[i]} : {nc[-1]}")
        plt.plot(range(1, len(nc) + 1), nc, label=labels[i], color=CB_color_cycle[i])
    plt.savefig(f"{save_dir}/all.png", dpi=300)


def plot_NC1_resolution():
    plt.figure(figsize=(10, 8))
    nc_1_32 = torch.load("values/imagenet100/NC1/vgg13_imagenet100_32.pt")
    nc_1_64 = torch.load("values/imagenet100/NC1/vgg13_imagenet100_64.pt")
    nc_1_128 = torch.load("values/imagenet100/NC1/vgg13_imagenet100_128.pt")
    nc_1_224 = torch.load("values/imagenet100/NC1/vgg13_imagenet100_224.pt")

    plt.plot(range(1, len(nc_1_224) + 1), nc_1_32, label=r"Resolution $32 \times 32$", color="#377EB8")
    plt.plot(range(1, len(nc_1_224) + 1), nc_1_64, label=r"Resolution $64 \times 64$", color="#FF7F00")
    plt.plot(range(1, len(nc_1_224) + 1), nc_1_128, label=r"Resolution $128 \times 128$", color="#4DAF4A")
    plt.plot(range(1, len(nc_1_224) + 1), nc_1_224, label=r"Resolution $224 \times 224$", color="#984EA3")

    set_plot(plt, nc_1_224, xlabel="Layer", ylabel="NC1")

    plt.savefig("NC1_resolution.png", dpi=300)


def plot_NC2(pretrained_data, model_name):
    nc2 = torch.load(f"values/{pretrained_data}/NC2/{model_name}.pt")
    save_path = f"plots/{pretrained_data}/NC2"
    if pretrained_data == "imagenet100":
        C = 100
    elif pretrained_data == "imagenet":
        C = 1000
    elif pretrained_data == "cifar10":
        C = 10

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    new_nc2 = []
    for val in nc2:
        new_val = (1 / (C - 1)) + val
        new_nc2.append(new_val)

    plt.figure(figsize=(10, 8))
    plt.title(f"{model_name_convert[model_name]} - {pretrained_data}")
    plt.plot(range(1, len(nc2) + 1), new_nc2, color=ID_COLOR)
    set_plot(plt, nc2, xlabel="Layer", ylabel="ETF")

    plt.savefig(f"{save_path}/{model_name}", dpi=300)


def process_nc2(nc2, C):
    new_nc2 = []
    for val in nc2:
        new_val = (1 / (C - 1)) + val
        new_nc2.append(new_val)
    return new_nc2


def plot_NC_by_dataset_models(dataset, nc_type):
    base_dir = f"values/{dataset}/{nc_type}"
    if nc_type == "NC1":
        label = "Normalized NC1 Scores"
    elif nc_type == "NC2":
        label = "ETF"
    elif nc_type == "NC4":
        label = "Prediction Agreement [%]"

    if dataset == "imagenet100":
        models = [
            "vgg13_imagenet100_32",
            "vgg13_imagenet100_64",
            "vgg13_imagenet100_128",
            "vgg13_imagenet100_224",
        ]
        legends = [
            r"Resolution $32 \times 32$",
            r"Resolution $64 \times 64$",
            r"Resolution $128 \times 128$",
            r"Resolution $224 \times 224$",
        ]

        normalized = False
        C = 100
    elif dataset == "imagenet":
        models = ["resnet50", "vit", "convnext", "swin", "dino", "mae", "mugs", "resnet50_swav"]
        legends = models
        C = 1000

    elif dataset == "cifar10":
        C = 10
        models = ["resnet34", "mlp"]
        legends = models

    plt.figure(figsize=(10, 8))
    for i in range(len(models)):
        model = models[i]
        nc = torch.load(f"{base_dir}/{model}.pt")
        if nc_type == "NC2":
            nc = process_nc2(nc, C)
        elif nc_type == "NC1":
            nc = [x / nc[0] for x in nc]
        elif nc_type == "NC4":
            nc = [x * nc[0] for x in nc]
        layer = range(1, len(nc) + 1)
        if normalized:
            layer = np.linspace(0, 1, len(nc))
        plt.plot(layer, nc, label=legends[i], color=CB_color_cycle[i])
        # plt.plot(range(1, len(nc4) + 1), nc4, label=model_name_convert[model])

    set_plot(plt, nc, xlabel="Layer", ylabel=label, normalize=normalized)
    plt.savefig(f"{nc_type}_{dataset}.png", dpi=300)


def plot_resolution_NC2():
    nc_2_32 = torch.load("values/imagenet100/NC2/vgg13_imagenet100_32.pt")
    nc_2_64 = torch.load("values/imagenet100/NC2/vgg13_imagenet100_64.pt")
    nc_2_128 = torch.load("values/imagenet100/NC2/vgg13_imagenet100_128.pt")
    nc_2_224 = torch.load("values/imagenet100/NC2/vgg13_imagenet100_224.pt")
    C = 100
    new_nc2_32 = []
    for val in nc_2_32:
        new_val = (1 / (C - 1)) + val
        new_nc2_32.append(new_val)

    new_nc2_64 = []
    for val in nc_2_64:
        new_val = (1 / (C - 1)) + val
        new_nc2_64.append(new_val)

    new_nc2_128 = []
    for val in nc_2_128:
        new_val = (1 / (C - 1)) + val
        new_nc2_128.append(new_val)

    new_nc2_224 = []
    for val in nc_2_224:
        new_val = (1 / (C - 1)) + val
        new_nc2_224.append(new_val)

    plt.figure(figsize=(10, 8))
    plt.plot(
        range(1, len(new_nc2_32) + 1),
        new_nc2_32,
        label=r"Resolution $32 \times 32$",
        color="#377EB8",
    )
    plt.plot(
        range(1, len(new_nc2_64) + 1),
        new_nc2_64,
        label=r"Resolution $64 \times 64$",
        color="#FF7F00",
    )
    plt.plot(
        range(1, len(new_nc2_128) + 1),
        new_nc2_128,
        label=r"Resolution $128 \times 128$",
        color="#4DAF4A",
    )
    plt.plot(
        range(1, len(new_nc2_224) + 1),
        new_nc2_224,
        label=r"Resolution $224 \times 224$",
        color="#984EA3",
    )

    set_plot(plt, nc_2_32, xlabel="Layer", ylabel="ETF")
    plt.savefig("NC2_resolution.png", dpi=300)


def plot_ranks_vgg11_different_sample_size():
    rank_100 = get_rankme_ranks(
        torch.load(f"values/imagenet/singular_values/imagenet/vgg13_imagenet_class_100.pt")
    )
    rank_200 = get_rankme_ranks(
        torch.load(f"values/imagenet/singular_values/imagenet/vgg11_imagenet_samples_200.pt")
    )
    rank_500 = get_rankme_ranks(
        torch.load(f"values/imagenet/singular_values/imagenet/vgg11_imagenet_samples_500.pt")
    )
    rank_1000 = get_rankme_ranks(
        torch.load(f"values/imagenet/singular_values/imagenet/vgg11_imagenet_samples_1000.pt")
    )

    plt.figure(figsize=(10, 8))
    plt.plot(
        range(1, len(rank_100) + 1),
        rank_100,
        label="100 samples per class",
        color="#377EB8",
    )
    plt.plot(
        range(1, len(rank_100) + 1),
        rank_200,
        label="200 samples per class",
        color="#FF7F00",
    )
    plt.plot(
        range(1, len(rank_100) + 1),
        rank_500,
        label="500 samples per class",
        color="#4DAF4A",
    )
    plt.plot(
        range(1, len(rank_100) + 1),
        rank_1000,
        label="1000 samples per class",
        color="#984EA3",
    )

    set_plot(plt, rank_100, xlabel="Layer", ylabel="Effective Rank")
    plt.savefig("vgg11_rank_different_samples.png", dpi=300)


def set_plot(plt, input_data, xlabel, ylabel, normalize=False):
    if not normalize:
        if len(input_data) > 40:
            plt.xticks(
                range(1, len(input_data) + 1, 3),
                labels=[str(x) for x in range(1, len(input_data) + 1, 3)],
            )
        elif len(input_data) > 20:
            plt.xticks(
                range(1, len(input_data) + 1, 2),
                labels=[str(x) for x in range(1, len(input_data) + 1, 2)],
            )
        else:
            plt.xticks(
                range(1, len(input_data) + 1),
                labels=[str(x) for x in range(1, len(input_data) + 1)],
            )

    plt.ylabel(ylabel, size=F, fontweight="bold")
    plt.xlabel(xlabel, size=F, fontweight="bold")
    plt.grid()
    if normalize:
        plt.xlim(left=0, right=1)
    else:
        plt.xlim(left=1, right=len(input_data))

    plt.legend(loc="best", frameon=True, framealpha=0.6)
    plt.tick_params(axis="both", which="major", labelsize=F)
    plt.tick_params(axis="both", which="minor", labelsize=F)


def plot_first_figure():
    ID_data_name = "imagenet100"
    save_dir = f"plots/{ID_data_name}/acc_rank"

    model_name_1 = "vgg13_imagenet100_32"
    model_name_2 = "vgg13_imagenet100_128"
    OOD_data_name = "ninco"

    ood_means, ood_stds = get_mean_std(ID_data_name, OOD_data_name, model_name_1)
    id_means, id_stds = get_mean_std(ID_data_name, ID_data_name, model_name_1)

    ood_means_2, ood_stds_2 = get_mean_std(ID_data_name, OOD_data_name, model_name_2)
    id_means_2, id_stds_2 = get_mean_std(ID_data_name, ID_data_name, model_name_2)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ID_data_name = data_name_convert[ID_data_name]
    OOD_data_name = data_name_convert[OOD_data_name]
    if len(ood_means) > 40:
        plt.xticks(
            range(1, len(ood_means) + 1, 3),
            labels=[str(x) for x in range(1, len(ood_means) + 1, 3)],
        )
    elif len(ood_means) > 20:
        plt.xticks(
            range(1, len(ood_means) + 1, 2),
            labels=[str(x) for x in range(1, len(ood_means) + 1, 2)],
        )
    else:
        plt.xticks(range(1, len(ood_means) + 1), labels=[str(x) for x in range(1, len(ood_means) + 1)])
    # plot ID
    ax1.plot(range(1, len(id_means) + 1), id_means, label=f"ID ({ID_data_name} - 32x32)", color=ID_COLOR)
    ax1.fill_between(
        range(1, len(id_means) + 1),
        id_means - id_stds,
        id_means + id_stds,
        alpha=0.2,
        color=ID_COLOR,
    )

    # plot OOD1
    ax1.plot(
        range(1, len(ood_means) + 1),
        ood_means,
        label=f"OOD ({OOD_data_name} - 32x32)",
        color=OOD_2_COLOR,
    )
    ax1.fill_between(
        range(1, len(ood_means) + 1),
        ood_means - ood_stds,
        ood_means + ood_stds,
        alpha=0.2,
        color=OOD_2_COLOR,
    )

    # plot ID2
    ax1.plot(
        range(1, len(id_means_2) + 1),
        id_means_2,
        linestyle="dashed",
        label=f"ID ({ID_data_name} - 128x128)",
        color=ID_COLOR,
    )
    ax1.fill_between(
        range(1, len(id_means) + 1),
        id_means_2 - id_stds_2,
        id_means_2 + id_stds_2,
        alpha=0.2,
        color=ID_COLOR,
    )

    # plot OOD2
    ax1.plot(
        range(1, len(ood_means_2) + 1),
        ood_means_2,
        linestyle="dashed",
        label=f"OOD ({OOD_data_name} - 128x128)",
        color=OOD_2_COLOR,
    )
    ax1.fill_between(
        range(1, len(ood_means_2) + 1),
        ood_means_2 - ood_stds_2,
        ood_means_2 + ood_stds_2,
        alpha=0.2,
        color=OOD_2_COLOR,
    )

    plt.grid()
    ax1.set_ylabel("Top-1 Accuracy [%]", size=F, fontweight="bold")
    plt.xlabel(f"Layer", size=F, fontweight="bold")

    plt.legend(loc="upper left", frameon=True, framealpha=0.6, ncol=2)
    plt.xlim(left=1, right=len(ood_means_2))

    ax1.set_ylim(bottom=0, top=80)

    plt.tick_params(axis="both", which="major", labelsize=F)
    plt.tick_params(axis="both", which="minor", labelsize=F)
    # 0 from yticks
    # yticks = list(plt.yticks()[0])[1:-1]
    # plt.yticks(yticks)
    # I want y ticks to be 0, 10, 20, 30, 40, 50, 60, 70, 80 ... current
    # plt.yticks(yticks)
    all_acc = np.concatenate((id_means, ood_means, ood_means_2))
    plt.yticks(np.arange(10, max(all_acc) + 10, 10))

    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))

    plt.savefig(f"{save_dir}/dimension_comparision.png", dpi=300)


def rankme(s, epsilon=1e-7):
    p_k = (s / (torch.norm(s, 1))) + epsilon

    entropy = -torch.sum(p_k * torch.log(p_k))

    rankme_value = torch.exp(entropy)

    return rankme_value.item()


def get_rankme_ranks(sigs):
    ranks = []
    for sig in sigs:
        rank = rankme(sig)
        ranks.append(rank)
    return ranks


def get_original_ranks(sigs):
    ranks = []
    for sig in sigs:
        threshold = torch.max(sig) * 0.001
        rank = (sig > threshold).sum().item()
        ranks.append(rank)
    return ranks


def plot_change_ranks(model_name, pretrained_data):
    id_path = f"values/{pretrained_data}/singular_values/{pretrained_data}/{model_name}.pt"

    arch_name = model_name.split("_")[0]
    random_path = f"values/{pretrained_data}/singular_values/{pretrained_data}/{arch_name}_random_init.pt"

    id_sigs = torch.load(id_path)

    random_sigs = torch.load(random_path)

    print(id_sigs[1][:3], random_sigs[1][:3])
    id_ranks = np.array(get_rankme_ranks(id_sigs))
    random_ranks = np.array(get_rankme_ranks(random_sigs))

    percentage = (id_ranks - random_ranks) * 100 / random_ranks
    plt.figure(figsize=(8, 6))
    plt.plot(percentage)
    plt.ylabel("Change in Rank [%]")
    plt.xlabel("Layer")
    plt.grid(axis="both")
    plt.savefig(f"{model_name}_{pretrained_data}_change_in_rank.png", dpi=300)


def plot_random_ranks(model_name, pretrained_data):
    arch_name = model_name.split("_")[0]
    random_path = f"values/{pretrained_data}/singular_values/{pretrained_data}/{arch_name}_random_init.pt"

    random_sigs = torch.load(random_path)

    random_ranks = np.array(get_rankme_ranks(random_sigs))

    plt.figure(figsize=(6, 4))
    plt.plot(random_ranks)
    plt.title(f"{model_name_convert[model_name]} - random init")
    plt.ylabel("Effective Rank")
    plt.xlabel("Layers")
    plt.grid()
    plt.savefig(f"{model_name}_{pretrained_data}_random_ranks.png", dpi=300)


def original_rank(sigs):
    ranks = []
    for sig in sigs:
        threshold = torch.max(sig) * 0.001
        rank = (sig > threshold).sum().item()
        ranks.append(rank)
    return ranks


def plot_NC(model_name, pretrained_data, nc_type):
    nc = torch.load(f"values/{pretrained_data}/{nc_type}/{model_name}.pt")
    base_dir = f"plots/{pretrained_data}/{nc_type}"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    plt.figure(figsize=(10, 8))
    if len(nc) > 40:
        plt.xticks(
            range(1, len(nc) + 1, 3),
            labels=[str(x) for x in range(1, len(nc) + 1, 3)],
        )
    elif len(nc) > 20:
        plt.xticks(
            range(1, len(nc) + 1, 2),
            labels=[str(x) for x in range(1, len(nc) + 1, 2)],
        )
    else:
        plt.xticks(range(1, len(nc) + 1), labels=[str(x) for x in range(1, len(nc) + 1)])

    plt.plot(range(1, len(nc) + 1), nc, color=ID_COLOR)
    plt.title(f"{model_name_convert[model_name]} - {pretrained_data}")

    plt.ylim(bottom=0)
    plt.xlim(left=1, right=len(nc))
    plt.xlabel("Layer", size=F, fontweight="bold")
    plt.ylabel("Prediction Agreement", size=F, fontweight="bold")
    plt.grid()
    # plt.yticks(np.arange(0.1, max(nc), 0.1))

    plt.tick_params(axis="both", which="major", labelsize=F)
    plt.tick_params(axis="both", which="minor", labelsize=F)
    plt.savefig(f"{base_dir}/{model_name}.png", dpi=300)


def plot_compare_aug(model_name: str):
    ID_data_name = "imagenet100"
    OOD_data_name = "places"
    OOD_data_name2 = "ninco"
    save_dir = f"plots/{ID_data_name}/acc_rank"
    two_ood = False
    if OOD_data_name2 is not None:
        two_ood = True

    aug_model_name = f"resnet18_imagenet100_aug_{model_name.split('_')[-1]}"

    ood_means, ood_stds = get_mean_std(ID_data_name, OOD_data_name, model_name)
    if two_ood:
        ood_means_2, ood_stds_2 = get_mean_std(ID_data_name, OOD_data_name2, model_name)

    id_means, id_stds = get_mean_std(ID_data_name, ID_data_name, model_name)

    aug_ood_means, aug_ood_stds = get_mean_std(ID_data_name, OOD_data_name, aug_model_name)
    if two_ood:
        aug_ood_means_2, aug_ood_stds_2 = get_mean_std(ID_data_name, OOD_data_name2, aug_model_name)
    aug_id_means, aug_id_stds = get_mean_std(ID_data_name, ID_data_name, aug_model_name)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    # label font size
    if len(ood_means) > 40:
        plt.xticks(
            range(1, len(ood_means) + 1, 3),
            labels=[str(x) for x in range(1, len(ood_means) + 1, 3)],
        )
    elif len(ood_means) > 20:
        plt.xticks(
            range(1, len(ood_means) + 1, 2),
            labels=[str(x) for x in range(1, len(ood_means) + 1, 2)],
        )
    else:
        plt.xticks(range(1, len(ood_means) + 1), labels=[str(x) for x in range(1, len(ood_means) + 1)])

    ID_data_name = data_name_convert[ID_data_name]
    OOD_data_name = data_name_convert[OOD_data_name]
    if two_ood:
        OOD_data_name2 = data_name_convert[OOD_data_name2]

    # plot ID
    ax1.plot(range(1, len(id_means) + 1), id_means, label=f"ID ({ID_data_name})", color=ID_COLOR)
    ax1.fill_between(
        range(1, len(id_means) + 1),
        id_means - id_stds,
        id_means + id_stds,
        alpha=0.2,
        color=ID_COLOR,
    )

    ax1.plot(
        range(1, len(aug_id_means) + 1),
        aug_id_means,
        label=f"ID ({ID_data_name} - Aug)",
        color=ID_COLOR,
        linestyle="dotted",
    )
    ax1.fill_between(
        range(1, len(aug_id_means) + 1),
        aug_id_means - aug_id_stds,
        aug_id_means + aug_id_stds,
        alpha=0.2,
        color=ID_COLOR,
    )

    # plot OOD1
    ax1.plot(range(1, len(ood_means) + 1), ood_means, label=f"OOD ({OOD_data_name})", color=OOD_COLOR)
    ax1.fill_between(
        range(1, len(ood_means) + 1),
        ood_means - ood_stds,
        ood_means + ood_stds,
        alpha=0.2,
        color=OOD_COLOR,
    )

    ax1.plot(
        range(1, len(aug_ood_means) + 1),
        aug_ood_means,
        label=f"OOD ({OOD_data_name} - Aug)",
        color=OOD_COLOR,
        linestyle="dotted",
    )
    ax1.fill_between(
        range(1, len(aug_ood_means) + 1),
        aug_ood_means - aug_ood_stds,
        aug_ood_means + aug_ood_stds,
        alpha=0.2,
        color=OOD_COLOR,
    )

    # plot OOD2
    if two_ood:
        ax1.plot(
            range(1, len(ood_means_2) + 1),
            ood_means_2,
            label=f"OOD ({OOD_data_name2})",
            color=OOD_2_COLOR,
        )
        ax1.fill_between(
            range(1, len(ood_means_2) + 1),
            ood_means_2 - ood_stds_2,
            ood_means_2 + ood_stds_2,
            alpha=0.2,
            color=OOD_2_COLOR,
        )

        ax1.plot(
            range(1, len(aug_ood_means_2) + 1),
            aug_ood_means_2,
            label=f"OOD ({OOD_data_name2} - Aug)",
            color=OOD_2_COLOR,
            linestyle="dotted",
        )
        ax1.fill_between(
            range(1, len(aug_ood_means_2) + 1),
            aug_ood_means_2 - aug_ood_stds_2,
            aug_ood_means_2 + aug_ood_stds_2,
            alpha=0.2,
            color=OOD_2_COLOR,
        )

    # Add best accuracy to y-ticks
    # yticks = list(plt.yticks()[0])
    # yticks.append(max(id_means))
    # yticks = sorted(yticks)
    # for i in range(len(yticks) - 1):
    #     if abs(yticks[i] - yticks[i + 1]) < 2:
    #         yticks[i] = 0

    # plt.yticks(list(plt.yticks()[0]) + [max(id_means)])
    # plt.yticks(list(plt.yticks()[0]) + [max(id_means)])
    plt.grid()

    ax1.set_ylabel("Top-1 Accuracy [%]", size=F, fontweight="bold")
    plt.xlabel(f"Layer", size=F, fontweight="bold")

    plt.legend(loc="upper left", frameon=True, framealpha=0.6)
    plt.xlim(left=1, right=len(ood_means))

    ax1.set_ylim(bottom=0)
    plt.tick_params(axis="both", which="major", labelsize=F)
    plt.tick_params(axis="both", which="minor", labelsize=F)

    if two_ood:
        all_acc = np.concatenate((id_means, ood_means, ood_means_2))
    else:
        all_acc = np.concatenate((id_means, ood_means))
    plt.yticks(np.arange(10, max(all_acc), 10))

    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))
    add_title = True
    normalize = False

    file_name = f"{model_name}_aug_compare.png"
    if add_title:
        plt.title(f"{model_name_convert[model_name]} - {ID_data_name}")
    plt.savefig(f"{save_dir}/{file_name}", dpi=300)


def plot_acc(
    ID_data_name,
    OOD_data_name,
    OOD_data_name2,
    model_name,
    tunnel_start=None,
    add_rank=False,
    add_title=False,
    normalize=False,
):
    save_dir = f"plots/{ID_data_name}/acc_rank"
    two_ood = False
    if OOD_data_name2 is not None:
        two_ood = True

    ood_means, ood_stds = get_mean_std(ID_data_name, OOD_data_name, model_name)
    if two_ood:
        ood_means_2, ood_stds_2 = get_mean_std(ID_data_name, OOD_data_name2, model_name)

    id_means, id_stds = get_mean_std(ID_data_name, ID_data_name, model_name)

    if add_rank:
        singular_values = torch.load(f"values/{ID_data_name}/singular_values/{ID_data_name}/{model_name}.pt")
        ranks = get_rankme_ranks(singular_values)
        # ranks = get_original_ranks(singular_values)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    # label font size
    if len(ood_means) > 40:
        plt.xticks(
            range(1, len(ood_means) + 1, 3),
            labels=[str(x) for x in range(1, len(ood_means) + 1, 3)],
        )
    elif len(ood_means) > 20:
        plt.xticks(
            range(1, len(ood_means) + 1, 2),
            labels=[str(x) for x in range(1, len(ood_means) + 1, 2)],
        )
    else:
        plt.xticks(range(1, len(ood_means) + 1), labels=[str(x) for x in range(1, len(ood_means) + 1)])

    ID_data_name = data_name_convert[ID_data_name]
    OOD_data_name = data_name_convert[OOD_data_name]
    if two_ood:
        OOD_data_name2 = data_name_convert[OOD_data_name2]

    # plot ID
    ax1.plot(range(1, len(id_means) + 1), id_means, label=f"ID ({ID_data_name})", color=ID_COLOR)
    ax1.fill_between(
        range(1, len(id_means) + 1),
        id_means - id_stds,
        id_means + id_stds,
        alpha=0.2,
        color=ID_COLOR,
    )

    # plot OOD1
    ax1.plot(range(1, len(ood_means) + 1), ood_means, label=f"OOD ({OOD_data_name})", color=OOD_COLOR)
    ax1.fill_between(
        range(1, len(ood_means) + 1),
        ood_means - ood_stds,
        ood_means + ood_stds,
        alpha=0.2,
        color=OOD_COLOR,
    )

    # plot OOD2
    if two_ood:
        ax1.plot(
            range(1, len(ood_means_2) + 1),
            ood_means_2,
            label=f"OOD ({OOD_data_name2})",
            color=OOD_2_COLOR,
        )
        ax1.fill_between(
            range(1, len(ood_means_2) + 1),
            ood_means_2 - ood_stds_2,
            ood_means_2 + ood_stds_2,
            alpha=0.2,
            color=OOD_2_COLOR,
        )

    # Add best accuracy to y-ticks
    # yticks = list(plt.yticks()[0])
    # yticks.append(max(id_means))
    # yticks = sorted(yticks)
    # for i in range(len(yticks) - 1):
    #     if abs(yticks[i] - yticks[i + 1]) < 2:
    #         yticks[i] = 0

    # plt.yticks(list(plt.yticks()[0]) + [max(id_means)])
    # plt.yticks(list(plt.yticks()[0]) + [max(id_means)])
    plt.grid()
    # if tunnel_start is not None:
    #     plt.axvspan(tunnel_start, len(ood_means), alpha=0.2, color="#4DAF4A")
    ax1.set_ylabel("Top-1 Accuracy [%]", size=F, fontweight="bold")
    plt.xlabel(f"Layer", size=F, fontweight="bold")
    if add_rank:
        ax2 = ax1.twinx()
        ax2.plot(range(1, len(ranks) + 1), ranks, label="Rank", color="#E41A1C", linestyle="dashed")
        ax2.set_ylabel("Effective Rank")
        ax2.set_ylim(bottom=0)
        lines = ax1.get_lines() + ax2.get_lines()
        labels = [line.get_label() for line in lines]
        plt.legend(lines, labels, loc="upper left", frameon=True, framealpha=0.6)

    else:
        plt.legend(loc="upper left", frameon=True, framealpha=0.6)
    plt.xlim(left=1, right=len(ood_means))

    ax1.set_ylim(bottom=0)
    plt.tick_params(axis="both", which="major", labelsize=F)
    plt.tick_params(axis="both", which="minor", labelsize=F)
    # 0 from yticks
    # yticks = list(plt.yticks()[0])[1:-1]
    # plt.yticks(yticks)
    # I want y ticks to be 0, 10, 20, 30, 40, 50, 60, 70, 80 ... current
    # plt.yticks(yticks)
    if two_ood:
        all_acc = np.concatenate((id_means, ood_means, ood_means_2))
    else:
        all_acc = np.concatenate((id_means, ood_means))
    plt.yticks(np.arange(10, max(all_acc), 10))

    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))

    file_name = f"{model_name}{'_title' if add_title else ''}{'_normalized' if normalize else ''}.png"
    if add_title:
        plt.title(f"{model_name_convert[model_name]} - {ID_data_name}")
    plt.savefig(f"{save_dir}/{file_name}", dpi=300)


def plot_acc_ninco(model_name: str):
    ninco_means, ninco_stds = get_mean_std("imagenet100", "ninco", model_name)
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(ninco_means) + 1), ninco_means, color=ID_COLOR)
    plt.fill_between(
        range(1, len(ninco_means) + 1),
        ninco_means - ninco_stds,
        ninco_means + ninco_stds,
        alpha=0.2,
        color=ID_COLOR,
    )

    plt.title(f"model name - ninco")
    plt.ylabel("Top-1 Accuracy [%]")
    plt.xlabel("Layer")
    plt.grid()
    plt.savefig(f"{model_name}_ninco.png", dpi=300)


def compute_correlation(ID_data_name, OOD_data_name, model_name):
    ood_means, ood_stds = get_mean_std(ID_data_name, OOD_data_name, model_name)
    id_means, id_stds = get_mean_std(ID_data_name, ID_data_name, model_name)
    correlation = np.corrcoef(ood_means, id_means)[0, 1]
    return correlation


def plot_GAP_full_comparision():
    ID_DATA = "cifar10"
    OOD_DATA = "cifar100"
    m, s = get_mean_std(ID_DATA, ID_DATA, "resnet34")
    o_m, o_s = get_mean_std(ID_DATA, OOD_DATA, "resnet34")
    plt.figure(figsize=(10, 8))
    plt.plot(
        range(1, len(m) + 1),
        m,
        label=f"ID ({data_name_convert[ID_DATA]}) - GAP",
        color=CB_color_cycle[0],
    )
    plt.fill_between(range(1, len(m) + 1), m - s, m + s, alpha=0.2, color=CB_color_cycle[0])

    plt.plot(
        range(1, len(o_m) + 1),
        o_m,
        label=f"OOD ({data_name_convert[OOD_DATA]}) - GAP",
        color=CB_color_cycle[1],
    )
    plt.fill_between(range(1, len(o_m) + 1), o_m - o_s, o_m + o_s, alpha=0.2, color=CB_color_cycle[1])

    m, s = get_mean_std(ID_DATA, ID_DATA, "resnet34_xx")
    o_m, o_s = get_mean_std(ID_DATA, OOD_DATA, "resnet34_xx")

    plt.plot(
        range(1, len(m) + 1),
        m,
        color=CB_color_cycle[2],
        linestyle="dashed",
        label=f"ID ({data_name_convert[ID_DATA]}) - Full Embedding",
    )
    plt.fill_between(range(1, len(m) + 1), m - s, m + s, alpha=0.2, color=CB_color_cycle[2])

    plt.plot(
        range(1, len(o_m) + 1),
        o_m,
        color=CB_color_cycle[3],
        linestyle="dashed",
        label=f"OOD ({data_name_convert[OOD_DATA]}) - Full Embedding",
    )
    plt.fill_between(range(1, len(o_m) + 1), o_m - o_s, o_m + o_s, alpha=0.2, color=CB_color_cycle[3])
    set_plot(plt, m, xlabel="Layer", ylabel="Top-1 Accuracy [%]", normalize=False)
    plt.savefig("GAP_full_comparision.png", dpi=300)


def plot_GFlops():
    GFlops = [0.229314048, 0.917101056, 3.668249088, 11.233906176]
    resolutions = [r"$32 \times 32$", r"$64 \times 64$", r"$128 \times 128$", r"$224 \times 224$"]
    plt.figure(figsize=(10, 8))
    plt.bar(resolutions, GFlops, color=ID_COLOR, alpha=0.7)
    plt.plot(resolutions, GFlops, color=ID_COLOR, alpha=0.7)

    plt.ylabel("GFLOPS", size=F, fontweight="bold")
    plt.xlabel("Image Resolution", size=F, fontweight="bold")
    plt.tick_params(axis="both", which="major", labelsize=F)
    plt.tick_params(axis="both", which="minor", labelsize=F)
    plt.grid()
    plt.savefig("GFlops.png", dpi=300)


def plot_different_class_ninco():
    ID_data_name = "imagenet"
    OOD_data_name = "ninco"
    save_dir = f"plots/{ID_data_name}/acc_rank"
    vgg11_10_name = "vgg11_imagenet_class_10"
    vgg11_50_name = "vgg11_imagenet_class_50"
    vgg11_100_name = "vgg11_imagenet_samples_100"

    means_10, std_10 = get_mean_std(ID_data_name, OOD_data_name, vgg11_10_name, True)
    means_50, std_50 = get_mean_std(ID_data_name, OOD_data_name, vgg11_50_name, True)
    means_100, std_100 = get_mean_std(ID_data_name, OOD_data_name, vgg11_100_name, True)
    # means_10 = np.array([x / max(means_10) for x in means_10])
    # means_50 = np.array([x / max(means_50) for x in means_50])
    # means_100 = np.array([x / max(means_100) for x in means_100])

    plt.figure(figsize=(10, 8))
    plt.plot(
        range(1, len(means_10) + 1),
        means_10,
        label=f"10 classes",
        color=CB_color_cycle[0],
    )
    plt.fill_between(
        range(1, len(means_10) + 1),
        means_10 - std_10,
        means_10 + std_10,
        alpha=0.2,
        color=CB_color_cycle[0],
    )
    plt.plot(
        range(1, len(means_10) + 1),
        means_50,
        label=f"50 classes",
        color=CB_color_cycle[1],
    )
    plt.fill_between(
        range(1, len(means_10) + 1),
        means_50 - std_50,
        means_50 + std_50,
        alpha=0.2,
        color=CB_color_cycle[1],
    )

    plt.plot(
        range(1, len(means_10) + 1),
        means_100,
        label=f"100 classes",
        color=CB_color_cycle[2],
    )
    plt.fill_between(
        range(1, len(means_10) + 1),
        means_100 - std_100,
        means_100 + std_100,
        alpha=0.2,
        color=CB_color_cycle[2],
    )
    set_plot(plt, means_10, xlabel="Layer", ylabel="Normalized Top-1 Accuracy [%]", normalize=False)
    plt.legend(loc="lower right", frameon=True, framealpha=0.6)

    plt.savefig("different_class_ninco.png", dpi=300)


def plot_different_samples_ninco():
    ID_data_name = "imagenet"
    OOD_data_name = "ninco"
    save_dir = f"plots/{ID_data_name}/acc_rank"
    vgg11_100_name = "vgg11_imagenet_samples_100"
    vgg11_200_name = "vgg11_imagenet_samples_200"
    vgg11_500_name = "vgg11_imagenet_samples_500"
    vgg11_1000_name = "vgg11_imagenet_samples_1000"

    means_100, std_100 = get_mean_std(ID_data_name, OOD_data_name, vgg11_100_name, True)
    means_200, std_200 = get_mean_std(ID_data_name, OOD_data_name, vgg11_200_name, True)
    means_500, std_500 = get_mean_std(ID_data_name, OOD_data_name, vgg11_500_name, True)
    means_1000, std_1000 = get_mean_std(ID_data_name, OOD_data_name, vgg11_1000_name, True)
    plt.figure(figsize=(10, 8))
    plot_acc_mean_std(
        plt,
        means_100,
        std_100,
        f"100 samples",
        CB_color_cycle[0],
    )
    plot_acc_mean_std(
        plt,
        means_200,
        std_200,
        f"200 samples",
        CB_color_cycle[1],
    )
    plot_acc_mean_std(
        plt,
        means_500,
        std_500,
        f"500 samples",
        CB_color_cycle[2],
    )
    plot_acc_mean_std(
        plt,
        means_1000,
        std_1000,
        f"1000 samples",
        CB_color_cycle[3],
    )

    set_plot(plt, means_100, xlabel="Layer", ylabel="Normalized Top-1 Accuracy [%]", normalize=False)

    plt.savefig("different_samples_ninco.png", dpi=300)


def plot_acc_mean_std(plt, means, std, label, color):
    plt.plot(
        range(1, len(means) + 1),
        means,
        label=label,
        color=color,
    )
    plt.fill_between(
        range(1, len(means) + 1),
        means - std,
        means + std,
        alpha=0.2,
        color=color,
    )


# scp -r ./data/NINCO kyungbok@klab-server2.rit.edu:/home/kyungbok/ninco
if __name__ == "__main__":
    for model in ["vit_tiny_patch8_imagenet100_64"]:
        # plot_compare_aug(model)
        plot_acc("imagenet100", "ninco", None, model, add_rank=False, add_title=True)
    # plot_acc_ninco("vit_tiny_patch8_imagenet100_64")
    # plot_ranks_vgg11_different_sample_size()
    # plot_acc(
    #     "imagenet",
    #     "places",
    #     "ninco",
    #     "vgg11_imagenet_class_10-2",
    #     add_rank=False,
    #     add_title=False,
    # )

    # plot_different_class_ninco()
    # plot_different_samples_ninco()

    # original_error = torch.load("values/errors/resnet34/original_32768.pt")
    # projection_error = torch.load("values/errors/resnet34/projection_32768.pt")
    # plot_error_dimension("resnet34", 32768, original=True, download=True)

    # model_names = ["mlp", "resnet34"]
    # plot_first_figure()
    # plot_NC4_train_test("imagenet", "mae")

    # model_names = ["resnet50", "resnet50_swav", "convnext", "vit", "swin", "dino", "mae", "mugs"]
    # for model_name in model_names:
    #     plot_NC1("imagenet", model_name)
    # plot_NC1_resolution()
    # plot_NC1("cifar10", "resnet34")
    # plot_NC4_resolution()
    # sing_224 = torch.load("values/imagenet100/singular_values/imagenet100/vgg13_imagenet100_224.pt")
    # sing_32 = torch.load("values/imagenet100/singular_values/imagenet100/vgg13_imagenet100_32.pt")
    # rank_32 = get_rankme_ranks(sing_32)
    # rank_224 = get_rankme_ranks(sing_224)

    # print(f"rank 32: {rank_32}")
    # print(f"rank 224: {rank_224}")
    # normalized_rank_32 = [x / max(rank_32) for x in rank_32]
    # normalized_rank_224 = [x / max(rank_224) for x in rank_224]
    # plt.figure(figsize=(10, 8))
    # plt.plot(range(1, len(normalized_rank_32) + 1), normalized_rank_32, label="32x32")
    # plt.plot(range(1, len(normalized_rank_224) + 1), normalized_rank_224, label="224x224")
    # plt.legend()
    # plt.savefig("temp.png", dpi=300)

    # plot_NC1_all()

    # model_names = [
    #     "vgg13_imagenet100_32",
    #     "vgg13_imagenet100_64",
    #     "vgg13_imagenet100_128",
    #     "vgg13_imagenet100_224",
    # ]
    # for model_name in model_names:
    #     plot_NC2("imagenet100", model_name)
    # for model_name in model_names:
    #     plot_NC2("imagenet100", model_name)
    # plot_NC("mlp", "cifar10", "NC1")
    # plot_NC1_resolution()

    # plot_resolution_NC2()
    # plot_acc("imagenet100", "places", "ninco", "vgg13_imagenet100_down_up", add_rank=False)

    # for model_name in model_names:
    #     id_data = "imagenet100"
    #     ood_data = "places"
    #     ood_data_2 = "ninco"
    #     tunnel_start = None
    #     plot_acc(
    #         id_data,
    #         ood_data,
    #         ood_data_2,
    #         model_name,
    #         tunnel_start,
    #         add_rank=False,
    #         add_title=False,
    #         normalize=False,
    #     )

    # model_names = [
    #     # "vgg13_imagenet100_32",
    #     # "vgg13_imagenet100_64",
    #     # "vgg13_imagenet100_128",
    #     "vgg13_imagenet100_224",
    # ]
    # for model in model_names:
    #     plot_NC(model, "imagenet100", "NC4")

    # model_names = ["mlp", "resnet34"]
    # for model_name in model_names:
    #     cor = compute_correlation("cifar10", "cifar100", model_name)
    #     print(f"{model_name} - {cor}")

    # plot_random_ranks(model_name=model_name, pretrained_data=pretrained)

    # models = ["convnextv2"]
    # ID_data_name = "imagenet"
    # OOD_data_name = "places"
    # for model in models:
    #     plot_acc(ID_data_name, OOD_data_name, model)
    #     plot_acc(ID_data_name, OOD_data_name, model, True)

    # models = ["resnet34", "resnet50_swav", "convnext", "resnet50"]

    # resnet50
    # plot_NC2_all()
