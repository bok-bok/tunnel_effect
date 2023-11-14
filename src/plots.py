import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import torch

plt.style.use("science")

plt.rc("text", usetex=False)  # disable LaTeX font
plt.rc("font", size=17, weight="bold")
os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"

plt.rc("lines", linewidth=3)


def get_ranks(sigs, threshold):
    ranks = []
    for sig in sigs:
        count = (sig > threshold).sum().item()
        ranks.append(count)
    return ranks


def plot_x_acc(model_name, title, ranks, acc, download=False):
    fig, ax1 = plt.subplots()
    if "resnet" in model_name.lower():
        ranks = ranks[2:]
        acc = acc[2:]
        print(len(acc))

        ax1.plot(range(3, len(ranks) + 3), ranks, label="ranks")
        ax2 = ax1.twinx()
        ax2.plot(range(3, len(acc) + 3), acc, label="acc")
        plt.xlim(left=3)
    else:
        ax1.plot(ranks, label="ranks")

        ax2 = ax1.twinx()
        ax2.plot(acc, label="acc")
        plt.xlim(left=0)

    plt.title(title)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray"]
    for i, line in enumerate(lines):
        line.set_color(colors[i % len(colors)])
    ax1.legend(lines, labels, loc="best")
    if download:
        if "count" in title.lower():
            plt.savefig(f"plots/sigs_count/{model_name}.png", dpi=300)
        elif "normalized" in title.lower():
            plt.savefig(f"plots/normalized/{model_name}.png", dpi=300)
        elif "ranks" in title.lower():
            plt.savefig(f"plots/ranks/{model_name}.png", dpi=300)
        else:
            plt.savefig(f"{title}.png", dpi=300)
        # plt.savefig(f"plots/{title}.png", dpi=300)
    plt.show()


def plot_ranks_acc(model_name, threshold, download=False):
    sigs_path = f"values/sigs/{model_name}.pt"
    sigs = torch.load(sigs_path)
    ranks = get_ranks(sigs, threshold)
    acc_path = f"values/acc/{model_name}.pt"
    acc = torch.load(acc_path)
    ranks_title = f"{model_name} acc vs ranks"
    plot_x_acc(model_name, ranks_title, ranks, acc, download)


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
        plt.plot(
            list(original_errors.keys()), list(original_errors.values()), label="Original Method"
        )
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


def plot_mean_std(mean, std, label):
    mean = np.array(list(mean))
    std = np.array(list(std))
    plt.plot(mean, label=label)
    plt.fill_between(
        mean - std,
        mean + std,
        alpha=0.2,
    )


def load_files(model_name, target_value_type):
    try:
        values_path = f"values/{target_value_type}"
    except:
        print(f"{target_value_type} not found")
        return
    files = os.listdir(values_path)
    model_files = [f"values/{target_value_type}/{file}" for file in files if model_name in file]
    return model_files


def get_mean_std(model_name, target_value_type):
    if target_value_type == "rank":
        files = load_files(model_name, "singular_values")
        values = [get_dynamic_ranks(model_name, torch.load(file)) for file in files]
    else:
        files = load_files(model_name, target_value_type)
        values = [torch.load(file) for file in files]
    if "resnet34" in model_name.lower():
        values = [value[2:] for value in values]
    means = []
    stds = []

    for i in range(len(values[0])):
        mean = np.mean([arg[i] for arg in values])
        std = np.std([arg[i] for arg in values])
        means.append(mean)
        stds.append(std)
    means = np.array(means)
    stds = np.array(stds)
    return means, stds


def plot_mlp():
    model_name = "mlp"
    counts_mean, counts_std = get_mean_std(model_name, "rank")
    acc_mean, acc_std = get_mean_std(model_name, "acc")
    ood_mean, ood_std = get_mean_std(model_name, "ood_acc")

    x_axis = range(1, len(counts_mean) + 1)

    # tunnel_start_idx = find_tunnel_start(acc_mean, 0.98)

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(x_axis, counts_mean, label="Rank")

    ax1.fill_between(
        x_axis,
        counts_mean - counts_std,
        counts_mean + counts_std,
        alpha=0.2,
    )

    ax1.set_xlabel("Layers")
    ax1.set_ylabel("Rank")
    ax2 = ax1.twinx()
    # ax2.plot(x_axis, acc_mean, label="Accuracy")
    # ax2.fill_between(
    #     x_axis,
    #     acc_mean - acc_std,
    #     acc_mean + acc_std,
    #     alpha=0.2,
    # )

    ax2.plot(x_axis, ood_mean, label="OOD acc")
    ax2.fill_between(
        x_axis,
        ood_mean - ood_std,
        ood_mean + ood_std,
        alpha=0.2,
    )
    x_axis = range(1, len(counts_mean) + 1)

    ax2.set_ylabel("Accuracy")
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray"]
    for i, line in enumerate(lines):
        line.set_color(colors[i % len(colors)])
    plt.legend(lines, labels, loc="lower center")
    plt.xlim(left=x_axis[0], right=x_axis[-1])
    # plt.axvspan(tunnel_start_idx, len(counts_mean), alpha=0.2, color="green")
    plt.xticks(x_axis, labels=[str(x) for x in x_axis])

    plt.grid(axis="both")
    plt.savefig(f"mlp_ood_rank_acc", dpi=300)
    plt.show()


def plot_resnet():
    model_name = "resnet34"
    counts_mean, counts_std = get_mean_std(model_name, "rank")
    acc_mean, acc_std = get_mean_std(model_name, "acc")
    ood_mean, ood_std = get_mean_std(model_name, "ood_acc")

    x_axis = range(3, len(counts_mean) + 3)

    tunnel_start_idx = find_tunnel_start(acc_mean) + 2

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(x_axis, counts_mean, label="Rank")

    ax1.fill_between(
        x_axis,
        counts_mean - counts_std,
        counts_mean + counts_std,
        alpha=0.2,
    )

    ax1.set_xlabel("Layers")
    ax1.set_ylabel("Rank")
    ax2 = ax1.twinx()

    # ax2.plot(x_axis, acc_mean, label="Accuracy")
    # ax2.fill_between(
    #     x_axis,
    #     acc_mean - acc_std,
    #     acc_mean + acc_std,
    #     alpha=0.2,
    # )

    ax2.plot(x_axis, ood_mean, label="OOD acc")
    ax2.fill_between(
        x_axis,
        ood_mean - ood_std,
        ood_mean + ood_std,
        alpha=0.2,
    )

    ax2.set_ylabel("Accuracy")
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray"]
    for i, line in enumerate(lines):
        line.set_color(colors[i % len(colors)])
    plt.legend(lines, labels, loc="lower center")
    plt.xlim(left=3, right=len(counts_mean) + 2)
    plt.axvspan(tunnel_start_idx, len(counts_mean) + 2, alpha=0.2, color="green")
    plt.xticks(x_axis[::2], labels=[str(x) for x in x_axis[::2]])

    plt.grid(axis="both")
    plt.savefig(f"resnet34_ood_ranks_acc", dpi=300)
    plt.show()


def plot_rank_acc(model_name, OOD, download=False, skip_count=0):
    singular_values = torch.load(f"values/singular_values/{model_name}.pt")

    # acc for probe start of tunnel
    acc_for_tunnel = torch.load(f"values/acc/{model_name}.pt")
    if not OOD:
        acc_path = f"values/acc/{model_name}.pt"
    else:
        acc_path = f"values/ood_acc/{model_name}.pt"

    rank = get_dynamic_ranks(model_name, singular_values)[skip_count:]
    accuracy = torch.load(acc_path)[skip_count:]

    x_axis = range(skip_count + 1, len(rank) + skip_count + 1)

    tunnel_start_idx = find_tunnel_start(acc_for_tunnel, 0.98)

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(x_axis, rank, label="Rank")

    ax1.set_xlabel("Layers")
    ax1.set_ylabel("Rank")
    ax2 = ax1.twinx()
    ax2.plot(x_axis, accuracy, label="Accuracy")

    ax2.set_ylabel("Accuracy")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    plt.legend(lines, labels, loc="lower center")
    # show xticks
    # if len(xticks) > 20: skip by 2
    if len(x_axis) > 20:
        plt.xticks(x_axis[::2], labels=[str(x) for x in x_axis[::2]])
    else:
        plt.xticks(x_axis, labels=[str(x) for x in x_axis])

    plt.xlim(left=x_axis[0], right=x_axis[-1])
    plt.axvspan(tunnel_start_idx, len(rank) + skip_count, alpha=0.2, color="green")

    plt.grid()
    if download:
        if OOD:
            plt.savefig(f"{model_name}_ood_rank_accuracy", dpi=300)
        else:
            plt.savefig(f"{model_name}_rank_accuracy", dpi=300)
    plt.show()


def get_dynamic_ranks(model_name, data_name, cov, threshold_name, square, original_dim):
    if cov:
        sigs_path = f"values/{data_name}/singular_values_cov/{model_name}.pt"
        sigs = torch.load(sigs_path)
    else:
        sigs_path = f"values/{data_name}/singular_values_direct/{model_name}.pt"
        sigs = torch.load(sigs_path)
        if square:
            sigs = [sig**2 for sig in sigs]

    structure_name = model_name.split("_")[0]

    dims = torch.load(f"values/{data_name}/dimensions/{structure_name}.pt")
    variances = torch.load(f"values/{data_name}/variances/{structure_name}_cov.pt")
    ranks = []
    if "cifar" in data_name:
        n = 10000
    else:
        n = 15000

    for i, sig in enumerate(sigs):
        # eps = torch.finfo(torch.float32).eps * dims[i]

        # threshold = torch.max(sig) * eps
        if original_dim:
            feature_number = dims[i]
        else:
            feature_number = len(sig)
        if threshold_name == "original":
            threshold = get_original_threshold(torch.max(sig), feature_number)
        elif threshold_name == "mp":
            threshold = get_mp_threshold(variances[i], feature_number, n)
        elif threshold_name == "med":
            threshold = median_threshold(sig, feature_number, n)
        # print(threshold, sig[-10:])
        count = (sig > threshold).sum().item()
        ranks.append(count)
    return ranks


def get_original_threshold(max_sig, dim):
    return max_sig * torch.finfo(torch.float32).eps * dim


def get_mp_threshold(variance, m, n):
    # m feature
    # n sample
    # m / n
    threshold = variance * (1 + np.sqrt(m / n)) ** 2
    return threshold


def median_threshold(sig, m, n):
    # m feature
    # n sample

    beta = m / n
    med = torch.median(sig)
    return (0.56 * (beta**3) - 0.95 * (beta**2) + 1.82 * (beta)) * med


def plot_rank(
    model_name, data_name, cov, threshold_name="original", square=False, original_dim=False
):
    ranks = get_dynamic_ranks(model_name, data_name, cov, threshold_name, square, original_dim)

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(ranks) + 1), ranks)
    plt.title(f"{model_name} rank")
    plt.xlabel("Layers")
    plt.ylabel("Rank")

    save_dir = f"./plots/{'squared' if square else 'non_squared'}/{threshold_name}/{'cov' if cov else 'direct'}/{'original_dim' if original_dim else 'reduced_dim'}/"
    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))

    title = f"{model_name} {data_name} {threshold_name} {'cov' if cov else ''} {'original_dim' if original_dim else 'reduced_dim'} {'square' if square else 'non_squared'}"
    plt.title(title)

    plt.savefig(f"{save_dir}/{model_name}.png", dpi=300)
    plt.close()


def get_mean_std(data_name, specific_name, model_name):
    # set OOD or ID
    OOD = False if data_name == specific_name else True

    # add specific name to directory if OOD
    directory = f"values/{data_name}/{f'ood_acc/{specific_name}' if OOD else 'acc'}/{model_name}"
    files = os.listdir(directory)
    values = [torch.load(f"{directory}/{file}") for file in files]
    means = []
    stds = []
    for i in range(len(values[0])):
        mean = np.mean([arg[i] for arg in values])
        std = np.std([arg[i] for arg in values])
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
    "vgg13_imagenet100_32": "VGG13_32",
    "vgg13_imagenet100_64": "VGG13_64",
    "vgg13_imagenet100_128": "VGG13_128",
    "vgg13_imagenet100_224": "VGG13_224",
    "vgg13_imagenet_class_10": "VGG13 - 10 Classes",
    "vgg13_imagenet_class_50": "VGG13 - 50 Classes",
    "vgg13_imagenet_class_100": "VGG13 - 100 Classes",
}


data_name_convert = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "imagenet": "ImageNet",
    "imagenet100": "ImageNet",
    "places": "Places",
    "ninco": "NINCO",
}


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
    random_path = (
        f"values/{pretrained_data}/singular_values/{pretrained_data}/{arch_name}_random_init.pt"
    )

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
    random_path = (
        f"values/{pretrained_data}/singular_values/{pretrained_data}/{arch_name}_random_init.pt"
    )

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
    ID_COLOR = "#377EB8"
    OOD_COLOR = "#4DAF4A"
    OOD_2_COLOR = "#A65628"
    RANK_COLOR = "#E41A1C"

    save_dir = f"plots/{ID_data_name}/acc_rank"

    ood_means, ood_stds = get_mean_std(ID_data_name, OOD_data_name, model_name)
    ood_means_2, ood_stds_2 = get_mean_std(ID_data_name, OOD_data_name2, model_name)

    id_means, id_stds = get_mean_std(ID_data_name, ID_data_name, model_name)
    if normalize:
        ood_means = np.array(ood_means) / max(ood_means)
        ood_means_2 = np.array(ood_means_2) / max(ood_means_2)
        id_means = np.array(id_means) / max(id_means)
        ood_stds = np.array(ood_stds) / max(ood_stds)
        ood_stds_2 = np.array(ood_stds_2) / max(ood_stds_2)
        id_stds = np.array(id_stds) / max(id_stds)

    # tunnel_start = find_tunnel_start(id_means, 0.95)

    if add_rank:
        singular_values = torch.load(
            f"values/{ID_data_name}/singular_values/{ID_data_name}/{model_name}.pt"
        )
        ranks = get_rankme_ranks(singular_values)
        # ranks = get_original_ranks(singular_values)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.rc("font", size=17, weight="bold")
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
        plt.xticks(
            range(1, len(ood_means) + 1), labels=[str(x) for x in range(1, len(ood_means) + 1)]
        )

    ID_data_name = data_name_convert[ID_data_name]
    OOD_data_name = data_name_convert[OOD_data_name]
    OOD_data_name2 = data_name_convert[OOD_data_name2]

    # plot ID
    ax1.plot(range(1, len(id_means) + 1), id_means, label=f"ID({ID_data_name})", color=ID_COLOR)
    ax1.fill_between(
        range(1, len(id_means) + 1),
        id_means - id_stds,
        id_means + id_stds,
        alpha=0.2,
        color=ID_COLOR,
    )

    # plot OOD1
    ax1.plot(
        range(1, len(ood_means) + 1), ood_means, label=f"OOD({OOD_data_name})", color=OOD_COLOR
    )
    ax1.fill_between(
        range(1, len(ood_means) + 1),
        ood_means - ood_stds,
        ood_means + ood_stds,
        alpha=0.2,
        color=OOD_COLOR,
    )

    # plot OOD2
    ax1.plot(
        range(1, len(ood_means_2) + 1),
        ood_means_2,
        label=f"OOD({OOD_data_name2})",
        color=OOD_2_COLOR,
    )
    ax1.fill_between(
        range(1, len(ood_means_2) + 1),
        ood_means_2 - ood_stds_2,
        ood_means_2 + ood_stds_2,
        alpha=0.2,
        color=OOD_2_COLOR,
    )

    y_start = min(min(ood_means), min(id_means)) - 1
    ax1.set_ylim(bottom=y_start)

    # Add best accuracy to y-ticks
    # yticks = list(plt.yticks()[0])
    # yticks.append(max(id_means))
    # yticks = sorted(yticks)
    # for i in range(len(yticks) - 1):
    #     if abs(yticks[i] - yticks[i + 1]) < 2:
    #         yticks[i] = 0
    # plt.yticks(yticks)
    plt.grid()

    # plt.yticks(list(plt.yticks()[0]) + [max(id_means)])
    # plt.yticks(list(plt.yticks()[0]) + [max(id_means)])
    if tunnel_start is not None:
        plt.axvspan(tunnel_start, len(ood_means), alpha=0.2, color="green")

    ax1.set_ylabel("Top-1 Accuracy [%]", size=20)
    plt.xlabel(f"Layer", size=20)
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
    plt.ylim(bottom=0)

    # 0 from yticks
    # yticks = list(plt.yticks()[0])[1:-1]
    # plt.yticks(yticks)

    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))

    file_name = (
        f"{model_name}{'_title' if add_title else ''}{'_normalized' if normalize else ''}.png"
    )
    if add_title:
        plt.title(f"{model_name_convert[model_name]} - {ID_data_name}")
    plt.savefig(f"{save_dir}/{file_name}", dpi=300)


def compute_correlation(ID_data_name, OOD_data_name, model_name):
    ood_means, ood_stds = get_mean_std(ID_data_name, OOD_data_name, model_name)
    id_means, id_stds = get_mean_std(ID_data_name, ID_data_name, model_name)
    correlation = np.corrcoef(ood_means, id_means)[0, 1]
    return correlation


if __name__ == "__main__":
    # original_error = torch.load("values/errors/resnet34/original_32768.pt")
    # projection_error = torch.load("values/errors/resnet34/projection_32768.pt")
    # plot_error_dimension("resnet34", 32768, original=True, download=True)

    # model_names = [
    #     "vgg13_imagenet100_32",
    #     "vgg13_imagenet100_64",
    #     "vgg13_imagenet100_128",
    #     "vgg13_imagenet100_224",
    # ]

    model_names = ["resnet50", "resnet50_swav", "convnext", "dino", "mae", "mugs", "swin", "vit"]
    # model_names = ["resnet50"]
    for model_name in model_names:
        id_data = "imagenet"
        ood_data = "places"
        ood_data_2 = "ninco"
        tunnel_start = None
        plot_acc(
            id_data,
            ood_data,
            ood_data_2,
            model_name,
            tunnel_start,
            add_rank=False,
            add_title=False,
            normalize=False,
        )

    # model_names = [
    #     "resnet50",
    #     "resnet50_swav",
    #     "convnext",
    #     "resnet34",
    #     "vit",
    #     "swin",
    #     "dino",
    #     "mae",
    # ]
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
