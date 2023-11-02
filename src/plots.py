import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import torch

plt.style.use("science")

plt.rc("text", usetex=False)  # disable LaTeX font
os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"
plt.rc("font", size=16, weight="bold")

plt.rc("lines", linewidth=2)


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

    plt.figure(figsize=(6, 4))
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
    plt.legend()
    plt.grid()
    if download:
        plt.savefig(f"{model_name}_MAE_dimension", dpi=300)
    plt.show()


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

    tunnel_start_idx = find_tunnel_start(acc_mean, 0.98)

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
    plt.axvspan(tunnel_start_idx, len(counts_mean), alpha=0.2, color="green")
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

    tunnel_start_idx = find_tunnel_start(acc_for_tunnel)

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(x_axis, rank, label="Rank")

    ax1.set_xlabel("Layers")
    ax1.set_ylabel("Rank")
    ax2 = ax1.twinx()
    ax2.plot(x_axis, accuracy, label="Accuracy")

    ax2.set_ylabel("Accuracy")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray"]
    for i, line in enumerate(lines):
        line.set_color(colors[i % len(colors)])
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
    print(save_dir)
    if not os.path.exists(os.path.dirname(save_dir)):
        os.makedirs(os.path.dirname(save_dir))

    title = f"{model_name} {data_name} {threshold_name} {'cov' if cov else ''} {'original_dim' if original_dim else 'reduced_dim'} {'square' if square else 'non_squared'}"
    plt.title(title)

    plt.savefig(f"{save_dir}/{model_name}.png", dpi=300)
    plt.close()


def get_dir(feature_type, normalize, knn):
    return f"values/cifar10/{'knn_ood_acc' if knn else 'ood_acc'}/{feature_type}/resnet34{'_norm' if normalize else ''}.pt"


def get_mean_std(data_name, OOD, model_name):
    directory = f"values/{data_name}/{'ood_acc' if OOD else 'acc'}/{model_name}"
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
    "resnet34": "ResNet34",
    "resnet50": "ResNet50",
    "resnet50_swav": "ResNet50 Swav",
    "convnext": "ConvNext",
    "swin": "Swin Transformer",
}


def plot_acc(data_name, model_name, add_title=False):
    ood_means, ood_stds = get_mean_std(data_name, True, model_name)
    id_means, id_stds = get_mean_std(data_name, False, model_name)
    tunnel_start = find_tunnel_start(id_means)

    plt.figure(figsize=(6, 4))
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
    ID_data_name = "ImageNet-1K" if data_name == "imagenet" else "CIFAR-10"
    OOD_data_name = "Places-365" if data_name == "imagenet" else "CIFAR-100"

    plt.plot(range(1, len(ood_means) + 1), ood_means, label=f"OOD({OOD_data_name})")
    plt.fill_between(
        range(1, len(ood_means) + 1), ood_means - ood_stds, ood_means + ood_stds, alpha=0.2
    )

    plt.plot(range(1, len(id_means) + 1), id_means, label=f"ID({ID_data_name})")

    # add best accuracy on y axis y ticks
    # if best accuracy is overlap with last y tick, remove last y tick
    # Get current y-ticks
    yticks = list(plt.yticks()[0])

    # Add best accuracy to y-ticks
    yticks.append(max(id_means))

    # Sort the y-ticks
    yticks = sorted(yticks)
    # print(yticks)

    for i in range(len(yticks) - 1):
        if abs(yticks[i] - yticks[i + 1]) < 2:
            yticks[i] = 0
    # Check and remove overlapping y-ticks
    # if (
    #     len(yticks) > 1 and abs(yticks[-1] - yticks[-2]) < 2
    # ):  # you might adjust the threshold (0.01) as needed
    #     yticks = yticks[:-2] + [yticks[-1]]

    # Set the y-ticks
    plt.yticks(yticks)

    # plt.yticks(list(plt.yticks()[0]) + [max(id_means)])
    plt.fill_between(range(1, len(id_means) + 1), id_means - id_stds, id_means + id_stds, alpha=0.2)
    if tunnel_start is not None:
        plt.axvspan(tunnel_start, len(ood_means), alpha=0.2, color="green")

    plt.ylabel("Top-1 Accuracy[%]")
    plt.xlabel(f"{model_name_convert[model_name]} Layer")
    plt.xlim(left=1, right=len(ood_means))

    plt.grid()
    plt.legend()
    if add_title:
        plt.title(f"{model_name_convert[model_name]} - {data_name}")
        plt.savefig(f"{model_name}_{data_name}_title.png", dpi=300)
    else:
        plt.savefig(f"{model_name}_{data_name}.png", dpi=300)


if __name__ == "__main__":
    models = ["resnet50_swav", "convnext"]
    for model in models:
        plot_acc("imagenet", model)
        plot_acc("imagenet", model, True)
    # plot_acc("cifar10", "resnet34")
    # ood_acc = torch.load(f"values/imagenet/acc/resnet50/1.pt")

    # print(ood_acc)
