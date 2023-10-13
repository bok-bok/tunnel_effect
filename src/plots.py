import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import expit  # This is the sigmoid function


def plot_normalized_vars(**kwargs):
    for names, sigs_path in kwargs.items():
        sigs = torch.load(sigs_path)
        normalized_sums = get_normalized_vars(sigs)
        plt.plot(normalized_sums, label=names)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend()
    plt.show()


def get_normalized_vars(sigs):
    sums = [torch.sum(sig).detach().cpu().item() for sig in sigs]
    mean = np.mean(sums)
    normalized_sums = [sum / mean for sum in sums]
    return normalized_sums


def get_normalized_mean_vars(vars):
    means = [torch.mean(var).detach().cpu() for var in vars]
    total_mean = np.mean(means)
    normalized_mean = [mean / total_mean for mean in means]
    return normalized_mean


def count_sigs(sigs):
    # count number of sigs value that cover 90% of the sum of sigs
    total_sum = sigs.sum()
    cumulative_sum = torch.cumsum(sigs, dim=0)
    important_sigs = (cumulative_sum / total_sum <= 0.9).sum().item()
    return important_sigs


def plot_ranks(**kwargs):
    threshold = 1
    for names, sigs_path in kwargs.items():
        sigs = torch.load(sigs_path)
        ranks = []
        for sig in sigs:
            count = (sig > threshold).sum().item()
            ranks.append(count)
        plt.plot(ranks, label=names)
    plt.ylim(bottom=0)
    plt.legend()
    plt.show()


def get_ranks(sigs, threshold):
    ranks = []
    for sig in sigs:
        count = (sig > threshold).sum().item()
        ranks.append(count)
    return ranks


def compare_rank_normalized_sigs(sigs, model_name, threshold, download=False):
    ranks = get_ranks(sigs, threshold)
    normalized_sigs = get_normalized_vars(sigs)
    plot_title = f"{model_name} normalized sigs vs ranks"
    label_threshold = f"{model_name} ranks threshold - {threshold}"
    label_normalized = f"{model_name} normalized sigs"

    fig, ax1 = plt.subplots()
    ax1.plot(ranks, label=label_threshold)
    ax1.set_ylabel("Ranks")

    ax2 = ax1.twinx()
    ax2.plot(normalized_sigs, label=label_normalized)
    ax2.set_ylabel("Normalized Sigs")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]

    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray"]
    for i, line in enumerate(lines):
        line.set_color(colors[i % len(colors)])

    ax1.legend(lines, labels, loc="best")

    plt.title(plot_title)
    if download:
        plt.savefig(f"plots/rank_normalized/{model_name}_{threshold}.png", dpi=300)
    plt.show()


def plot_normalized_vars_acc(title, vars, acc, acc_OOD, download=False):
    x_axis = [i for i in range(len(acc))]
    if "resnet" in title.lower():
        vars = vars[2:]
        acc = acc[2:]
        if acc_OOD is not None:
            acc_OOD = acc_OOD[2:]
        x_axis = [i + 3 for i in range(len(acc))]

    # normalized_vars = get_normalized_vars(vars)
    normalized_vars = vars
    fig, ax1 = plt.subplots()
    if "sing" in title.lower():
        label = "sig counts"
    else:
        label = "normalized vars"
    ax1.plot(x_axis, normalized_vars, label=label)
    plt.ylim(bottom=0)
    ax2 = ax1.twinx()
    ax2.plot(x_axis, acc, label="acc")
    if acc_OOD is not None:
        ax2.plot(x_axis, acc_OOD, label="OOD acc")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray"]
    for i, line in enumerate(lines):
        line.set_color(colors[i % len(colors)])
    plt.title(title)

    ax1.legend(lines, labels, loc="best")
    plt.xlim(min(x_axis), max(x_axis))
    # disply x axis considering xlim
    plt.xticks(np.arange(min(x_axis), max(x_axis) + 1, 2.0))
    if download:
        plt.savefig(f"plots/{title}.png", dpi=300)
    plt.show()


def compare_normalized_sigs(sig_dict, plot_title):
    scratch_sig = None
    sig = None
    fig, ax1 = plt.subplots()
    for name, sigs in sig_dict.items():
        normalized_sigs = get_normalized_vars(sigs)
        if "scratch" in name:
            scratch_sig = normalized_sigs
        else:
            sig = normalized_sigs
        ax1.plot(normalized_sigs, label=name)
    plt.ylim(bottom=0)
    if scratch_sig is not None and sig is not None:
        ax2 = ax1.twinx()
        new_sigs = [sig / scratch_sig for sig, scratch_sig in zip(sig, scratch_sig)]
        ax2.plot(new_sigs, label="normalized(sig/scratch_sig)")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]

    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray"]
    for i, line in enumerate(lines):
        line.set_color(colors[i % len(colors)])

    ax1.legend(lines, labels, loc="best")
    plt.title(plot_title)
    plt.ylim(bottom=0)
    plt.show()


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


def plot_normalized_acc(model_name, download=False):
    var_path = f"values/vars/{model_name}.pt"
    var = torch.load(var_path)
    normalized_vars = get_normalized_vars(var)
    normalized_title = f"{model_name} acc vs normalized vars"
    acc_path = f"values/acc/{model_name}.pt"
    acc = torch.load(acc_path)
    plot_x_acc(model_name, normalized_title, normalized_vars, acc, download)


def plot_count_sigs_acc(model_name, download=False):
    sig_count_path = f"values/sigs_count/{model_name}.pt"
    sig_count = torch.load(sig_count_path)
    sig_count_title = f"{model_name} acc vs sig counts"
    acc_path = f"values/acc/{model_name}.pt"
    acc = torch.load(acc_path)
    plot_x_acc(model_name, sig_count_title, sig_count, acc, download)


def show(data, title, download=False):
    if "resnet" in title.lower():
        data = data[2:]
        plt.plot(range(3, len(data) + 3), data)
        plt.title(title)
    else:
        plt.plot(range(1, len(data) + 1), data)
        plt.title(title)

    if download:
        plt.savefig(f"plots/{title}.png", dpi=300)
    plt.show()


model_name = "resnet34"


# var_mean = [torch.mean(var).detach().cpu().item() for var in vars]
# var_var = [torch.var(var).detach().cpu().item() for var in vars]
# max_var = [torch.max(var).detach().cpu().item() for var in vars]
# min_var = [torch.min(var).detach().cpu().item() for var in vars]
# n_vars = [sum(var) / m_var for var, m_var in zip(vars, max_var)]
# sknewness = [() for var in vars]
# get mean of top 10% vars consider it's range


def get_top_var(vars, threshold, size):
    top_10_var = []
    # cut_vars = []
    # for var in vars:
    #     if var.shape[0] > size:
    #         cur_var = var[:size]
    #     else:
    #         cur_var = var
    #     cut_vars.append(cur_var)
    threshold = 0.1
    for i, var in enumerate(vars):
        range_ = max(var) - min(var)
        ten_vars = [v for v in var if v > (range_ * threshold)]
        top_10_var.append(len(ten_vars))
    return top_10_var


vars_0 = torch.load(f"values/vars/{model_name}_0.pt")
vars_1 = torch.load(f"values/vars/{model_name}_1.pt")
vars_2 = torch.load(f"values/vars/{model_name}_2.pt")
vars_3 = torch.load(f"values/vars/{model_name}_3.pt")

mean_var_0 = [torch.mean(var).detach().cpu().item() for var in vars_3]
print(mean_var_0)
count_bigger_than_mean = [sum(var > mean) / len(var) for var, mean in zip(vars_0, mean_var_0)]
plt.plot(count_bigger_than_mean)
plt.show()

# top_10_var0 = get_top_var(vars_0, 0.1, 8000)
# top_10_var1 = get_top_var(vars_1, 0.1, 8000)
# top_10_var2 = get_top_var(vars_2, 0.1, 8000)
# top_10_var3 = get_top_var(vars_3, 0.1, 8000)

# avg_top_10_var = np.array(
#     [
#         (v0 + v1 + v2 + v3) / 4
#         for v0, v1, v2, v3 in zip(top_10_var0, top_10_var1, top_10_var2, top_10_var3)
#     ]
# )
# sdv_top_10_var = np.array(
#     [
#         np.std([v0, v1, v2, v3])
#         for v0, v1, v2, v3 in zip(top_10_var0, top_10_var1, top_10_var2, top_10_var3)
#     ]
# )
# acc = torch.load(f"values/acc/{model_name}.pt")
# # plot avg and std of top 10% vars
# plot_x_acc(model_name, "top_10_avg_full_variance", avg_top_10_var, acc, True)


# plt.plot(n_vars, label="cut vars")
# plt.plot(normalized_vars, label="vars")
# plt.legend()
# plt.show()
def threshold_from_skewness(skewness, skew_min, skew_max):
    def custom_sigmoid(x, k=3, x0=0):
        return 1 / (1 + np.exp(-k * (x - x0)))

    if skewness <= skew_min:
        return 0
    elif skewness >= skew_max:
        return 1
    else:
        # Linear scaling between skew_min and skew_max
        return 1 - custom_sigmoid(skewness)


threshold = 0.1

top_10_var_1 = []
# for i, var in enumerate(cut_vars):
#     print(f"layer {i} var shape: {len(var)}")
#     range_ = max(var) - min(var)
#     ten_vars = [v for v in var if v > (range_ * threshold)]
#     top_10_var_1.append(len(ten_vars))


# top_10_var = []
# for i, var in enumerate(vars):
#     print(f"layer {i} var shape: {len(var)}")
#     range_ = max(var) - min(var)
#     ten_vars = [v for v in var if v > (range_ * threshold)]
#     top_10_var.append(len(ten_vars))

# plt.plot(top_10_var, label="top 10% vars")
# plt.plot(top_10_var_1, label="top 10% vars 1")
