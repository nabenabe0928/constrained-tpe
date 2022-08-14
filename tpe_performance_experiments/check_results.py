from typing import Dict, List, Literal

import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, wilcoxon
from tpe_performance_experiments.constants import TASK_NAMES


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18  # 24
plt.rcParams["mathtext.fontset"] = "stix"  # The setting of math font
COLOR_DICT = {
    "tpe": "red",
    "hyperopt": "black",
    "turbo5": "olive",
    "turbo1": "magenta",
    "cocabo": "lime",
}
LABEL_DICT = {
    "tpe": "TPE",
    "hyperopt": "Hyperopt",
    "turbo5": "TuRBO-5",
    "turbo1": "TuRBO-1",
    "cocabo": "CoCaBO",
}
ORACLE_DICT = {
    "slice_localization": 0.00015851979,
    "protein_structure": 0.22013874,
    "naval_propulsion": 2.4792556e-05,
    "parkinsons_telemonitoring": 0.003821034,
    "cifar10A": 0.04817706346511841,
    "cifar10B": 0.04817706346511841,
    "imagenet": 52.733333333333334,
    "cifar10": 8.28000000976563,
    "cifar100": 26.120000000000005,
}


def get_data(
    metric: Literal["mean", "median"],
    epochs: List[int] = [49, 99, 149, 199],
    return_ste: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    data = {}
    ste_data = {}
    opt_names = ["hyperopt", "turbo1", "turbo5", "cocabo", "tpe"]
    for opt_name in opt_names:
        results = json.load(open(f"results-tpe-performance/{opt_name}.json"))
        data[opt_name] = {}
        ste_data[opt_name] = {}

        for task_name, loss_vals in results.items():
            if len(loss_vals) == 0:
                continue

            if metric == "median":
                data[opt_name][task_name] = np.median(
                    np.minimum.accumulate(loss_vals, axis=-1), axis=0
                )
            elif metric == "mean":
                data[opt_name][task_name] = np.mean(
                    np.minimum.accumulate(loss_vals, axis=-1), axis=0
                )
            if return_ste:
                loss_vals = np.asarray(loss_vals)
                oracle = ORACLE_DICT[task_name]
                loss_vals = (loss_vals - oracle) / oracle
                ste_data[opt_name][task_name] = np.std(
                    np.minimum.accumulate(loss_vals, axis=-1), axis=0
                ) / np.sqrt(loss_vals.shape[0])
                ste_data[opt_name][task_name] = ste_data[opt_name][task_name][epochs]

            data[opt_name][task_name] = data[opt_name][task_name][epochs]

    if return_ste:
        return data, ste_data
    else:
        return data


def main(
    epoch_idx: int, metric: Literal["mean", "median"], print_detail: bool = False
) -> None:
    data = get_data(metric)
    opt_names = list(data.keys())
    if print_detail:
        for task_name in TASK_NAMES:
            print("###", task_name, "###")
            for opt_name in opt_names:
                try:
                    print(opt_name, data[opt_name][task_name][epoch_idx])
                except KeyError:
                    pass

    for i in range(len(opt_names)):
        opt1 = opt_names[i]
        for j in range(i + 1, len(opt_names)):
            opt2 = opt_names[j]
            p_val = wilcoxon(
                np.array(
                    [data[opt1][task_name][epoch_idx] for task_name in TASK_NAMES]
                ),
                np.array(
                    [data[opt2][task_name][epoch_idx] for task_name in TASK_NAMES]
                ),
                alternative="less",
            ).pvalue
            print(f"{opt2} is better than {opt1} with p={p_val}")


def plot_perf_over_time(figsize=(27, 18)):
    fig, axes = plt.subplots(
        nrows=3, ncols=3, figsize=figsize, sharex=True,
    )
    means, stes = get_data(metric="mean", epochs=list(range(200)), return_ste=True)
    opt_names = list(means.keys())
    dx = np.arange(200) + 1
    idx = 0

    for task_name in ORACLE_DICT.keys():
        lines, labels = [], []
        r, c = idx // 3, idx % 3
        ax = axes[r][c]
        idx += 1
        for opt_name in opt_names:
            label, color = LABEL_DICT[opt_name], COLOR_DICT[opt_name]
            m, s = means[opt_name][task_name], stes[opt_name][task_name]
            (line, ) = ax.plot(dx, m, color=color, label=label)
            ax.fill_between(dx, m - s, m + s, alpha=0.2, color=color)
            lines.append(line)
            labels.append(label)
            ax.set_yscale("log")
            ax.grid(which='minor', color='gray', linestyle=':')
            ax.grid(which='major', color='black')
            ax.set_title(task_name)

    ax.grid()
    ax.legend(
        handles=lines,
        labels=labels,
        loc="upper center",
        fontsize=24,
        bbox_to_anchor=(-0.7, -0.15),
        fancybox=False,
        shadow=False,
        ncol=len(labels),
    )
    plt.show()


def plot_average_rank(figsize=(9, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    data = get_data(metric="mean", epochs=list(range(200)))
    opt_names = list(data.keys())

    ranks = np.zeros((len(opt_names), 200), dtype=np.float32)
    for task_name in TASK_NAMES:
        ranks += rankdata([data[opt_name][task_name] for opt_name in opt_names], axis=0)

    ranks /= len(TASK_NAMES)
    dx = np.arange(1, 201)
    lines, labels = [], []
    for idx, opt_name in enumerate(opt_names):
        (line,) = ax.plot(
            dx, ranks[idx], label=LABEL_DICT[opt_name], color=COLOR_DICT[opt_name]
        )
        lines.append(line)
        ax.set_xlabel("# of config evaluations")
        ax.set_ylabel("Average rank")
        labels.append(LABEL_DICT[opt_name])

    ax.grid()
    ax.legend(
        handles=lines,
        labels=labels,
        loc="upper center",
        fontsize=24,
        bbox_to_anchor=(0.5, -0.2),
        fancybox=False,
        shadow=False,
        ncol=len(labels),
    )
    plt.savefig("figs/tpe-performance.pdf", bbox_inches="tight")


if __name__ == "__main__":
    # for i in range(4):
    #     print(f"Epoch {50 * (i + 1)}")
    #     main(epoch_idx=i, metric="mean", print_detail=False)
    plot_perf_over_time()
