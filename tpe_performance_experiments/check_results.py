from typing import Dict, List, Literal

import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, wilcoxon
from tpe_performance_experiments.constants import TASK_NAMES


def get_data(
    metric: Literal["mean", "median"], epochs: List[int] = [49, 99, 149, 199]
) -> Dict[str, Dict[str, np.ndarray]]:
    data = {}
    opt_names = ["hyperopt", "turbo1", "turbo5", "cocabo", "tpe"]
    for opt_name in opt_names:
        results = json.load(open(f"results-tpe-performance/{opt_name}.json"))
        data[opt_name] = {}
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

            data[opt_name][task_name] = data[opt_name][task_name][epochs]

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


def plot_average_rank(figsize=(9, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 24
    plt.rcParams['mathtext.fontset'] = 'stix'  # The setting of math font
    data = get_data(metric="mean", epochs=list(range(200)))
    opt_names = list(data.keys())
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

    ranks = np.zeros((len(opt_names), 200), dtype=np.float32)
    for task_name in TASK_NAMES:
        ranks += rankdata([data[opt_name][task_name] for opt_name in opt_names], axis=0)

    ranks /= len(TASK_NAMES)
    dx = np.arange(1, 201)
    lines, labels = [], []
    for idx, opt_name in enumerate(opt_names):
        line, = ax.plot(dx, ranks[idx], label=LABEL_DICT[opt_name], color=COLOR_DICT[opt_name])
        lines.append(line)
        ax.set_xlabel('# of config evaluations')
        ax.set_ylabel('Average rank')
        labels.append(LABEL_DICT[opt_name])

    ax.grid()
    ax.legend(
        handles=lines,
        labels=labels,
        loc='upper center',
        fontsize=24,
        bbox_to_anchor=(0.5, -0.2),
        fancybox=False,
        shadow=False,
        ncol=len(labels)
    )
    plt.savefig('figs/tpe-performance.pdf', bbox_inches='tight')


if __name__ == "__main__":
    for i in range(4):
        print(f"Epoch {50 * (i + 1)}")
        main(epoch_idx=i, metric="mean", print_detail=False)
