import json
import os
from typing import Dict, List, Tuple, Union

import numpy as np

from tpe_performance_experiments.TuRBO.turbo import Turbo1, TurboM
from tpe_performance_experiments.constants import MAX_EVALS, N_INIT, N_SEEDS, RESULTS_DIR, TASK_NAMES
from tpe_performance_experiments.utils import ResultsManager


def get_config_space(
    search_space: Dict[str, List[Union[int, str, float]]]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    lb, ub = [], []
    hp_names = []
    for hp_name, choices in search_space.items():
        is_cat = isinstance(choices[0], str)
        n_choices = len(choices)
        if is_cat:
            for i in range(n_choices):
                lb.append(0)
                ub.append(1)
                hp_names.append(f"{hp_name}:cat_{i}")
        else:
            hp_names.append(hp_name)
            lb.append(0)
            ub.append(n_choices - 1e-12)

    return np.asarray(lb), np.asarray(ub), hp_names


def X_to_config(
    X: np.ndarray,
    hp_names: List[str],
    search_space: Dict[str, List[Union[int, str, float]]],
) -> Dict[str, Union[int, str]]:
    config = {}
    cur = 0
    while cur < len(hp_names):
        hp_name, x = hp_names[cur], X[cur]
        if not hp_name.endswith("cat_0"):
            config[hp_name] = int(x)
            cur += 1
        else:
            hp_name_prefix = hp_name.split(":cat_0")[0]
            n_choices = len(search_space[hp_name_prefix])
            choice = np.argmax([X[cur + i] for i in range(n_choices)])
            config[hp_name_prefix] = choice
            cur += n_choices
    return config


def choose_turbo_algo(
    f, lb: np.ndarray, ub: np.ndarray, n_trust_regions: int
) -> Union[Turbo1, TurboM]:
    n_init, max_evals = N_INIT, MAX_EVALS
    if n_trust_regions == 1:
        return Turbo1(
            f=f,
            lb=lb,
            ub=ub,
            n_init=n_init,
            max_evals=max_evals,
        )
    elif n_trust_regions == 5:
        return TurboM(
            f=f,
            lb=lb,
            ub=ub,
            n_trust_regions=n_trust_regions,
            n_init=n_init,
            max_evals=max_evals,
        )
    else:
        raise ValueError(
            f"We only check Turbo1 or Turbo5, but tried Turbo{n_trust_regions}"
        )


def main(n_trust_regions: int) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    n_seeds = N_SEEDS
    data = {}
    for task_name in TASK_NAMES:
        data[task_name] = []
        for seed in range(n_seeds):
            res_manager = ResultsManager(task_name, seed)
            obj_func = res_manager.obj_func
            lb, ub, hp_names = get_config_space(res_manager.search_space)

            opt = choose_turbo_algo(
                f=lambda x: obj_func(
                    X_to_config(x, hp_names, res_manager.search_space)
                ),
                lb=lb,
                ub=ub,
                n_trust_regions=n_trust_regions,
            )
            opt.optimize()
            data[task_name].append(res_manager.loss)
            print(f"{task_name}, seed={seed}: {np.min(res_manager.loss)}")

    with open(f"{RESULTS_DIR}/turbo{n_trust_regions}.json", mode="w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main(n_trust_regions=1)
    main(n_trust_regions=5)
