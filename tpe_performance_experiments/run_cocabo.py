import json
import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from tpe_performance_experiments.CoCaBO_code.methods.CoCaBO import CoCaBO
from tpe_performance_experiments.constants import MAX_EVALS, N_INIT, N_SEEDS, RESULTS_DIR, TASK_NAMES
from tpe_performance_experiments.utils import ResultsManager


DUMMY_KEY = "dummy"


def get_config_space(
    search_space: Dict[str, List[Union[int, str, float]]]
) -> Tuple[List[Dict[str, Any]], List[int], List[str]]:
    n_choices_list: List[int] = []
    config_space: List[Dict[str, Any]] = []
    hp_names: List[str] = []
    for append_cat in [True, False]:  # Categoricals come first
        for hp_name, choices in search_space.items():
            is_cat = isinstance(choices[0], str)
            if append_cat != is_cat:
                continue

            hp_names.append(hp_name)
            if is_cat:
                n_choices_list.append(len(choices))

            n_choices = len(choices)
            config_space.append(
                {
                    "name": hp_name,
                    "type": "categorical" if is_cat else "continuous",
                    "domain": tuple(range(n_choices))
                    if is_cat
                    else (0, n_choices - 1e-12),
                }
            )

    # We need at least one continuous param, so we add it to circumvent issues for only categorical settings.
    config_space.append({"name": DUMMY_KEY, "type": "continuous", "domain": (0, 1e-12)})
    hp_names.append(DUMMY_KEY)
    return config_space, n_choices_list, hp_names


def X_to_config(
    ht_list: np.ndarray, X: np.ndarray, hp_names: List[str], n_choices_list: List[int]
) -> Dict[str, Union[int, str]]:
    n_cont, n_cat = 0, 0
    config = {}
    for idx, hp_name in enumerate(hp_names):
        is_cat = idx < len(n_choices_list)
        if is_cat:
            config[hp_name] = ht_list[n_cat]
            n_cat += 1
        else:
            config[hp_name] = int(X[n_cont])
            n_cont += 1

    return config


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    n_seeds = N_SEEDS
    data = {}
    for task_name in TASK_NAMES:
        data[task_name] = []
        for seed in range(n_seeds):
            res_manager = ResultsManager(task_name, seed)
            obj_func = res_manager.obj_func
            search_space = res_manager.search_space
            config_space, n_choices_list, hp_names = get_config_space(
                res_manager.search_space
            )
            opt = CoCaBO(
                objfn=lambda ht_list, X: obj_func(
                    X_to_config(ht_list, X, hp_names, n_choices_list)
                ),
                initN=N_INIT,
                bounds=config_space,
                acq_type="LCB",
                C=n_choices_list,
                kernel_mix=0.5,
            )
            opt.runOptim(budget=MAX_EVALS, seed=seed)
            data[task_name].append(res_manager.loss)
            print(f"{task_name}, seed={seed}: {np.min(res_manager.loss)}")

    with open(f"{RESULTS_DIR}/cocabo.json", mode="w") as f:
        json.dump(data, f, indent=4)
