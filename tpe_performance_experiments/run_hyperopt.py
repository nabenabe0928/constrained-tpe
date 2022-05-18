import json
import os
from typing import Any, Dict, List, Union

from hyperopt import fmin, hp, tpe
from tpe_performance_experiments.constants import MAX_EVALS, N_SEEDS, RESULTS_DIR, TASK_NAMES
from tpe_performance_experiments.utils import ResultsManager


def get_config_space(
    search_space: Dict[str, List[Union[int, str, float]]]
) -> Dict[str, Any]:
    return {
        hp_name: hp.choice(hp_name, list(range(len(choices))))
        if isinstance(choices[0], str)
        else hp.quniform(hp_name, low=0, high=len(choices) - 1, q=1)
        for hp_name, choices in search_space.items()
    }


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    n_seeds = N_SEEDS
    data = {}
    for task_name in TASK_NAMES:
        data[task_name] = []
        for seed in range(n_seeds):
            res_manager = ResultsManager(task_name, seed)
            obj_func = res_manager.obj_func
            config_space = get_config_space(res_manager.search_space)
            best = fmin(
                fn=obj_func, space=config_space, algo=tpe.suggest, max_evals=MAX_EVALS
            )
            data[task_name].append(res_manager.loss)
            print(f"{task_name}, seed={seed}: {best}")

    with open(f"{RESULTS_DIR}/hyperopt.json", mode="w") as f:
        json.dump(data, f, indent=4)
