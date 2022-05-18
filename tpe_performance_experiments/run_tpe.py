import json
import os
from typing import Dict, List, Union

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from tpe.optimizer import TPEOptimizer
from tpe_performance_experiments.constants import MAX_EVALS, N_INIT, N_SEEDS, RESULTS_DIR, TASK_NAMES
from tpe_performance_experiments.utils import ResultsManager


def get_config_space(
    search_space: Dict[str, List[Union[int, str, float]]]
) -> CS.ConfigurationSpace:
    config_space = CS.ConfigurationSpace()
    for hp_name, choices in search_space.items():
        if isinstance(choices[0], str):
            hp = CSH.CategoricalHyperparameter(
                hp_name, [str(idx) for idx in range(len(choices))]
            )
        else:
            hp = CSH.OrdinalHyperparameter(
                hp_name,
                list(range(len(choices))),
                meta={"lower": 0, "upper": len(choices) - 1},
            )
        config_space.add_hyperparameter(hp)

    return config_space


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
            opt = TPEOptimizer(
                obj_func=lambda config: ({"loss": obj_func(config)}, 0.0),
                config_space=config_space,
                max_evals=MAX_EVALS,
                n_init=N_INIT,
                only_requirements=True,
                seed=seed,
            )
            data[task_name].append(res_manager.loss)
            best = opt.optimize()
            print(f"{task_name}, seed={seed}: {best}")

    with open(f"{RESULTS_DIR}/tpe.json", mode="w") as f:
        json.dump(data, f, indent=4)
