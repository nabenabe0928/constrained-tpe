from typing import Any, Callable, Dict, Optional

import numpy as np

import ConfigSpace as CS

from optimizer.base_optimizer import BaseOptimizer


class RandomSearch(BaseOptimizer):
    def __init__(self, obj_func: Callable, config_space: CS.ConfigurationSpace,
                 resultfile: str, n_init: int = 10, constraints: Dict[str, float] = {},
                 max_evals: int = 100, seed: Optional[int] = None, metric_name: str = 'loss'):

        super().__init__(
            obj_func=obj_func,
            config_space=config_space,
            constraints=constraints,
            resultfile=resultfile,
            n_init=n_init,
            max_evals=max_evals,
            seed=seed,
            metric_name=metric_name
        )

        self._observations = {metric_name: np.array([])}
        self._observations.update({hp_name: np.array([]) for hp_name in self.hp_names})
        self._observations.update({obj_name: np.array([]) for obj_name in self.constraints.keys()})

    def update(self, eval_config: Dict[str, Any], results: Dict[str, float]) -> None:
        for hp_name, val in eval_config.items():
            self._observations[hp_name] = np.append(self._observations[hp_name], val)

        loss = results[self.metric_name]
        self._observations[self.metric_name] = np.append(self._observations[self.metric_name], loss)
        for obj_name in self.constraints.keys():
            self._observations[obj_name] = np.append(self._observations[obj_name], results[obj_name])

    def fetch_observations(self) -> Dict[str, np.ndarray]:
        return self._observations

    def sample(self) -> Dict[str, Any]:
        return self.initial_sample()
