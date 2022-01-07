from logging import Logger
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

import ConfigSpace as CS

from hypermapper import optimizer

from optimizer.base_optimizer import BaseOptimizer
from util.constants import NumericType
from util.utils import store_results


class HyperMapper(BaseOptimizer):
    def __init__(
        self,
        obj_func: Callable,
        config_space: CS.ConfigurationSpace,
        resultfile: str,
        hypermapper_json: str,
        max_evals: int = 100,
        metric_name: str = 'loss',
        seed: Optional[int] = None,  # random seed is not supported by HyperMapper
        constraints: Dict[str, float] = {}
    ):
        super().__init__(
            obj_func=obj_func,
            config_space=config_space,
            constraints=constraints,
            resultfile=resultfile,
            n_init=0,
            max_evals=max_evals,
            seed=seed,
            metric_name=metric_name
        )

        self._observations = {hp_name: np.array([]) for hp_name in self._hp_names}
        self._observations.update({obj_name: np.array([]) for obj_name in self._constraints.keys()})
        self._observations[self._metric_name] = np.array([])
        self._hypermapper_json = hypermapper_json

        self._t = 0
        self.best_loss = np.inf
        self.best_config = {}

    def update(self, eval_config: Dict[str, Any], results: Dict[str, float]) -> None:
        raise NotImplementedError

    def fetch_observations(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def sample(self) -> Dict[str, Any]:
        raise NotImplementedError

    def optimize(self, logger: Logger) -> Tuple[Dict[str, Any], float]:
        optimizer.optimize(
            parameters_file=self._hypermapper_json,
            black_box_function=self._get_objective_func(self.obj_func, logger=logger)
        )

        store_results(best_config=self.best_config, logger=logger,
                      observations=self._observations, file_name=self.resultfile)

        return self.best_config, self.best_loss

    def _get_objective_func(self, func: Callable, logger: Logger) -> Callable:
        def _wrapper_func(eval_config: Dict[str, Union[NumericType, str]]) -> Dict[str, float]:
            eval_config = {hp_name: int(value) if isinstance(value, str) else value
                           for hp_name, value in eval_config.items()}
            logger.info(f'\nIteration: {self._t + 1}')
            self._t += 1
            eval_config = self._revert_eval_config(eval_config)
            results = func(eval_config)
            loss = results[self._metric_name]

            is_feasible = all(results[obj_name] <= ub for obj_name, ub in self.constraints.items())
            if self.best_loss > loss and is_feasible:
                self.best_loss = loss
                self.best_config = eval_config

            results.pop(self._metric_name)
            logger.info('Cur. loss: {:.5f}, Const.: {}, Cur. Config: {}'.format(loss, results, eval_config))
            logger.info('Best loss: {:.5f}, Best Config: {}'.format(self.best_loss, self.best_config))

            results[self._metric_name] = loss
            for hp_name, val in eval_config.items():
                self._observations[hp_name] = np.append(self._observations[hp_name], val)
            self._observations[self._metric_name] = np.append(self._observations[self._metric_name], loss)
            for obj_name in self.constraints.keys():
                self._observations[obj_name] = np.append(self._observations[obj_name], results[obj_name])

            return {'Valid': is_feasible, self._metric_name: loss}

        return _wrapper_func
