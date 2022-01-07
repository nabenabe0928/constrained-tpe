from logging import Logger
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

from ax.service.managed_loop import optimize

import ConfigSpace as CS

from optimizer.base_optimizer import BaseOptimizer
from util.constants import NumericType
from util.utils import store_results


class ConstraintBayesianOptimization(BaseOptimizer):
    def __init__(
        self,
        obj_func: Callable,
        config_space: CS.ConfigurationSpace,
        resultfile: str,
        max_evals: int = 100,
        seed: Optional[int] = None,
        metric_name: str = 'loss',
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

        self._seed = seed
        self._observations = {hp_name: np.array([]) for hp_name in self._hp_names}
        self._observations.update({obj_name: np.array([]) for obj_name in self._constraints.keys()})
        self._observations[self._metric_name] = np.array([])

        self._t = 0
        self.best_loss = np.inf
        self.best_config = {}

    def update(self, eval_config: Dict[str, Any], results: Dict[str, float]) -> None:
        raise NotImplementedError

    def fetch_observations(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def sample(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _config_to_sample(self, hp_name: str) -> Dict[str, Any]:
        # Parameters definition: https://ax.dev/docs/core.html
        is_ordinal = self.is_ordinals[hp_name]
        is_categorical = self.is_categoricals[hp_name]
        config = self.config_space.get_hyperparameter(hp_name)
        if is_categorical:
            choices = list(range(len(config.choices)))
            return dict(
                name=hp_name,
                type='choice',
                values=choices,
                value_type='int',
                is_ordered=False
            )
        elif is_ordinal:
            info = config.meta
            log = info.get('log', False)
            choices = np.log(config.sequence) if log else np.array(config.sequence)
            return dict(
                name=hp_name,
                type='choice',
                values=choices.tolist(),
                value_type='float',
                is_ordered=True
            )
        else:
            lb, ub, log = config.lower, config.upper, config.log
            return dict(
                name=hp_name,
                type='range',
                value_type='float',
                bounds=[
                    np.log(lb) if log else lb,
                    np.log(ub) if log else ub
                ]
            )

    def optimize(self, logger: Logger) -> Tuple[Dict[str, Any], float]:
        obj_func = self._get_objective_func(self.obj_func, logger)
        optimize(
            parameters=[
                self._config_to_sample(hp_name)
                for hp_name in self.hp_names
            ],
            outcome_constraints=[
                f'{obj_name} <= {ub}'
                for obj_name, ub in self.constraints.items()
            ],
            evaluation_function=obj_func,
            objective_name=self._metric_name,
            minimize=True,
            total_trials=self.max_evals,
            random_seed=self._seed
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

            if self.best_loss > loss and all(results[obj_name] <= ub for obj_name, ub in self.constraints.items()):
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

            return results

        return _wrapper_func
