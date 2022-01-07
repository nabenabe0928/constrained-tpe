from logging import Logger
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import ConfigSpace as CS

import optuna

from optimizer.base_optimizer import BaseOptimizer
from util.constants import NumericType
from util.utils import store_results


class NSGA2(BaseOptimizer):
    def __init__(
        self,
        obj_func: Callable,
        config_space: CS.ConfigurationSpace,
        resultfile: str,
        population_size=8,
        max_evals: int = 100,
        metric_name: str = 'loss',
        seed: Optional[int] = None,
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
        sampler = optuna.samplers.NSGAIISampler(
            population_size=8, seed=seed,
            constraints_func=self.constraints_func
        )

        self.study = optuna.create_study(
            directions=["minimize"],
            sampler=sampler,
            pruner=optuna.pruners.NopPruner
        )
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

    def constraints_func(self, frozen_trial: optuna.trial.FrozenTrial) -> List[float]:
        return [
            self.recent_results[obj_name] - ub
            for obj_name, ub in self.constraints.items()
        ]

    def _get_eval_config_from_trial(self, trial: optuna.trial.Trial) -> Dict[str, NumericType]:
        eval_config = {}
        for hp_name in self.hp_names:
            is_categorical, is_ordinal = self.is_categoricals[hp_name], self.is_ordinals[hp_name]
            config = self.config_space.get_hyperparameter(hp_name)
            if is_categorical:
                choices = config.choices
                eval_config[hp_name] = trial.suggest_categorical(
                    name=hp_name, choices=list(range(len(choices)))
                )
            elif is_ordinal:
                info = config.meta
                lb, ub = info['lower'], info['upper']
                log = info.get('log', False)
                eval_config[hp_name] = trial.suggest_float(
                    name=hp_name,
                    low=np.log(lb) if log else lb,
                    high=np.log(ub) if log else ub
                )
            else:
                lb, ub, log = config.lower, config.upper, config.log
                eval_config[hp_name] = trial.suggest_float(
                    name=hp_name,
                    low=np.log(lb) if log else lb,
                    high=np.log(ub) if log else ub
                )

        return eval_config

    def optimize(self, logger: Logger) -> Tuple[Dict[str, Any], float]:
        self.study.optimize(
            self._get_objective_func(self.obj_func, logger=logger),
            n_trials=self.max_evals
        )

        store_results(best_config=self.best_config, logger=logger,
                      observations=self._observations, file_name=self.resultfile)

        return self.best_config, self.best_loss

    def _get_objective_func(self, func: Callable, logger: Logger) -> Callable:
        def _wrapper_func(trial: optuna.trial.Trial) -> Dict[str, float]:
            logger.info(f'\nIteration: {self._t + 1}')
            self._t += 1
            eval_config = self._get_eval_config_from_trial(trial)
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
            self.recent_results = results

            for hp_name, val in eval_config.items():
                self._observations[hp_name] = np.append(self._observations[hp_name], val)
            self._observations[self._metric_name] = np.append(self._observations[self._metric_name], loss)
            for obj_name in self.constraints.keys():
                self._observations[obj_name] = np.append(self._observations[obj_name], results[obj_name])

            return loss

        return _wrapper_func
