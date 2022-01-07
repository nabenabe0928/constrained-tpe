from abc import abstractmethod, ABCMeta
from logging import Logger
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import ConfigSpace as CS

from util.constants import NumericType
from util.utils import (
    get_random_sample,
    revert_eval_config,
    store_results
)


class BaseOptimizer(metaclass=ABCMeta):
    def __init__(self, obj_func: Callable, config_space: CS.ConfigurationSpace,
                 constraints: Dict[str, float], resultfile: str,
                 n_init: int = 10, max_evals: int = 100,
                 seed: Optional[int] = None, metric_name: str = 'loss'):
        """
        Attributes:
            rng (np.random.RandomState): random state to maintain the reproducibility
            resultfile (str): The name of the result file to output in the end
            n_init (int): The number of random sampling before using TPE
            obj_func (Callable): The objective function
            hp_names (List[str]): The list of hyperparameter names
            metric_name (str): The name of the metric (or objective function value)
            constraints (Dict[str, float]): The pair of constraint name and its lower bound
            observations (Dict[str, Any]): The storage of the observations
            config_space (CS.ConfigurationSpace): The searching space of the task
            is_categoricals (Dict[str, bool]): Whether the given hyperparameter is categorical
            is_ordinals (Dict[str, bool]): Whether the given hyperparameter is ordinal
        """

        self._rng = np.random.RandomState(seed)
        self._n_init, self._max_evals = n_init, max_evals
        self.resultfile = resultfile
        self._obj_func = obj_func
        self._hp_names = list(config_space._hyperparameters.keys())
        self._metric_name = metric_name
        self._constraints = constraints

        self._config_space = config_space
        self._is_categoricals = {
            hp_name: self._config_space.get_hyperparameter(hp_name).__class__.__name__ == 'CategoricalHyperparameter'
            for hp_name in self._hp_names
        }
        self._is_ordinals = {
            hp_name: self._config_space.get_hyperparameter(hp_name).__class__.__name__ == 'OrdinalHyperparameter'
            for hp_name in self._hp_names
        }

    def optimize(self, logger: Logger) -> Tuple[Dict[str, Any], float]:
        """
        Optimize obj_func using TPE Sampler and store the results in the end.

        Args:
            logger (Logger): The logging to write the intermediate results

        Returns:
            best_config (Dict[str, Any]): The configuration that has the best loss
            best_loss (float): The best loss value during the optimization
        """

        best_config, best_loss, t = {}, np.inf, 0

        while True:
            logger.info(f'\nIteration: {t + 1}')
            eval_config = self.initial_sample() if t < self.n_init else self.sample()

            results = self.obj_func(eval_config)
            loss = results[self.metric_name]
            self.update(eval_config=eval_config, results=results)

            if (
                best_loss > loss
                and all(results[obj_name] <= ub for obj_name, ub in self.constraints.items())
            ):
                best_loss = loss
                best_config = eval_config

            results.pop(self.metric_name)
            logger.info('Cur. loss: {:.5f}, Const.: {}, Cur. Config: {}'.format(loss, results, eval_config))
            logger.info('Best loss: {:.5f}, Best Config: {}'.format(best_loss, best_config))
            t += 1

            if t >= self.max_evals:
                break

        observations = self.fetch_observations()
        store_results(best_config=best_config, logger=logger,
                      observations=observations, file_name=self.resultfile)

        return best_config, best_loss

    @abstractmethod
    def update(self, eval_config: Dict[str, Any], results: Dict[str, float]) -> None:
        """
        Update of the child sampler.

        Args:
            eval_config (Dict[str, Any]): The configuration to be evaluated
            results (Dict[str, float]): The dict of loss value and constraints of the eval_config
        """
        raise NotImplementedError

    @abstractmethod
    def fetch_observations(self) -> Dict[str, np.ndarray]:
        """
        Fetch observations of this optimization.

        Returns:
            observations (Dict[str, np.ndarray]):
                observations of this optimization.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self) -> Dict[str, Any]:
        """
        Sample a configuration using a child class instance sampler

        Returns:
            eval_config (Dict[str, Any]): A sampled configuration
        """
        raise NotImplementedError

    def initial_sample(self) -> Dict[str, Any]:
        """
        Sampling method up to n_init configurations

        Returns:
            samples (Dict[str, Any]):
                Typically randomly sampled configurations

        """
        eval_config = {hp_name: self._get_random_sample(hp_name=hp_name) for hp_name in self.hp_names}
        return self._revert_eval_config(eval_config=eval_config)

    def _get_random_sample(self, hp_name: str) -> NumericType:
        return get_random_sample(hp_name=hp_name, rng=self.rng, config_space=self.config_space,
                                 is_categorical=self.is_categoricals[hp_name],
                                 is_ordinal=self.is_ordinals[hp_name])

    def _revert_eval_config(self, eval_config: Dict[str, NumericType]) -> Dict[str, Any]:
        return revert_eval_config(eval_config=eval_config, config_space=self.config_space,
                                  is_categoricals=self.is_categoricals, is_ordinals=self.is_ordinals,
                                  hp_names=self.hp_names)

    @property
    def config_space(self) -> CS.ConfigurationSpace:
        return self._config_space

    @property
    def hp_names(self) -> List[str]:
        return self._hp_names

    @property
    def metric_name(self) -> str:
        return self._metric_name

    @property
    def is_categoricals(self) -> Dict[str, bool]:
        return self._is_categoricals

    @property
    def is_ordinals(self) -> Dict[str, bool]:
        return self._is_ordinals

    @property
    def rng(self) -> np.random.RandomState:
        return self._rng

    @property
    def max_evals(self) -> int:
        return self._max_evals

    @property
    def n_init(self) -> int:
        return self._n_init

    @property
    def obj_func(self) -> Callable:
        return self._obj_func

    @property
    def constraints(self) -> Dict[str, float]:
        return self._constraints
