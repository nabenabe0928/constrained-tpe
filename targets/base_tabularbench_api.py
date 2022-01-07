from abc import abstractmethod, ABCMeta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import json

import ConfigSpace as CS

from util.constants import DOMAIN_SIZE_CHOICES
from util.utils import get_config_space, ParameterSettings


class BaseTabularBenchAPI(metaclass=ABCMeta):
    def __init__(
        self,
        hp_module_path: str,
        dataset_name: str,
        constraints: List[Enum],
        seed: Optional[int] = None,
        feasible_domain_ratio: Optional[int] = None,
        cheap_metrics: List[str] = []
    ):
        self._rng = np.random.RandomState(seed)
        self._oracle: Optional[float] = None
        js = open(f'{hp_module_path}/params.json')
        searching_space: Dict[str, ParameterSettings] = json.load(js)
        self._config_space = get_config_space(searching_space, hp_module_path='.'.join(hp_module_path.split('/')))
        self._constraints = self._get_constraints(
            hp_module_path=hp_module_path,
            dataset_name=dataset_name,
            constraints=constraints,
            feasible_domain_ratio=feasible_domain_ratio
        )
        self._cheap_metrics = cheap_metrics
        self._expensive_metrics: List[str] = []

    def is_satisfied_constraints(self, results: Dict[str, float]) -> bool:
        return all(
            results[obj_name] <= lower_bound
            for obj_name, lower_bound in self.constraints.items()
            if lower_bound is not None
        )

    def _get_non_constraint(
        self,
        constraints: List[Enum],
        feasible_domain_ratio: Optional[int],
        constraint_dict: Dict[str, Union[List[float], float]]
    ) -> Dict[str, float]:

        if len(constraints) > 1:
            raise ValueError('Constraints have multiple values, but include the constraint of None.')
        elif feasible_domain_ratio is not None:
            raise ValueError(
                'feasible_domain_ratio for non-constraint optimization must be None, '
                f'but got {feasible_domain_ratio}'
            )

        if constraint_dict.get('oracle', None) is not None:
            # In case the value is not yet available
            assert not isinstance(constraint_dict['oracle'], list)
            self._oracle = constraint_dict['oracle']

        return {}

    def _get_single_constraint(
        self,
        constraint: Enum,
        feasible_domain_ratio: int,
        constraint_dict: Dict[str, Union[List[float], float]]
    ) -> Dict[str, float]:

        constraint_vals = constraint_dict[constraint.name]

        assert isinstance(constraint_vals, dict)
        constraint_lower_bound = constraint_vals[str(feasible_domain_ratio)]

        return {constraint.name: constraint_lower_bound}

    def _get_constraints(
        self,
        hp_module_path: str,
        dataset_name: str,
        constraints: List[Enum],
        feasible_domain_ratio: Optional[int]
    ) -> Dict[str, float]:

        js = open(f'{hp_module_path}/constraints.json')
        constraint_dict = json.load(js)[dataset_name]

        if any([c.value is None for c in constraints]):
            return self._get_non_constraint(
                constraints=constraints,
                feasible_domain_ratio=feasible_domain_ratio,
                constraint_dict=constraint_dict
            )

        if feasible_domain_ratio not in DOMAIN_SIZE_CHOICES:
            raise ValueError(f'feasible_domain_ratio must be in {DOMAIN_SIZE_CHOICES}, '
                             f'but got {feasible_domain_ratio}')

        assert isinstance(feasible_domain_ratio, int)
        _constraints = {}
        for constraint in constraints:
            _constraints.update(
                self._get_single_constraint(
                    constraint=constraint,
                    feasible_domain_ratio=feasible_domain_ratio,
                    constraint_dict=constraint_dict
                )
            )

        constraint_suffix = ','.join([c.name for c in constraints])
        print(constraints)
        self._oracle = constraint_dict[f'oracle::{constraint_suffix}'].get(str(feasible_domain_ratio), None)

        return _constraints

    def cheap_objective_func(self, config: Dict[str, Any], budget: Dict[str, Any] = {}) -> Dict[str, float]:
        """
        Args:
            config (Dict[str, Any]):
                The dict of the configuration and the corresponding value
            budget (Dict[str, Any]):
                The budget information

        Returns:
            results (Dict[str, float]):
                A pair of loss or constraint value and its name.
        """
        if len(self.cheap_metrics) == 0:
            raise ValueError('The length of cheap_metrics must be positive.')

        results = self.objective_func(config=config, budget=budget)

        if len(self.expensive_metrics) == 0:
            self._expensive_metrics = list(set(results.keys()) - set(self.cheap_metrics))

        for metric_name in self.expensive_metrics:
            results.pop(metric_name)

        return results

    def find_oracle(self) -> Tuple[float, float]:
        """
        Find the oracle based on the constraint given in this instance.

        Returns:
            best_oracle, worst_oracle (Tuple[float, float]):
                The best and worst possible loss value available in this benchmark.
                It considers each seed independently.
                Note that worst oracle also satisfies given constraints.
        """
        loss_vals = self.find_satisfactory_losses()
        return loss_vals.min(), loss_vals.max()

    @abstractmethod
    def find_satisfactory_losses(self) -> np.ndarray:
        """
        Find the loss values that satisfy constraints given in this instance.

        Returns:
            losses (np.ndarray):
                The satisfactory loss values available in this benchmark.
                It considers each seed independently.
        """
        raise NotImplementedError

    @abstractmethod
    def objective_func(self, config: Dict[str, Any], budget: Dict[str, Any] = {}) -> Dict[str, float]:
        """
        Args:
            config (Dict[str, Any]):
                The dict of the configuration and the corresponding value
            budget (Dict[str, Any]):
                The budget information

        Returns:
            results (Dict[str, float]):
                A pair of loss or constraint value and its name.
        """
        raise NotImplementedError

    @property
    def rng(self) -> np.random.RandomState:
        return self._rng

    @property
    def config_space(self) -> CS.ConfigurationSpace:
        """ The config space of the child tabular benchmark """
        return self._config_space

    @property
    def constraints(self) -> Dict[str, float]:
        """ The constraints of the child tabular benchmark """
        return self._constraints

    @property
    def cheap_metrics(self) -> List[str]:
        """ The name of cheap metrics """
        return self._cheap_metrics

    @property
    def expensive_metrics(self) -> List[str]:
        """ The name of expensive metrics """
        return self._expensive_metrics

    @property
    def oracle(self) -> Optional[float]:
        """The global best performance given a constraint"""
        return self._oracle

    @property
    @abstractmethod
    def data(self) -> Any:
        """ API for the target dataset """
        raise NotImplementedError
