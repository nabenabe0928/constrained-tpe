from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np

import ConfigSpace as CS

from optimizer.base_optimizer import BaseOptimizer
from optimizer.parzen_estimator.loglikelihoods import compute_config_loglikelihoods
from optimizer.parzen_estimator.parzen_estimator import (
    CategoricalParzenEstimator,
    NumericalParzenEstimator,
    build_categorical_parzen_estimator,
    build_numerical_parzen_estimator
)
from util.constants import (
    CategoricalHPType,
    EPS,
    NumericType,
    NumericalHPType,
    config2type,
    default_percentile_maker,
    default_threshold_maker,
    default_weights
)


ParzenEstimatorType = Union[NumericalParzenEstimator, CategoricalParzenEstimator]
HPType = Union[CategoricalHPType, NumericalHPType]


class PercentileFuncMaker(Protocol):
    def __call__(self, **kwargs: Dict[str, Any]) -> Callable[[np.ndarray], int]:
        ...


class TreeStructuredParzenEstimator:
    def __init__(self, config_space: CS.ConfigurationSpace,
                 percentile_func: Callable[[np.ndarray], int],
                 weight_func: Callable[[int, int], np.ndarray],
                 n_ei_candidates: int, metric_name: str = 'loss',
                 naive_mode: bool = False,
                 min_bandwidth_factor: float = 1e-2, seed: Optional[int] = None):
        """
        Attributes:
            rng (np.random.RandomState): random state to maintain the reproducibility
            n_ei_candidates (int): The number of samplings to optimize the EI value
            config_space (CS.ConfigurationSpace): The searching space of the task
            hp_names (List[str]): The list of hyperparameter names
            metric_name (str): The name of the metric (or objective function value)
            observations (Dict[str, Any]): The storage of the observations
            sorted_observations (Dict[str, Any]): The storage of the observations sorted based on loss
            min_bandwidth_factor (float): The minimum bandwidth for numerical parameters
            is_categoricals (Dict[str, bool]): Whether the given hyperparameter is categorical
            is_ordinals (Dict[str, bool]): Whether the given hyperparameter is ordinal
            percentile_func (Callable[[np.ndarray], int]):
                The function that returns the number of a better group based on the total number of evaluations.
            weight_func (Callable[[int, int], np.ndarray]):
                The function that returns the coefficients of each kernel.
        """
        self._rng = np.random.RandomState(seed)
        self._n_ei_candidates = n_ei_candidates
        self._config_space = config_space
        self._hp_names = list(config_space._hyperparameters.keys())
        self._metric_name = metric_name
        self._n_lower, self._percentile = 0, 0
        self._min_bandwidth_factor = min_bandwidth_factor
        self._naive_mode = naive_mode

        self._observations = {hp_name: np.array([]) for hp_name in self._hp_names}
        self._sorted_observations = {hp_name: np.array([]) for hp_name in self._hp_names}
        self._observations[self.metric_name] = np.array([])
        self._sorted_observations[self.metric_name] = np.array([])
        self._sorted_is_satisfied = np.array([], dtype=np.bool8)

        self._weight_func = weight_func
        self._percentile_func = percentile_func

        self._is_categoricals = {
            hp_name: self._config_space.get_hyperparameter(hp_name).__class__.__name__ == 'CategoricalHyperparameter'
            for hp_name in self._hp_names
        }

        self._is_ordinals = {
            hp_name: self._config_space.get_hyperparameter(hp_name).__class__.__name__ == 'OrdinalHyperparameter'
            for hp_name in self._hp_names
        }

    def apply_knowledge_augmentation(self, observations: Dict[str, np.ndarray]) -> None:
        if self.observations[self.metric_name].size != 0:
            raise ValueError('Knowledge augmentation must be applied before the optimization.')

        self._observations = {hp_name: vals.copy() for hp_name, vals in observations.items()}
        order = np.argsort(self.observations[self.metric_name])
        self._sorted_observations = {
            hp_name: observations[order]
            for hp_name, observations in self.observations.items()
        }

    def _compute_min_num_for_lower_observations(self) -> int:
        n_lower = self.percentile_func(self._sorted_observations[self.metric_name])
        n_satisfied = self._sorted_is_satisfied.cumsum()
        # At least, lower observations should include n_lower satisfactory observations.
        idx = np.searchsorted(n_satisfied, n_lower, side='left') + 1
        return min(idx, n_satisfied.size)

    def update_observations(
        self,
        eval_config: Dict[str, NumericType],
        loss: float,
        is_satisfied: Optional[bool] = None
    ) -> None:
        """
        Update the observations for the TPE construction

        Args:
            eval_config (Dict[str, NumericType]): The configuration to evaluate (after conversion)
            loss (float): The loss value as a result of the evaluation
            is_satisfied (Optional[bool]):
                Whether the latest observation satisfied the constraints.
                If None, no update regarding this happens.
                Basically, we perform this special update only for the objective.
        """
        sorted_losses, losses = self.sorted_observations[self.metric_name], self.observations[self.metric_name]
        insert_loc = np.searchsorted(sorted_losses, loss, side='right')
        self._observations[self.metric_name] = np.append(losses, loss)
        self._sorted_observations[self.metric_name] = np.insert(sorted_losses, insert_loc, loss)

        if is_satisfied is not None:
            self._sorted_is_satisfied = np.insert(self._sorted_is_satisfied, insert_loc, is_satisfied)
            min_num = self._compute_min_num_for_lower_observations() if not self._naive_mode else 1
            self._n_lower = self.percentile_func(self._sorted_observations[self.metric_name], min_num=min_num)
        else:
            self._n_lower = self.percentile_func(self._sorted_observations[self.metric_name])

        self._percentile = self._n_lower / self.observations[self.metric_name].size

        observations, sorted_observations = self._observations, self._sorted_observations
        for hp_name in self.hp_names:
            is_categorical = self.is_categoricals[hp_name]
            config = self.config_space.get_hyperparameter(hp_name)
            config_type = config.__class__.__name__
            val = eval_config[hp_name]

            if is_categorical:
                observations[hp_name] = np.append(observations[hp_name], val)
                if sorted_observations[hp_name].size == 0:  # cannot cast str to float!
                    sorted_observations[hp_name] = np.array([val], dtype='U32')
                else:
                    sorted_observations[hp_name] = np.insert(sorted_observations[hp_name], insert_loc, val)
            else:
                dtype = config2type[config_type]
                observations[hp_name] = np.append(observations[hp_name], val).astype(dtype)
                sorted_observations[hp_name] = np.insert(sorted_observations[hp_name], insert_loc, val).astype(dtype)

    def get_config_candidates(self) -> List[np.ndarray]:
        """
        Since we compute the probability improvement of each objective independently,
        we need to sample the configurations in advance.

        Returns:
            config_cands (List[np.ndarray]): arrays of candidates in each dimension
        """
        config_cands = []
        n_lower = self.n_lower

        for hp_name in self.hp_names:
            lower_vals = self.sorted_observations[hp_name][:n_lower]
            empty = np.array([lower_vals[0]])

            is_categorical = self.is_categoricals[hp_name]
            pe_lower, _ = self._get_parzen_estimator(lower_vals=lower_vals, upper_vals=empty,
                                                     hp_name=hp_name, is_categorical=is_categorical)

            config_cands.append(pe_lower.sample(self.rng, self.n_ei_candidates))

        return config_cands

    def _compute_basis_loglikelihoods(self, hp_name: str, samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the log likelihood of each basis of the provided hyperparameter

        Args:
            hp_name (str): The name of a hyperparameter
            samples (np.ndarray): The samples to compute the basis loglikelihoods

        Returns:
            basis_loglikelihoods (np.ndarray):
                The shape is (n_basis, n_samples).
        """
        is_categorical = self.is_categoricals[hp_name]
        sorted_observations = self.sorted_observations[hp_name]
        n_lower = self.n_lower

        # split observations
        lower_vals = sorted_observations[:n_lower]
        upper_vals = sorted_observations[n_lower:]

        pe_lower, pe_upper = self._get_parzen_estimator(lower_vals=lower_vals, upper_vals=upper_vals,
                                                        hp_name=hp_name, is_categorical=is_categorical)

        return pe_lower.basis_loglikelihood(samples), pe_upper.basis_loglikelihood(samples)

    def compute_config_loglikelihoods(self, config_cands: List[np.ndarray]
                                      ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the probability improvement given configurations

        Args:
            config_cands (List[np.ndarray]):
                The list of candidate values for each dimension.
                The length is the number of dimensions and
                each array has the length of n_ei_candidates.

        Returns:
            config_ll_lower, config_ll_upper (Tuple[np.ndarray]):
                The loglikelihoods of each configuration in
                the good group or bad group.
                The shape is (n_ei_candidates, ) for each.
        """
        dim = len(self.hp_names)
        n_evals = self.sorted_observations[self.metric_name].size
        n_lower = self.n_lower

        n_candidates = config_cands[0].size
        basis_loglikelihoods_lower = np.zeros((dim, n_lower + 1, n_candidates))
        weights_lower = self.weight_func(n_lower + 1)
        basis_loglikelihoods_upper = np.zeros((dim, n_evals - n_lower + 1, n_candidates))
        weights_upper = self.weight_func(n_evals - n_lower + 1)

        for dim, (hp_name, samples) in enumerate(zip(self.hp_names, config_cands)):
            bll_lower, bll_upper = self._compute_basis_loglikelihoods(hp_name=hp_name, samples=samples)
            basis_loglikelihoods_lower[dim] += bll_lower
            basis_loglikelihoods_upper[dim] += bll_upper

        config_ll_lower = compute_config_loglikelihoods(basis_loglikelihoods_lower, weights_lower)
        config_ll_upper = compute_config_loglikelihoods(basis_loglikelihoods_upper, weights_upper)

        return config_ll_lower, config_ll_upper

    def compute_probability_improvement(self, config_cands: List[np.ndarray]) -> np.ndarray:
        """
        Compute the probability improvement given configurations

        Args:
            config_cands (List[np.ndarray]):
                The list of candidate values for each dimension.
                The length is the number of dimensions and
                each array has the length of n_ei_candidates.

        Returns:
            config_ll_ratio (np.ndarray):
                The log of the likelihood ratios of each configuration.
                The shape is (n_ei_candidates, )

        Note:
            We need to consider the gamma unlike the normal TPE implementation.
            p(y < y*) propto (r + (1 - r)g(x)/l(x))^-1
                      = (exp(log(r)) + exp(log(1 - r) + log(g(x)/l(x))))^-1
        """
        config_ll_lower, config_ll_upper = self.compute_config_loglikelihoods(config_cands=config_cands)
        if self._naive_mode:
            return config_ll_lower - config_ll_upper
        else:
            first_term = np.log(self.percentile + EPS)
            second_term = np.log(1. - self.percentile + EPS) + config_ll_upper - config_ll_lower
            pi = - np.logaddexp(first_term, second_term)
            return pi

    def _get_parzen_estimator(self, lower_vals: np.ndarray, upper_vals: np.ndarray, hp_name: str,
                              is_categorical: bool) -> Tuple[ParzenEstimatorType, ParzenEstimatorType]:
        """
        Construct parzen estimators for the lower and the upper groups and return them

        Args:
            lower_vals (np.ndarray): The array of the values in the lower group
            upper_vals (np.ndarray): The array of the values in the upper group
            hp_name (str): The name of the hyperparameter
            is_categorical (bool): Whether the given hyperparameter is categorical

        Returns:
            pe_lower (ParzenEstimatorType): The parzen estimator for the lower group
            pe_upper (ParzenEstimatorType): The parzen estimator for the upper group
        """
        config = self.config_space.get_hyperparameter(hp_name)
        config_type = config.__class__.__name__
        is_ordinal = self.is_ordinals[hp_name]
        parzen_estimator_args = dict(config=config, weight_func=self.weight_func)

        if is_categorical:
            pe_lower = build_categorical_parzen_estimator(vals=lower_vals, **parzen_estimator_args)
            pe_upper = build_categorical_parzen_estimator(vals=upper_vals, **parzen_estimator_args)
        else:
            parzen_estimator_args.update(dtype=config2type[config_type], is_ordinal=is_ordinal)
            min_bandwidth_factor = 1.0 / len(config.sequence) if is_ordinal else self.min_bandwidth_factor
            parzen_estimator_args.update(min_bandwidth_factor=min_bandwidth_factor)
            pe_lower = build_numerical_parzen_estimator(vals=lower_vals, **parzen_estimator_args)
            pe_upper = build_numerical_parzen_estimator(vals=upper_vals, **parzen_estimator_args)

        return pe_lower, pe_upper

    @property
    def config_space(self) -> CS.ConfigurationSpace:
        return self._config_space

    @property
    def observations(self) -> Dict[str, np.ndarray]:
        return self._observations

    @property
    def sorted_observations(self) -> Dict[str, np.ndarray]:
        return self._sorted_observations

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
    def n_ei_candidates(self) -> int:
        return self._n_ei_candidates

    @property
    def n_lower(self) -> int:
        return self._n_lower

    @property
    def hp_names(self) -> List[str]:
        return self._hp_names

    @property
    def metric_name(self) -> str:
        return self._metric_name

    @property
    def percentile(self) -> float:
        return self._percentile

    @property
    def min_bandwidth_factor(self) -> float:
        return self._min_bandwidth_factor

    @property
    def weight_func(self) -> Callable:
        return self._weight_func

    @property
    def percentile_func(self) -> Callable:
        return self._percentile_func


class TPEOptimizer(BaseOptimizer):
    def __init__(self, obj_func: Callable, config_space: CS.ConfigurationSpace,
                 resultfile: str, n_init: int = 10, constraints: Dict[str, float] = {},
                 max_evals: int = 100, seed: Optional[int] = None, metric_name: str = 'loss',
                 n_ei_candidates: int = 24,
                 naive_mode: bool = False,
                 percentile_func_maker: PercentileFuncMaker = default_percentile_maker,
                 threshold_func_maker: PercentileFuncMaker = default_threshold_maker,
                 weight_func: Callable[[int, int], np.ndarray] = default_weights):

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

        self.tpe_samplers = {
            metric_name: TreeStructuredParzenEstimator(
                config_space=config_space,
                metric_name=metric_name,
                n_ei_candidates=n_ei_candidates,
                seed=seed,
                percentile_func=percentile_func_maker(),
                weight_func=weight_func,
                naive_mode=naive_mode,
            )
        }

        for obj_name, upper_bound in constraints.items():
            self.tpe_samplers[obj_name] = TreeStructuredParzenEstimator(
                config_space=config_space,
                metric_name=obj_name,
                n_ei_candidates=n_ei_candidates,
                seed=seed,
                percentile_func=threshold_func_maker(upper_bound=upper_bound),
                weight_func=weight_func,
                naive_mode=naive_mode,
            )

    def update(self, eval_config: Dict[str, Any], results: Dict[str, float]) -> None:
        # Adapt here
        loss = results[self.metric_name]
        is_satisfied = all(results[obj_name] <= ub for obj_name, ub in self.constraints.items())
        self.tpe_samplers[self.metric_name].update_observations(
            eval_config=eval_config,
            loss=loss,
            is_satisfied=is_satisfied
        )

        for obj_name in self.constraints.keys():
            self.tpe_samplers[obj_name].update_observations(eval_config=eval_config, loss=results[obj_name])

    def fetch_observations(self) -> Dict[str, np.ndarray]:
        observations = self.tpe_samplers[self.metric_name].observations
        for obj_name, val in self.constraints.items():
            observations[obj_name] = self.tpe_samplers[obj_name].observations[obj_name]

        return observations

    def sample(self) -> Dict[str, Any]:
        """
        Sample a configuration using tree-structured parzen estimator (TPE)

        Returns:
            eval_config (Dict[str, Any]): A sampled configuration from TPE
        """
        config_cands = self.tpe_samplers[self.metric_name].get_config_candidates()

        for obj_name in self.constraints.keys():
            if self.constraints[obj_name] == np.inf:
                continue

            configs = self.tpe_samplers[obj_name].get_config_candidates()
            config_cands = [np.concatenate([cfg0, cfg1]) for cfg0, cfg1 in zip(config_cands, configs)]

        pi_config = self.tpe_samplers[self.metric_name].compute_probability_improvement(config_cands=config_cands)
        for obj_name in self.constraints.keys():
            if self.constraints[obj_name] == np.inf:
                continue

            pi_config += self.tpe_samplers[obj_name].compute_probability_improvement(config_cands=config_cands)

        best_idx = int(np.argmax(pi_config))
        eval_config = {hp_name: config_cands[dim][best_idx]
                       for dim, hp_name in enumerate(self.hp_names)}

        return self._revert_eval_config(eval_config=eval_config)
