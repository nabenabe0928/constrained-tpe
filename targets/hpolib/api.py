import json
import os
import pickle

import numpy as np

from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

import h5py

from targets.hpolib.hyperparameters import BudgetConfig, Hyperparameters
from targets.base_tabularbench_api import BaseTabularBenchAPI


DATA_DIR = f'{os.environ["HOME"]}/tabular_benchmarks/hpolib'
PKL_FILE = "targets/hpolib/metric_vals.pkl"
N_CONFIGS = 62208
AVAIL_SEEDS = [0, 1, 2, 3]


class TabularDataRowType(TypedDict):
    """
    The row data type of the tabular dataset.
    Each row is specified by a string that can be
    casted to dict and this dict is the hyperparameter
    configuration of this row data.

    Attributes:
        final_test_error (List[float]):
            The final test error over 4 seeds
        n_params (List[float]):
            The number of parameters of the model over 4 seeds
        runtime (List[float]):
            The runtime of the model over 4 seeds
        train_loss (List[List[float]]):
            The training loss of the model over 100 epochs
            with 4 different seeds
        train_mse (List[List[float]]):
            The training mse of the model over 100 epochs
            with 4 different seeds
        valid_loss (List[List[float]]):
            The validation loss of the model over 100 epochs
            with 4 different seeds
        valid_mse (List[List[float]]):
            The validation mse of the model over 100 epochs
            with 4 different seeds
    """
    final_test_error: List[float]
    n_params: List[float]
    runtime: List[float]
    train_loss: List[List[float]]
    train_mse: List[List[float]]
    valid_loss: List[List[float]]
    valid_mse: List[List[float]]


class DatasetChoices(Enum):
    slice_localization = "fcnet_slice_localization_data.hdf5"
    protein_structure = "fcnet_protein_structure_data.hdf5"
    naval_propulsion = "fcnet_naval_propulsion_data.hdf5"
    parkinsons_telemonitoring = "fcnet_parkinsons_telemonitoring_data.hdf5"


class ConstraintChoices(Enum):
    runtime = 'runtime'
    n_params = 'n_params'
    none = None


class HPOBench(BaseTabularBenchAPI):
    """
    Download the datasets via:
        $ wget http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz
        $ tar xf fcnet_tabular_benchmarks.tar.gz
    """
    def __init__(
        self,
        path: str = DATA_DIR,
        dataset: DatasetChoices = DatasetChoices.protein_structure,
        seed: Optional[int] = None,
        constraints: List[ConstraintChoices] = [ConstraintChoices.none],
        feasible_domain_ratio: Optional[int] = None,
        cheap_metrics: List[str] = [ConstraintChoices.n_params.name]
    ):
        super().__init__(
            hp_module_path='targets/hpolib',
            dataset_name=dataset.name,
            constraints=constraints,  # type: ignore
            seed=seed,
            feasible_domain_ratio=feasible_domain_ratio,
            cheap_metrics=cheap_metrics
        )
        self._path = path
        self._data = h5py.File(os.path.join(path, dataset.value), "r")
        self._dataset = dataset
        self._metric_name = 'valid_mse'

    def _collect_dataset_info(self, dataset_name: str) -> Dict[str, np.ndarray]:
        data = h5py.File(os.path.join(self._path, dataset_name), "r")
        loss_key = 'loss'
        runtime_key, network_key = ConstraintChoices.runtime.name, ConstraintChoices.n_params.name
        results = {
            loss_key: np.empty(N_CONFIGS * len(AVAIL_SEEDS)),
            runtime_key: np.empty(N_CONFIGS * len(AVAIL_SEEDS)),
            network_key: np.empty(N_CONFIGS * len(AVAIL_SEEDS))
        }
        epochs, cnt = 99, 0
        for key in data.keys():
            info = data[key]
            runtime_vals = info[runtime_key]
            n_params_vals = info[network_key]
            loss_vals = info[self._metric_name][:, epochs]

            for loss, runtime, n_params in zip(loss_vals, runtime_vals, n_params_vals):
                results[loss_key][cnt] = loss
                results[runtime_key][cnt] = runtime
                results[network_key][cnt] = n_params
                cnt += 1

        return {k: v[:cnt] for k, v in results.items()}

    def _create_pickle_and_return_results(self) -> Dict[str, np.ndarray]:
        data = {}
        for dataset in DatasetChoices:
            data[dataset.name] = self._collect_dataset_info(dataset_name=dataset.value)

        with open(PKL_FILE, 'wb') as f:
            pickle.dump(data, f)

        return data

    def find_satisfactory_losses(self) -> np.ndarray:
        """ the oracle results are available in targets/hpolib/constraints.json """
        if os.path.exists(PKL_FILE):
            with open(PKL_FILE, 'rb') as f:
                data = pickle.load(f)[self.dataset.name]
        else:
            data = self._create_pickle_and_return_results()[self.dataset.name]

        loss_vals, cnt = np.empty(N_CONFIGS * len(AVAIL_SEEDS)), 0
        runtime_key, network_key = ConstraintChoices.runtime.name, ConstraintChoices.n_params.name
        for loss, runtime, network_size in zip(data['loss'], data[runtime_key], data[network_key]):
            results = {
                network_key: network_size,
                runtime_key: runtime
            }
            if self.is_satisfied_constraints(results):
                loss_vals[cnt] = loss
                cnt += 1

        return loss_vals[:cnt]

    def objective_func(self, config: Dict[str, Any], budget: Dict[str, Any] = {}) -> Dict[str, float]:
        _budget = BudgetConfig(**budget)
        config = Hyperparameters(**config).__dict__

        idx = self.rng.randint(4)
        key = json.dumps(config, sort_keys=True)
        runtime_key, n_params_key = ConstraintChoices.runtime.name, ConstraintChoices.n_params.name
        loss_key = 'loss'
        results = {
            loss_key: float(self.data[key][self._metric_name][idx][_budget.epochs - 1]),
            runtime_key: float(self.data[key][runtime_key][idx]),
            n_params_key: float(self.data[key][n_params_key][idx])
        }
        return results

    @property
    def data(self) -> Any:
        return self._data

    @property
    def dataset(self) -> DatasetChoices:
        return self._dataset
