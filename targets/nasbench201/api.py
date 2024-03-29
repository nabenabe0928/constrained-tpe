import os
from enum import Enum
import pickle
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np

from nats_bench.api_topology import NATStopology
import nats_bench

from targets.base_tabularbench_api import BaseTabularBenchAPI
from targets.nasbench201.hyperparameters import BudgetConfig, Hyperparameters


DATA_DIR = f"{os.environ['HOME']}/tabular_benchmarks/nasbench201"
PKL_FILE = "targets/nasbench201/metric_vals.pkl"
N_ARCH = 15625
AVAIL_SEEDS = [777, 888, 999]


"""
The information obtained via

Attributes:
    xxx-loss (float): cross-entropy value
    xxx-accuracy (float): accuracy
    xxx-per-time: Runtime for this epoch
    xxx-all-time: Runtime up to this epoch
"""
DataRowType = TypedDict(
    'DataRowType', {
        'train-loss': float,
        'train-accuracy': float,
        'train-per-time': float,
        'train-all-time': float,
        # For `cifar10-valid`
        'valid-loss': float,
        'valid-accuracy': float,
        'valid-per-time': float,
        'valid-all-time': float,
        # For `cifar100` and `ImageNet16-120`
        'valtest-loss': float,
        'valtest-accuracy': float,
        'valtest-per-time': float,
        'valtest-all-time': float,
        # Static information
        'size_in_mb': float
    }
)


class DatasetChoices(Enum):
    imagenet = 'ImageNet16-120'
    cifar10 = 'cifar10-valid'
    cifar100 = 'cifar100'


class ConstraintChoices(Enum):
    runtime = 'runtime'
    size_in_mb = 'size_in_mb'
    none = None


class NASBench201(BaseTabularBenchAPI):
    """
    Download the dataset via:
        https://drive.google.com/file/d/17_saCsj_krKjlCBLOJEpNtzPXArMCqxU/view?usp=sharing
    """
    def __init__(
        self,
        path: str = DATA_DIR,
        dataset: DatasetChoices = DatasetChoices.cifar100,
        seed: Optional[int] = None,
        constraints: List[ConstraintChoices] = [ConstraintChoices.none],
        feasible_domain_ratio: Optional[int] = None,
        cheap_metrics: List[str] = [ConstraintChoices.size_in_mb.name]
    ):
        super().__init__(
            hp_module_path='targets/nasbench201',
            dataset_name=dataset.name,
            constraints=constraints,  # type: ignore
            seed=seed,
            feasible_domain_ratio=feasible_domain_ratio,
            cheap_metrics=cheap_metrics
        )

        self._data = nats_bench.create(path, 'tss', fast_mode=True, verbose=False)
        self.dataset = dataset

    @staticmethod
    def config2archstr(config: Hyperparameters) -> str:
        # i<-j := the j-th node to the i-th node
        # |1<-0|+|2<-0|2<-1|+|3<-0|3<-1|3<-2|
        arch = '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(
            config.edge_0_1,
            config.edge_0_2,
            config.edge_1_2,
            config.edge_0_3,
            config.edge_1_3,
            config.edge_2_3
        )
        return arch

    def _collect_dataset_info(self, dataset_name: str) -> Dict[str, np.ndarray]:
        prefix = 'valid' if dataset_name == 'cifar10-valid' else 'valtest'
        loss_key = 'loss'
        runtime_key, network_key = ConstraintChoices.runtime.name, ConstraintChoices.size_in_mb.name
        results = {
            loss_key: np.empty(N_ARCH * len(AVAIL_SEEDS)),
            runtime_key: np.empty(N_ARCH * len(AVAIL_SEEDS)),
            network_key: np.empty(N_ARCH * len(AVAIL_SEEDS))
        }
        epochs, cnt = 199, 0
        for idx in range(N_ARCH):
            run_info = self.data.query_by_index(idx, dataname=dataset_name, hp='200')
            seeds = AVAIL_SEEDS[:len(run_info)]

            for i, seed in enumerate(seeds):
                info = self.data.get_more_info(
                    index=idx,
                    dataset=dataset_name,
                    hp='200',
                    iepoch=epochs,
                    is_random=seed
                )
                results[loss_key][cnt] = 100 - info[f'{prefix}-accuracy']
                results[runtime_key][cnt] = info['train-all-time']
                results[network_key][cnt] = run_info[seeds[0]].params
                cnt += 1

            # Delete the cache so that we can exaustively search all the architecture
            self.data.arch2infos_dict = {}

        return {k: v[:cnt] for k, v in results.items()}

    def _create_pickle_and_return_results(self) -> Dict[str, np.ndarray]:
        data = {}
        for dataset in DatasetChoices:
            data[dataset.name] = self._collect_dataset_info(dataset_name=dataset.value)

        with open(PKL_FILE, 'wb') as f:
            pickle.dump(data, f)

        return data

    def find_satisfactory_losses(self) -> np.ndarray:
        """ the oracle results are available in targets/nasbench201/constraints.json """

        if os.path.exists(PKL_FILE):
            with open(PKL_FILE, 'rb') as f:
                data = pickle.load(f)[self.dataset.name]
        else:
            data = self._create_pickle_and_return_results()[self.dataset.name]

        loss_vals, cnt = np.empty(N_ARCH * len(AVAIL_SEEDS)), 0
        runtime_key, network_key = ConstraintChoices.runtime.name, ConstraintChoices.size_in_mb.name
        for loss, runtime, network_size in zip(data['loss'], data[runtime_key], data[network_key]):
            results = {
                network_key: network_size,
                runtime_key: runtime
            }
            if self.is_satisfied_constraints(results):
                loss_vals[cnt] = loss
                cnt += 1

        return loss_vals[:cnt]

    def query_data(self, index: int, epochs: int) -> DataRowType:
        run_info = self.data.query_by_index(index, dataname=self.dataset.value, hp='200')

        seed = self.rng.randint(len(run_info))
        random_seed = AVAIL_SEEDS[seed]

        results = {ConstraintChoices.size_in_mb.name: run_info[random_seed].params}
        results.update(
            self.data.get_more_info(
                index=index,
                dataset=self.dataset.value,
                hp='200',
                iepoch=epochs,
                is_random=random_seed
            )
        )

        return results  # type: ignore

    def objective_func(self, config: Dict[str, Any], budget: Dict[str, Any] = {}) -> Dict[str, float]:
        _config = Hyperparameters(**config)
        _budget = BudgetConfig(**budget)

        arch = self.config2archstr(_config)
        index = self.data.archstr2index[arch]

        info = self.query_data(
            index=index,
            epochs=_budget.epochs - 1
        )

        loss = 100.0
        prefix = 'valid' if self.dataset.value == 'cifar10-valid' else 'valtest'

        loss -= info[f'{prefix}-accuracy']  # type: ignore
        runtime = info['train-all-time']

        return {
            'loss': loss,
            ConstraintChoices.runtime.name: runtime,
            ConstraintChoices.size_in_mb.name: info[ConstraintChoices.size_in_mb.name]
        }

    @property
    def data(self) -> NATStopology:
        return self._data
