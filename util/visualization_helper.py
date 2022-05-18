from typing import Dict, List, Optional, Tuple

import numpy as np

import ujson as json


OPT_NAMES = ['KA_tpe', 'tpe', 'vanilla_tpe', 'random_search', 'nsga2', 'cbo', 'hm']


class DataCollector:
    def __init__(
        self,
        file_path: str,
        bench_name: str,
        dataset: str,
        feasible_domain: int,
        constraint_keys: List[str],
        n_seeds: int = 50,
        max_evals: int = 200
    ):
        self.loss = np.array([])
        self.cum_loss = np.array([])
        self.cstr_vals: Dict[str, np.ndarray] = {}
        self.oracle = np.inf
        self.worst_oracle = -np.inf

        self.max_evals, self.n_seeds, self.file_path = max_evals, n_seeds, file_path
        self.constraint_keys = constraint_keys
        self.dataset, self.bench_name, self.feasible_domain = dataset, bench_name, feasible_domain
        self.constraints = self.get_constraints()

        self._build_data()

    def _build_data(self) -> None:
        self.loss = np.empty((self.n_seeds, self.max_evals))
        self.cstr_vals = {
            key: np.empty((self.n_seeds, self.max_evals))
            for key in self.constraint_keys
        }

        for seed in range(self.n_seeds):
            data = json.load(open(f'{self.file_path}/{seed:0>3}.json'))
            n_evals = len(data['loss'])
            n_KA = max(len(data[obj_name]) for obj_name in self.constraints) - n_evals
            self.loss[seed] = data['loss'][:self.max_evals]

            for obj_name in self.constraints:
                start = n_KA if len(data[obj_name]) > n_evals else 0
                self.cstr_vals[obj_name][seed] = data[obj_name][start:start+self.max_evals]

    def compute_cum_loss(self, inf_pad: Optional[float] = None) -> None:
        is_satisfied = self.is_satisfied_constraints()
        self.cum_loss = self.loss.copy()
        self.cum_loss[~is_satisfied] = self.worst_oracle if inf_pad is None else inf_pad
        self.cum_loss = np.minimum.accumulate(self.cum_loss, axis=-1)

    def get_cum_absolute_percentage_loss(self, inf_pad: Optional[float] = None) -> np.ndarray:
        self.compute_cum_loss()
        return (self.cum_loss - self.oracle) / self.oracle

    def get_cum_regret(self, inf_pad: Optional[float] = None) -> np.ndarray:
        self.compute_cum_loss()
        return self.cum_loss - self.oracle

    def get_best_loss(
        self,
        inf_pad: Optional[float] = None,
        end: int = 200
    ) -> np.ndarray:
        assert end <= self.max_evals

        inf_pad = self.worst_oracle if inf_pad is None else inf_pad
        is_satisfied = self.is_satisfied_constraints()
        return np.array([
            loss_array[s].min() if np.any(s)
            else inf_pad
            for loss_array, s in zip(self.loss[:, :end], is_satisfied[:, :end])
        ])

    def get_best_absolute_percentage_loss(
        self,
        inf_pad: Optional[float] = None,
        end: int = 200
    ) -> np.ndarray:

        inf_pad = (self.worst_oracle - self.oracle) / self.oracle if inf_pad is None else inf_pad
        best_loss = self.get_best_loss(inf_pad=inf_pad, end=end)
        return np.minimum((best_loss - self.oracle) / self.oracle, inf_pad)

    def get_best_regret(
        self,
        inf_pad: Optional[float] = None,
        end: int = 200
    ) -> np.ndarray:

        inf_pad = self.worst_oracle - self.oracle if inf_pad is None else inf_pad
        best_loss = self.get_best_loss(inf_pad=inf_pad, end=end)
        return np.minimum(best_loss - self.oracle, inf_pad)

    def get_constraints(self) -> Dict[str, float]:
        js = open(
            f'targets/{self.bench_name}/constraints.json' if self.bench_name != 'nasbench101'
            else f'targets/{self.bench_name}/{self.dataset}/constraints.json'
        )
        constraints_json = json.load(js)[self.dataset]
        domain_size_key = str(self.feasible_domain)

        oracle_key = 'oracle::' + ','.join(self.constraint_keys)
        self.oracle = constraints_json[oracle_key][domain_size_key]
        self.worst_oracle = constraints_json[f'worst_{oracle_key}'][domain_size_key]

        return {key: constraints_json[key][domain_size_key] for key in self.constraint_keys}

    def is_satisfied_constraints(self) -> np.ndarray:
        is_satisfied = np.ones((self.n_seeds, self.max_evals), dtype=np.bool8)
        for obj_name, lb in self.constraints.items():
            vals = self.cstr_vals[obj_name]
            is_satisfied &= (vals <= lb)

        return is_satisfied


class ExperimentCollector:
    def __init__(
        self,
        dir_name: str,
        bench_name: str,
        dataset: str,
        constraint_keys: List[str],
        opt_names: List[str] = OPT_NAMES,
        feasible_domains: List[int] = list(range(10, 100, 10)),
        n_seeds: int = 50
    ):

        self.data_collectors: Dict[int, Dict[str, DataCollector]] = {}
        self.n_seeds = n_seeds
        self.feasible_domains = feasible_domains

        opt_names_to_delete = set()
        for domain_size in feasible_domains:
            self.data_collectors[domain_size] = {}
            for opt_name in opt_names:
                file_path = f'{dir_name}/{bench_name}/{dataset}/feasible_{domain_size:0>3}per/{opt_name}'
                try:
                    self.data_collectors[domain_size][opt_name] = DataCollector(
                        file_path=file_path,
                        bench_name=bench_name,
                        dataset=dataset,
                        n_seeds=self.n_seeds,
                        feasible_domain=domain_size,
                        constraint_keys=constraint_keys
                    )
                except FileNotFoundError:
                    opt_names_to_delete.add(opt_name)
                    print(f"Skip {file_path}")

        self.opt_names = list(set(opt_names) - set(opt_names_to_delete))

    def _compute_cum_loss(self, inf_pad: Optional[float] = None) -> None:
        for domain_size in self.feasible_domains:
            dcs = self.data_collectors[domain_size]
            for opt_name in self.opt_names:
                dcs[opt_name].compute_cum_loss(inf_pad=inf_pad)

    def compute_rank_over_time(
        self,
        inf_pad: Optional[float] = None
    ) -> Dict[int, Dict[str, np.ndarray]]:

        self._compute_cum_loss(inf_pad=inf_pad)
        rank_dict: Dict[int, Dict[str, np.ndarray]] = {}

        for domain_size in self.feasible_domains:
            rank_dict[domain_size] = {}
            dcs = self.data_collectors[domain_size]
            # means: List[np.ndarray] = []
            medians: List[np.ndarray] = []

            for opt_name in self.opt_names:
                # average in the seed direction
                # mean_over_seed: np.ndarray = dcs[opt_name].cum_loss.mean(axis=0)
                median_over_seed: np.ndarray = np.median(dcs[opt_name].cum_loss, axis=0)
                medians.append(median_over_seed)

            rank = np.argsort(np.argsort(medians, axis=0), axis=0) + 1  # rank in the opt name direction
            rank_dict[domain_size] = {opt_name: r for opt_name, r in zip(self.opt_names, rank)}

        return rank_dict

    def compute_rank(
        self,
        inf_pad: Optional[float] = None,
        end: int = 200
    ) -> Dict[int, Dict[str, int]]:

        rank_dict: Dict[int, Dict[str, int]] = {}
        for domain_size in self.feasible_domains:
            rank_dict[domain_size] = {}
            means = []
            dcs = self.data_collectors[domain_size]
            for opt_name in self.opt_names:
                means.append(dcs[opt_name].get_best_loss(inf_pad=inf_pad, end=end).mean())

            rank = np.argsort(np.argsort(means)) + 1
            rank_dict[domain_size] = {opt_name: r for opt_name, r in zip(self.opt_names, rank)}

        return rank_dict

    def get_is_satisfied_moving_average(self, window_size: int) -> Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        is_satisfied_dict: Dict[int, Dict[str, np.ndarray]] = {}
        sqrtN = np.sqrt(self.n_seeds)
        for domain_size in self.feasible_domains:
            is_satisfied_dict[domain_size] = {}
            dcs = self.data_collectors[domain_size]
            for opt_name in self.opt_names:
                cum_is_satisfied = dcs[opt_name].is_satisfied_constraints().astype(np.int32).cumsum(axis=-1)
                mean_cum = cum_is_satisfied.mean(axis=0)
                ste_cum = cum_is_satisfied.std(axis=0) / sqrtN
                moving_avg_mean = np.convolve(mean_cum, np.ones(window_size), 'valid') / window_size
                moving_avg_ste = np.convolve(ste_cum, np.ones(window_size), 'valid') / window_size

                is_satisfied_dict[domain_size][opt_name] = (
                    moving_avg_mean,
                    moving_avg_ste,
                )

        return is_satisfied_dict
