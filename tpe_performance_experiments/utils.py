import json
from typing import Dict, List, Tuple, Union

from targets.hpolib.api import DatasetChoices as hpolib_choices
from targets.hpolib.api import HPOBench
from targets.nasbench101.api import SearchSpaceChoices as nb101_choices
from targets.nasbench101.api import NASBench101
from targets.nasbench201.api import DatasetChoices as nb201_choices
from targets.nasbench201.api import NASBench201


TabularBenchmarkType = Union[HPOBench, NASBench101, NASBench201]


def get_search_space(task_name: str) -> Dict[str, List[Union[int, str, float]]]:
    return json.load(open("tpe_performance_experiments/search_spaces.json"))[task_name]


def convert_config(
    config: Dict[str, Union[int, str, float]],
    search_space: Dict[str, List[Union[int, str, float]]],
) -> Dict[str, Union[int, str, float]]:
    eval_config = {key: search_space[key][int(idx)] for key, idx in config.items()}
    return eval_config


def get_benchmark(
    task_name: str, seed: int
) -> Tuple[TabularBenchmarkType, Dict[str, List[Union[int, str, float]]]]:
    hpolib_tasks = [c.name for c in hpolib_choices]
    nb101_tasks = [c.name for c in nb101_choices]
    nb201_tasks = [c.name for c in nb201_choices]
    if task_name in hpolib_tasks:
        search_space = get_search_space(task_name="hpolib")
        return (
            HPOBench(dataset=getattr(hpolib_choices, task_name), seed=seed),
            search_space,
        )
    if task_name in nb201_tasks:
        search_space = get_search_space(task_name="nasbench201")
        return (
            NASBench201(dataset=getattr(nb201_choices, task_name), seed=seed),
            search_space,
        )
    if task_name in nb101_tasks:
        search_space = get_search_space(task_name=task_name)
        return (
            NASBench101(search_space=getattr(nb101_choices, task_name), seed=seed),
            search_space,
        )

    raise ValueError(
        f"task_name must be in {hpolib_tasks} or {nb101_tasks} or {nb201_tasks}, but got {task_name}"
    )


class ResultsManager:
    def __init__(self, task_name: str, seed: int):
        self.bm, self.search_space = get_benchmark(task_name, seed)
        self.loss: List[float] = []

    def obj_func(self, config: Dict[str, Union[int, str, float]]) -> float:
        if "dummy" in config:  # Ad-hoc solution for cocabo
            config.pop("dummy")

        eval_config = convert_config(config, self.search_space)
        loss_val = self.bm.objective_func(eval_config)["loss"]
        self.loss.append(loss_val)
        return loss_val
