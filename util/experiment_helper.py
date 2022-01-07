from argparse import Namespace
from logging import Logger
from typing import Callable, Dict, List

import ConfigSpace as CS

from optimizer import opts
from optimizer import TPEOptimizer

import ujson as json


def exist_file(filename: str, max_evals: int) -> bool:
    try:
        data = json.load(open(f'results/{filename}.json'))
        return len(data['loss']) >= max_evals
    except FileNotFoundError:
        return False
    except KeyError:
        return False


def apply_knowledge_augmentation(
    args: Namespace,
    opt: TPEOptimizer,
    logger: Logger,
    cheap_obj: Callable,
    cheap_metrics: List[str],
    constraints: Dict[str, float],
    config_space: CS.ConfigurationSpace,
    file_name: str,
    seed: int
) -> None:
    if args.knowledge_augmentation:
        if args.opt_name != 'tpe':
            raise ValueError('Knowledge augmentation is available only for TPE')

        # Fetch the constraint keys for knowledge augmentation
        KA_keys = list(set(cheap_metrics) & set(constraints.keys()))
        print(f'Collect {KA_keys} for knowledge augmentation')
        _constraints = {
            key: constraints[key] for key in KA_keys
            if key != KA_keys[0]
        }

        rs = opts['random_search'](
            obj_func=cheap_obj,
            config_space=config_space,
            resultfile=file_name,
            max_evals=args.max_evals,
            constraints=_constraints,  # Ad-hoc solution
            metric_name=KA_keys[0],  # Ad-hoc solution
            seed=1000+seed,  # Avoid overlap in the initial configs
        )
        rs.optimize(logger)
        hp_values = {
            hp_name: rs._observations[hp_name]
            for hp_name in rs.hp_names
        }
        for metric_name in KA_keys:
            hp_values[metric_name] = rs._observations[metric_name]
            opt.tpe_samplers[metric_name].apply_knowledge_augmentation(
                hp_values
            )
            hp_values.pop(metric_name)
