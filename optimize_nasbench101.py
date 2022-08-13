import sys

import numpy as np

from util.utils import get_logger

from optimizer import opts

from util.experiment_helper import apply_knowledge_augmentation, exist_file
from targets.nasbench101.api import (
    ConstraintChoices,
    DatasetChoices,
    NASBench101,
    SearchSpaceChoices
)
from util.utils import get_args_from_parser, get_filename_from_args


constraint_choices = [
    [ConstraintChoices.runtime, ConstraintChoices.n_params],
    [ConstraintChoices.n_params],
    [ConstraintChoices.runtime]
]


if __name__ == '__main__':
    args = get_args_from_parser(SearchSpaceChoices, opts=opts)
    constraints = constraint_choices[args.constraint_mode]

    file_name = get_filename_from_args('nasbench101', constraints, args)
    if exist_file(file_name, args.max_evals):
        print('Skip the optimization')
        sys.exit()

    logger = get_logger(file_name=file_name, logger_name=file_name)
    seed = args.exp_id

    bm = NASBench101(
        dataset=DatasetChoices.only108,
        search_space=getattr(SearchSpaceChoices, args.dataset),
        feasible_domain_ratio=args.feasible_domain,
        constraints=constraint_choices[args.constraint_mode],
        seed=seed
    )

    obj_func = bm.objective_func
    kwargs = dict(
        obj_func=obj_func,
        config_space=bm.config_space,
        resultfile=file_name,
        max_evals=args.max_evals,
        constraints={k: v if args.constraint else np.inf for k, v in bm.constraints.items()},
        seed=seed
    )
    if args.opt_name == 'hm':
        kwargs.update(hypermapper_json=f'targets/nasbench101/{args.dataset}/hypermapper.json')
    if args.naive:
        kwargs.update(naive_mode=True)

    opt = opts[args.opt_name](**kwargs)

    apply_knowledge_augmentation(
        args=args,
        opt=opt,
        logger=logger,
        cheap_obj=bm.cheap_objective_func,
        cheap_metrics=bm.cheap_metrics,
        constraints=bm.constraints,
        config_space=bm.config_space,
        file_name=file_name,
        seed=seed
    )

    opt.optimize(logger)
