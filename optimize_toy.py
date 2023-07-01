from typing import Dict, Any

import ConfigSpace as CS

from util.utils import get_logger

from optimizer import TPEOptimizer


def func(eval_config: Dict[str, Any]) -> Dict[str, float]:
    x, y = eval_config["x"], eval_config["y"]
    return dict(loss=x**2 + y**2, c1=x, c2=x)


if __name__ == '__main__':
    fn = "toy-example"
    logger = get_logger(file_name=fn, logger_name=fn)
    
    config_space = CS.ConfigurationSpace()
    config_space.add_hyperparameters([
        CS.UniformFloatHyperparameter(name="x", lower=-5.0, upper=5.0),
        CS.UniformFloatHyperparameter(name="y", lower=-5.0, upper=5.0),
    ])

    kwargs = dict(
        obj_func=func,
        config_space=config_space,
        resultfile=fn,
        max_evals=100,  # the number of configurations to evaluate
        constraints={"c1": 0.0, "c2": 0.0},  # c1 <= 0.0 and c2 <= 0.0 must hold
    )
    opt = TPEOptimizer(**kwargs)
    opt.optimize(logger)
