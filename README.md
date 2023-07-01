# Constrained Tree-structured Parzen Estimator (c-TPE)
This package was used for the experiments of the paper `c-TPE: Tree-structured Parzen Estimator with Inequality Constraints for Expensive Hyperparameter Optimization`.
Note that the inference speed of c-TPE is not optimized to avoid bugs; however, when we optimize the speed, it will run as quick as NSGA-II.

## Usage
A simple example of c-TPE is available in [optimize_toy.py](optimize_toy.py).
After you run `pip install -r requirements.txt`, you can run the Python file with:

```shell
$ python optimize_toy.py
```

The example looks like this:

```python
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

```

## Setup to Reproduce Our Results
This package requires python 3.8 or later version.
You can install the dependency by:
```bash
$ conda create -n ctpe python==3.9
$ pip install -r requirements.txt

# NASBench101 setup
# Quicker version of NASBench101 (plus, tensorflow2+ works)
$ git clone https://github.com/nabenabe0928/nasbench
$ cd nasbench
$ pip install -e .
$ cd ..
$ pip install ./nasbench

# Create a directory for tabular datasets
$ mkdir ~/tabular_benchmarks
$ cd ~/tabular_benchmarks

# The download of HPOLib
$ cd ~/tabular_benchmarks
$ wget http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz
$ tar xf fcnet_tabular_benchmarks.tar.gz
$ mv fcnet_tabular_benchmarks hpolib

# The download of NASBench101
$ cd ~/tabular_benchmarks
$ mkdir nasbench101
$ cd nasbench101
# Table data (0.5GB) with only 1 type of budget
$ wget https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord

# The download of NASBench201
$ cd ~/tabular_benchmarks
$ wget https://drive.google.com/file/d/17_saCsj_krKjlCBLOJEpNtzPXArMCqxU/view
$ mv NATS-tss-v1_0-3ffb9-simple nasbench201
```

The constraint information used in the experiments is available in each `constraints.json` in the [targets](targets/) directory.

## Running Command to Reproduce Our Results
The data obtained in the experiments are reproduced by the following command:
```bash
# from seed 0 to 19
./submit.sh -s 0 -d 19
```
Note that `submit.sh` will perform 81,000 of optimizations each with 200 evaluations of configurations.

The test run for each optimization method or benchmark is performed by the following:
```bash
# Optimize the hyperparameters defined in `targets/hpolib/hyperparameters.py` and `targets/hpolib/params.json`
$ python optimize_hpolib.py

# Optimize the hyperparameters defined in `targets/nasbench101/cifar10B/hyperparameters.py` and `targets/nasbench101/cifar10B/params.json`
# The choices of search_space are {cifar10A, cifar10B, cifar10C}
$ python optimize_nasbench101.py

# Optimize the hyperparameters defined in `targets/nasbench201/hyperparameters.py` and `targets/nasbench201/params.json`
$ python optimize_nasbench201.py

# Test of cTPE
$ python optimize_hpolib.py --opt_name tpe

# Test of cBO
$ python optimize_hpolib.py --opt_name cbo

# Test of CNSGA-II
$ python optimize_hpolib.py --opt_name nsga2

# Test of Random search
$ python optimize_hpolib.py --opt_name random_search
```

## Citations

For the citation, use the following format:
```
@article{watanabe2023ctpe,
  title={{c-TPE}: Tree-structured {P}arzen Estimator with Inequality Constraints for Expensive Hyperparameter Optimization},
  author={S. Watanabe and F. Hutter},
  journal={International Joint Conference on Artificial Intelligence},
  year={2023}
}
```