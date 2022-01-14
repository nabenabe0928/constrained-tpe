# Constraint Tree-structured Parzen Estimator (c-TPE)
This package was used for the experiments of the paper `c-TPE: Generalizing Tree-structured Parzen Estimator with Inequality Constraints for Continuous and Categorical Hyperparameter Optimization`.
Note that the inference speed of c-TPE is not optimized to avoid bugs; however, when we optimize the speed, it will run as quick as NSGA-II.

## Setup
This package requires python 3.8 or later version.
You can install the dependency by:
```bash
$ conda create -n ctpe python==3.8
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

## Running example
The data obtained in the experiments are reproduced by the following command:
```bash
./submit.sh
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
