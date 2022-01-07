import logging
import warnings

import optuna

from optimizer.cbo import ConstraintBayesianOptimization
from optimizer.hm_opt import HyperMapper
from optimizer.nsga2 import NSGA2
from optimizer.random_search import RandomSearch
from optimizer.tpe import TPEOptimizer


logging.getLogger("ax.core.parameter").setLevel(logging.CRITICAL)
logging.getLogger("ax.core.experiment").setLevel(logging.CRITICAL)
logging.getLogger("ax.service.managed_loop").setLevel(logging.CRITICAL)
logging.getLogger("ax.modelbridge.dispatch_utils").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.CRITICAL)


opts = {
    'tpe': TPEOptimizer,
    'nsga2': NSGA2,
    'cbo': ConstraintBayesianOptimization,
    'random_search': RandomSearch,
    'hm': HyperMapper
}
