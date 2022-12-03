from typing import Union, Dict

import numpy as np
from numpy import random


class GaussianDist:
    def __init__(self, n: int, param: Dict[str, float] = {'mu': 0.0, 'sigma': 1.0}):
        v = random.normal(loc=param['mu'], scale=param['sigma'], size=n)
        self.sampled_vector = v if n > 1 else v[0]


class UniformDist:
    def __init__(self, n: int, param: Dict[str, float] = {'a': -1.0, 'b': 1.0}):
        self.sampled_vector = random.uniform(low=param['a'], high=param['b'],
                                             size=n)


class BernoulliDist:
    def __init__(self, n: int, param: Dict[str, Union[float, np.array]] = {'p': 0.5}):
        self.sampled_vector = random.binomial(n=1, p=param['p'], size=n)
