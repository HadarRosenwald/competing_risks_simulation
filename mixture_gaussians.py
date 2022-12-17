import functools
import numpy as np
from scipy.optimize import root_scalar


def _vectorize_float(f):
    vectorized = np.vectorize(f, otypes=[float], signature="(),()->()")

    @functools.wraps(f)
    def wrapper(*args):
        return vectorized(*args)

    return wrapper


class MixtureDistribution:
    def __init__(self, distributions, weights):
        self._distributions = list(distributions)
        self._weights = list(weights)

        if not (all(w >= 0 for w in self._weights) and sum(self._weights) == 1):
            raise ValueError("Invalid weight vector.")

        if len(self._distributions) != len(self._weights):
            raise ValueError("Mixtures and weights must have the same length.")

        if len(self._distributions) < 2:
            raise ValueError("Must have at least two component distributions.")

    @_vectorize_float
    def pdf(self, x):
        return sum(w * d.pdf(x) for w, d in zip(self._weights, self._distributions))

    @_vectorize_float
    def cdf(self, x):
        return sum(w * d.cdf(x) for w, d in zip(self._weights, self._distributions))

    @_vectorize_float
    def ppf(self, p):
        # The percentage-point function inverts the cdf using the standard root_scalar function. As a cdf is monotonic, the bracketing interval must be bounded by the minimum and maximum of the quantile function across all the component distributions.

        if p == 0:
            return -float('Inf')  # alpha=0 -> quantile=-inf
        elif p == 1:
            return float('Inf')  # alpha=1 -> quantile=inf
        else:
            bracket = [min(dist.ppf(p) for dist in self._distributions),
                       max(dist.ppf(p) for dist in self._distributions)]

            try:
                r = root_scalar(
                    f=lambda x: self.cdf(x) - p,
                    fprime=self.pdf,
                    bracket=bracket,
                    x0=0
                )
                if not r.converged:
                    raise ValueError("Mixture of gaussian quantiles failed to converge.")
                return r.root
            except ValueError as e:
                if bracket[0] == bracket[1]:
                    return bracket[0] # TODO THINK IF THIS IS CORRECT
                else:
                    raise ValueError("Mixture of gaussian quantiles failed to find a root.")


