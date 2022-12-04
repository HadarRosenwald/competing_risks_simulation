from typing import List
import pandas as pd
from consts import *
from distributions import *
from strata import Strata, get_strata


def generate_y(mu: List[float], strata: List[Strata], y_dist_param: Dict[str, float], t: int)-> List[Union[float, None]]:
    y = []
    relevant_strata = [Strata.AS.name, Strata.P.name] if t == 1 else [Strata.AS.name, Strata.H.name]
    for mu_i, s in zip(mu, strata):
        if s not in relevant_strata:
            y.append(None)
        else:
            sigma = y_dist_param[f'sigma_{t}1'] if s == Strata.AS else y_dist_param[f'sigma_{t}0']
            y.append(GaussianDist(n=1, param={'mu': mu_i, 'sigma': sigma}).sampled_vector)
    return y


def generate_mu0_mu1(y0_dist_param: Dict[str, float],
                     y1_dist_param: Dict[str, float],
                     x: np.ndarray, s0: np.ndarray, s1: np.ndarray):
    mu0 = [(y0_dist_param['a0'] + y0_dist_param['b0'] * x_i + y0_dist_param['c0'] * s0_i) if s0_i is not None else None
           for x_i, s0_i in zip(x, s0)]
    mu1 = [(y1_dist_param['a1'] + y1_dist_param['b1'] * x_i + y1_dist_param['c1'] * s1_i) if s1_i is not None else None
           for x_i, s1_i in zip(x, s1)]
    return mu0, mu1


def create_sample(x_dist: Union[GaussianDist, UniformDist, float] = x_dist_default,
                  y0_dist_param: Dict[str, float] = y0_dist_param_default,
                  y1_dist_param: Dict[str, float] = y1_dist_param_default,
                  treatment_prob: float = treatment_prob_default,
                  omega: float = omega_default,
                  beta_d: List[float] = beta_d_default,
                  population_size: int = population_size_default,
                  random_seed: int = default_random_seed) -> pd.DataFrame:
    random.seed(random_seed)
    variables_dict = dict()

    if x_dist == GaussianDist or x_dist == UniformDist:
        variables_dict['x'] = x_dist(n=population_size).sampled_vector
    else:
        variables_dict['x'] = np.full((population_size), x_dist)

    variables_dict['t'] = BernoulliDist(n=population_size, param={'p': treatment_prob}).sampled_vector

    D0_prob = [(1 / (1 + np.exp(-omega * (beta_d[0] + beta_d[1] * 0 + beta_d[2] * variables_dict['x'][i])))) for i in
               range(0, population_size)]
    D1_prob = [(1 / (1 + np.exp(-omega * (beta_d[0] + beta_d[1] * 1 + beta_d[2] * variables_dict['x'][i])))) for i in
               range(0, population_size)]
    variables_dict['D0'] = BernoulliDist(n=population_size, param={'p': D0_prob}).sampled_vector
    variables_dict['D1'] = BernoulliDist(n=population_size, param={'p': D1_prob}).sampled_vector

    variables_dict['stratum'] = [get_strata(d0, d1).name for d0, d1 in zip(variables_dict['D0'], variables_dict['D1'])]

    variables_dict['S1'] = [1 if s == Strata.AS.name else 0 if s == Strata.P.name else None for s in
                            variables_dict['stratum']]
    variables_dict['S0'] = [1 if s == Strata.AS.name else 0 if s == Strata.H.name else None for s in
                            variables_dict['stratum']]

    mu0, mu1 = generate_mu0_mu1(y0_dist_param, y1_dist_param, variables_dict['x'], variables_dict['S0'],
                                variables_dict['S1'])

    variables_dict['Y0'] = generate_y(mu0, variables_dict['stratum'], y0_dist_param, 0)
    variables_dict['Y1'] = generate_y(mu1, variables_dict['stratum'], y1_dist_param, 1)

    variables_dict['D_obs'] = [d1 if t else d0 for t, d0, d1 in
                               zip(variables_dict['t'], variables_dict['D0'], variables_dict['D1'])]
    variables_dict['Y_obs'] = [y1 if t else y0 for t, y0, y1 in
                               zip(variables_dict['t'], variables_dict['Y0'], variables_dict['Y1'])]

    variables_dict['beta_d'] = [beta_d for i in range(0, population_size)]
    variables_dict['mu0'] = mu0
    variables_dict['mu1'] = mu1
    variables_dict['sigma_0'] = y0_dist_param['sigma_01']
    variables_dict['sigma_1'] = y1_dist_param['sigma_11']

    variables_dict['a0'] = y0_dist_param['a0']
    variables_dict['b0'] = y0_dist_param['b0']
    variables_dict['c0'] = y0_dist_param['c0']
    variables_dict['a1'] = y1_dist_param['a1']
    variables_dict['b1'] = y1_dist_param['b1']
    variables_dict['c1'] = y1_dist_param['c1']

    return pd.DataFrame.from_dict(variables_dict)
