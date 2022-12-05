from typing import List, Tuple, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import random
from scipy import integrate

from consts import default_random_seed

random.seed(default_random_seed)


################# zhang and rubin entities ########################
def calc_non_parametric_p_t0d0_p_t1d0(df: pd.DataFrame) -> Tuple[float]:
    p_t0d0 = df.loc[(df.D_obs == 0) & (df.t == 0)].shape[0] / df.loc[df.t == 0].shape[0]
    p_t1d0 = df.loc[(df.D_obs == 0) & (df.t == 1)].shape[0] / df.loc[df.t == 1].shape[0]
    return p_t0d0, p_t1d0


def get_y_0_obs_y_1_obs(df: pd.DataFrame) -> Tuple[np.ndarray]:
    y_0_obs = df.loc[df.t == 0]['Y0'].dropna()
    y_1_obs = df.loc[df.t == 1]['Y1'].dropna()
    return y_0_obs, y_1_obs


def get_q1_q2(p_t0d0: float, p_t1d0: float, pi_h: float) -> Tuple[float]:
    q1 = p_t0d0 / p_t1d0 - pi_h / p_t1d0
    q2 = 1 - pi_h / p_t0d0
    return q1, q2


################# zhang and rubin non parametric bounds ########################
def calc_non_parametric_zhang_rubin_given_arg(y_0_obs: np.ndarray, y_1_obs: np.ndarray, q1: float, q2: float) -> Tuple[
    float]:
    y_0_obs_sorted = np.sort(y_0_obs)
    y_1_obs_sorted = np.sort(y_1_obs)
    zhang_rubin_lb = np.average(y_1_obs_sorted[0: round(len(y_1_obs_sorted) * q1)]) - np.average(
        y_0_obs_sorted[round(len(y_0_obs_sorted) * (1 - q2)): len(y_0_obs_sorted)])
    zhang_rubin_ub = np.average(
        y_1_obs_sorted[round(len(y_1_obs_sorted) * (1 - q1)): len(y_1_obs_sorted)]) - np.average(
        y_0_obs_sorted[0: round(len(y_0_obs_sorted) * q2)])
    return zhang_rubin_lb, zhang_rubin_ub


def calc_non_parametric_zhang_rubin(sample_df: pd.DataFrame,
                                    plot_and_print: bool = False,
                                    pi_h_step: float = 0.001):
    p_t0d0, p_t1d0 = calc_non_parametric_p_t0d0_p_t1d0(sample_df)
    lower_pi = max(0, p_t0d0 - p_t1d0)
    upper_pi = min(p_t0d0, 1 - p_t1d0)
    pi_h_list = np.arange(lower_pi, upper_pi + pi_h_step, pi_h_step)

    y_0_obs, y_1_obs = get_y_0_obs_y_1_obs(sample_df)

    # bounds
    zhang_rubin_lb_results = []
    zhang_rubin_ub_results = []
    for pi_h in pi_h_list:
        q1, q2 = get_q1_q2(p_t0d0, p_t1d0, pi_h)
        zhang_rubin_lb, zhang_rubin_ub = calc_non_parametric_zhang_rubin_given_arg(y_0_obs, y_1_obs, q1, q2)

        if plot_and_print:
            print(f"""pi_h {round(pi_h, 3)}:
            Zhang and Rubin bounds are [{zhang_rubin_lb}, {zhang_rubin_ub}]
            """)
        zhang_rubin_lb_results.append(zhang_rubin_lb)
        zhang_rubin_ub_results.append(zhang_rubin_ub)

    zhang_rubin_lb = np.nanmin(zhang_rubin_lb_results)
    zhang_rubin_ub = np.nanmax(zhang_rubin_ub_results)

    return (zhang_rubin_lb, zhang_rubin_ub)


################# plots ########################
def plot_integrand_function(y_phi: Callable[[float], float], pi_h_step: float):
    min_y_phi, max_y_phi = -7.5, 7.5
    y_values, y_phi_values = zip(*[(y, y_phi(y)) for y in np.arange(min_y_phi, max_y_phi, pi_h_step)])
    plt.figure(figsize=(15, 3))
    plt.title(r'Plot of the integrand $y\cdot\varphi(y)$')
    plt.xlabel(r'$y$')
    plt.ylabel(r'$y\cdot\varphi(y)$')
    plt.plot(y_values, y_phi_values)
    plt.show()


def plot_pi_h(lower_pi: float, upper_pi: float):
    plt.plot((lower_pi, upper_pi), (1, 1), 'ro-', color='orange')
    plt.annotate(r'$max(0,P_{T=0,D=0}-P_{T=1,D=0})$', (lower_pi, 1.005))
    plt.annotate(round(lower_pi, 3), (lower_pi, 1))
    plt.annotate(r'$min(P_{T=0,D=0},1-P_{T=1,D=0})$', (upper_pi, 1.005))
    plt.annotate(round(upper_pi, 3), (upper_pi, 1))
    plt.xlim(lower_pi - 0.005, upper_pi + 0.005)
    plt.yticks([])
    axes = plt.gca()
    plt.box(False)
    plt.xlabel(r'$\pi_h$')
    plt.show()


def plot_pi_h_and_bounds(pi_h: List[float], zhang_rubin_lb: List[float], zhang_rubin_ub: List[float]):
    plt.scatter(pi_h, zhang_rubin_lb, label="zhang_rubin_lb", s=2.5)
    plt.scatter(pi_h, zhang_rubin_ub, label="zhang_rubin_ub", s=2.5)
    # plt.legend(markerscale=10)
    plt.legend()
    plt.title(r"Grid search of $\pi_h$ and Zhang & Rubin's")
    plt.xlabel(r'$\pi_h$')
    plt.ylabel(r'Zhang & Rubin Bounds')
    plt.show()


################# ndtri of mixture of gaussian ########################
def continuous_bisect_fun_left(f, v, lo, hi):
    # Return the smallest value x between lo and hi such that f(x) >= v
    val_range = [lo, hi]
    k = 0.5 * sum(val_range)
    for i in range(32):  # TODO WHY 32
        val_range[int(f(k) > v)] = k
        next_k = 0.5 * sum(val_range)
        if next_k == k:
            break
        k = next_k
    return k


def get_mixture_cdf(component_distributions, ps):
    # Return the function that is the cdf of the mixture distribution
    return lambda x: sum(component_dist.cdf(x) * p for component_dist, p in zip(component_distributions, ps))


def mixture_quantile(p, component_distributions, ps):
    # Return the pth quantile of the mixture distribution given by the component distributions and their probabilities
    mixture_cdf = get_mixture_cdf(component_distributions, ps)

    lo = np.min([dist.ppf(p) for dist in component_distributions])
    hi = np.max([dist.ppf(p) for dist in component_distributions])

    return continuous_bisect_fun_left(mixture_cdf, p, lo, hi)


def cdf_of_mixture_of_quantile(f1, f2, quantile, weight):
    ps_component1 = f1.cdf(quantile)
    ps_component2 = f2.cdf(quantile)
    ps_mixture = weight * ps_component1 + (1 - weight) * ps_component2
    return ps_mixture


def get_ndtri_of_mix(alpha, component_distributions, ps, weight):
    quantile = mixture_quantile(alpha, component_distributions, ps)
    ndtri_mix = cdf_of_mixture_of_quantile(component_distributions[0], component_distributions[0], quantile, weight)
    return ndtri_mix


################# zhang and rubin parametric bounds ########################
def calculate_integral(func, lower_limit_integration, upper_limit_integration):
    integral_result, estimate_absolute_error = integrate.quad(func, lower_limit_integration, upper_limit_integration)
    return integral_result


def calculate_integral_bounds(component_distributions, weight, ps, alpha_list, mu, sigma):
    argmt_integral_list = []
    for alpha in alpha_list:
        if alpha == 0:
            ndtri_mix = 0.0  # alpha=0 -> quantile=-inf -> ndtri=0
        elif alpha == 1:
            ndtri_mix = 1.0  # alpha=1 -> quantile=inf -> ndtri=1
        else:
            ndtri_mix = get_ndtri_of_mix(alpha, component_distributions, ps, weight)
        argmt_integral_list.append((ndtri_mix - mu) / sigma)
    return argmt_integral_list