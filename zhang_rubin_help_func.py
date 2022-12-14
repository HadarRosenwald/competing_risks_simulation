from math import exp, sqrt, pi, pow
from typing import Callable
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import random
from scipy import integrate
from scipy import stats

from consts import default_random_seed
from consts import y1_dist_param_default, y0_dist_param_default
from sample_generation import create_sample
from strata import Strata

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


def pi_h_and_bounds_plots_controller(df: pd.DataFrame) -> None:
    df_for_analysis = df.sample(n=40, random_state=1).sort_values(by="x")
    for i, row in df_for_analysis.iterrows():
        print(f"x is: {round(row.x, 2)}")
        calc_non_parametric = not (i % 10)
        if calc_non_parametric:
            print("** including the non parametric bounds **")
        calc_zhang_rubin_bounds_per_x(x=row.x, plot_and_print=True, mu_y_0_x=row.mu0, mu_y_1_x=row.mu1,
                                      sigma_0=row.sigma_0, sigma_1=row.sigma_1, a0=row.a0, b0=row.b0, c0=row.c0,
                                      a1=row.a1, b1=row.b1, c1=row.c1, beta_d=row.beta_d,
                                      calc_non_parametric=calc_non_parametric)
    # zr_as_cate_bound_for_given_x = calc_zhang_rubin_bounds_per_x(x=0.0065, plot_and_print=True, beta_d = [5.0, -10.0, 6.0])


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


def plot_zhang_rubin_bounds(df: pd.DataFrame, zhang_rubin_bounds: List[Tuple[float, float]],
                            y0_dist_param: Dict[str, float] = y0_dist_param_default,
                            y1_dist_param: Dict[str, float] = y1_dist_param_default):
    df_plot_as = df.loc[df.stratum == Strata.AS.name]
    plt.scatter(df_plot_as.x, (df_plot_as.Y1 - df_plot_as.Y0), label="True Y1 - Y0|AS", s=0.1)

    mu_y_0_x = df.mu0
    mu_y_1_x = df.mu1

    lb, up = zip(*zhang_rubin_bounds)
    print(f"lower bound: {lb}, upper bound: {up}, true value: {mu_y_1_x - mu_y_0_x}")
    plt.scatter(list(df.x), lb, label="Lower bound", s=0.1)
    plt.scatter(list(df.x), up, label="Upper bound", s=0.1)
    plt.scatter(list(df.x), mu_y_1_x - mu_y_0_x, label=r'$\mu_{y(1)|x}-\mu_{y(0)|x}$', s=0.1)

    plt.legend(markerscale=12)
    # plt.legend()
    plt.title("Bounding Y1-Y0|AS by Zhang and Rubin")
    plt.xlabel('X')
    plt.ylabel('Y1-Y0|AS bounds')
    plt.ylim((min(-3, min(lb)), max(3, max(up))))
    plt.show()
    return {'lb': lb, 'up': up, 'true value': mu_y_1_x - mu_y_0_x}


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
    if alpha == 0:
        ndtri_mix = 0.0  # alpha=0 -> quantile=-inf -> ndtri=0
    elif alpha == 1:
        ndtri_mix = 1.0  # alpha=1 -> quantile=inf -> ndtri=1
    else:
        quantile = mixture_quantile(alpha, component_distributions, ps)
        ndtri_mix = cdf_of_mixture_of_quantile(component_distributions[0], component_distributions[0], quantile, weight)
    return ndtri_mix


################# zhang and rubin parametric bounds ########################
def calculate_integral(func, lb_integration, ub_integration):
    integral_result, estimate_absolute_error = integrate.quad(func, lb_integration, ub_integration)
    return integral_result


def cdf_s_normal(y: float) -> float:
    return 1 / (sqrt(2 * pi)) * exp(-1 / 2 * pow(y, 2))


def integrand_func(sigma: float, mu: float) -> Callable[[float, float], float]:
    return lambda y: (y * sigma + mu) * cdf_s_normal(y)


def calc_zhang_rubin_bounds_per_x(
        x: float, mu_y_0_x: float, mu_y_1_x: float, sigma_0: float, sigma_1: float,
        a0: float, b0: float, c0: float, a1: float, b1: float, c1: float,
        beta_d: List[float], plot_and_print: bool = False, pi_h_step: float = 0.001,
        calc_non_parametric=False) -> Tuple[float, float]:
    beta_d0, beta_d1, beta_d2 = beta_d
    # print(f"mu0: {mu_y_0_x}, mu1: {mu_y_1_x}, sigma0: {sigma_0}, sigma1: {sigma_1}, beta: {beta_d}")

    # pi
    p_t0d0 = 1 - 1 / (1 + exp(-beta_d0 - beta_d2 * x))
    p_t1d0 = 1 - 1 / (1 + exp(-beta_d0 - beta_d1 - beta_d2 * x))
    lower_pi = max(0, p_t0d0 - p_t1d0)
    upper_pi = min(p_t0d0, 1 - p_t1d0)
    pi_h_list = np.arange(lower_pi, upper_pi + pi_h_step, pi_h_step)
    if plot_and_print:
        plot_pi_h(lower_pi, upper_pi)

    # bounds
    zhang_rubin_lb_results = []
    zhang_rubin_ub_results = []
    for pi_h in pi_h_list:
        # Y1
        mu_1_as = a1 + b1 * x + c1
        mu_1_p = a1 + b1 * x
        # QUESTION: Where do we account for the fact that this x resulted in EITHER AS or P? we don't have another x like this with the opposite strata. We are taking weighted average but where does the probability of being AS/P - in terms of Beta - is addressed?

        weight = 1 - p_t0d0 / p_t1d0 + pi_h / p_t1d0
        f1 = stats.norm(loc=mu_1_p, scale=sigma_1)
        f2 = stats.norm(loc=mu_1_as, scale=sigma_1)
        component_distributions = [f1, f2]
        ps = [weight, 1 - weight]

        ndtri_0 = get_ndtri_of_mix(0, component_distributions, ps, weight)
        ndtri_1_minus_weight = get_ndtri_of_mix(1 - weight, component_distributions, ps, weight)
        ndtri_weight = get_ndtri_of_mix(weight, component_distributions, ps, weight)
        ndtri_1 = get_ndtri_of_mix(1, component_distributions, ps, weight)

        lb_frst_argmt_integral_p = calculate_integral(
            func=integrand_func(sigma_1, mu_1_p),
            lb_integration=(ndtri_0 - mu_1_p) / sigma_1,
            ub_integration=(ndtri_1_minus_weight - mu_1_p) / sigma_1)
        lb_frst_argmt_integral_as = calculate_integral(
            func=integrand_func(sigma_1, mu_1_as),
            lb_integration=(ndtri_0 - mu_1_as) / sigma_1,
            ub_integration=(ndtri_1_minus_weight - mu_1_as) / sigma_1)

        lb_frst_argmt_integral = weight * lb_frst_argmt_integral_p + (1 - weight) * lb_frst_argmt_integral_as

        ub_frst_argmt_integral_p = calculate_integral(
            func=integrand_func(sigma_1, mu_1_p),
            lb_integration=(ndtri_weight - mu_1_p) / sigma_1,
            ub_integration=(ndtri_1 - mu_1_p) / sigma_1)
        ub_frst_argmt_integral_as = calculate_integral(
            func=integrand_func(sigma_1, mu_1_as),
            lb_integration=(ndtri_weight - mu_1_as) / sigma_1,
            ub_integration=(ndtri_1 - mu_1_as) / sigma_1)

        ub_frst_argmt_integral = weight * ub_frst_argmt_integral_p + (1 - weight) * ub_frst_argmt_integral_as

        # Y0
        mu_0_as = a0 + b0 * x + c0
        mu_0_h = a0 + b0 * x
        # QUESTION: Where do we account for the fact that this x resulted in EITHER AS or P? we don't have another x like this with the opposite strata. We are taking weighted average but where does the probability of being AS/P - in terms of Beta - is addressed?

        weight = pi_h / p_t0d0
        f1 = stats.norm(loc=mu_0_h, scale=sigma_0)
        f2 = stats.norm(loc=mu_0_as, scale=sigma_0)
        component_distributions = [f1, f2]
        ps = [weight, 1 - weight]

        ndtri_h_t1d0 = get_ndtri_of_mix(pi_h / p_t1d0, component_distributions, ps, weight)
        ndtri_1 = get_ndtri_of_mix(1, component_distributions, ps, weight)
        ndtri_0 = get_ndtri_of_mix(0, component_distributions, ps, weight)
        ndtri_1_minus_h_t1d0 = get_ndtri_of_mix(1 - pi_h / p_t1d0, component_distributions, ps, weight)

        lb_scnd_argmt_integral_h = calculate_integral(
            func=integrand_func(sigma_0, mu_0_h),
            lb_integration=(ndtri_h_t1d0 - mu_0_h) / sigma_0,
            ub_integration=(ndtri_1 - mu_0_h) / sigma_0)
        lb_scnd_argmt_integral_as = calculate_integral(
            func=integrand_func(sigma_0, mu_0_as),
            lb_integration=(ndtri_h_t1d0 - mu_0_as) / sigma_0,
            ub_integration=(ndtri_1 - mu_0_as) / sigma_0)

        lb_scnd_argmt_integral = weight * lb_scnd_argmt_integral_h + (1 - weight) * lb_scnd_argmt_integral_as

        ub_scnd_argmt_integral_h = calculate_integral(
            func=integrand_func(sigma_0, mu_0_h),
            lb_integration=(ndtri_0 - mu_0_h) / sigma_0,
            ub_integration=(ndtri_1_minus_h_t1d0 - mu_0_h) / sigma_0)
        ub_scnd_argmt_integral_as = calculate_integral(
            func=integrand_func(sigma_0, mu_0_as),
            lb_integration=(ndtri_0 - mu_0_as) / sigma_0,
            ub_integration=(ndtri_1_minus_h_t1d0 - mu_0_as) / sigma_0)

        ub_scnd_argmt_integral = weight * ub_scnd_argmt_integral_h + (1 - weight) * ub_scnd_argmt_integral_as

        zhang_rubin_lb = lb_frst_argmt_integral - lb_scnd_argmt_integral
        zhang_rubin_ub = ub_frst_argmt_integral - ub_scnd_argmt_integral
        if plot_and_print:
            print(f"""pi_h {round(pi_h, 3)}:
            Zhang and Rubin bounds are [{zhang_rubin_lb}, {zhang_rubin_ub}]
            """)
        zhang_rubin_lb_results.append(zhang_rubin_lb)
        zhang_rubin_ub_results.append(zhang_rubin_ub)

    zhang_rubin_lb = np.nanmin(zhang_rubin_lb_results)
    zhang_rubin_ub = np.nanmax(zhang_rubin_ub_results)

    if plot_and_print:
        print(f"mu_y_1_x-mu_y_0_x: {mu_y_1_x - mu_y_0_x}")
        print(f"Zhang and Rubins bounds: [{zhang_rubin_lb}, {zhang_rubin_ub}]")
        if calc_non_parametric:
            zhang_rubin_lb_non_parametric, zhang_rubin_ub_non_parametric = calc_non_parametric_zhang_rubin(
                create_sample(x_dist=x))
            print(
                f"Zhang and Rubins non parametric bounds: [{zhang_rubin_lb_non_parametric}, {zhang_rubin_ub_non_parametric}]")
        plot_pi_h_and_bounds(pi_h_list, zhang_rubin_lb_results, zhang_rubin_ub_results)

    return (zhang_rubin_lb, zhang_rubin_ub)


def calc_zhang_rubin_bounds(df: pd.DataFrame) -> List[Tuple[float, float]]:
    list_of_bounds = list()
    for index, row in df.iterrows():
        if index % 1000 == 0:
            print(f"row {index} out of {df.shape[0]}")
        bound = calc_zhang_rubin_bounds_per_x(x=row.x, mu_y_0_x=row.mu0, mu_y_1_x=row.mu1, sigma_0=row.sigma_0,
                                              sigma_1=row.sigma_1, a0=row.a0, b0=row.b0, c0=row.c0, a1=row.a1,
                                              b1=row.b1, c1=row.c1, beta_d=row.beta_d, pi_h_step=0.01)
        list_of_bounds.append(bound)
    return list_of_bounds
