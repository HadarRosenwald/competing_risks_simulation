from math import exp, sqrt, pi, pow
from typing import Callable
from typing import Dict, List, Tuple

from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import random
from scipy import integrate
from scipy import stats

from consts import default_random_seed
from consts import y1_dist_param_default, y0_dist_param_default
from mixture_gaussians import GaussianMixtureDistribution
from sample_generation import create_sample
from strata import Strata
from superquantile_model import *

random.seed(default_random_seed)


################# zhang and rubin entities ########################
def calc_non_parametric_p_t0d0_p_t1d0(df: pd.DataFrame) -> Tuple[float, float]:
    p_t0d0 = df.loc[(df.D_obs == 0) & (df.t == 0)].shape[0] / df.loc[df.t == 0].shape[0]
    p_t1d0 = df.loc[(df.D_obs == 0) & (df.t == 1)].shape[0] / df.loc[df.t == 1].shape[0]
    return p_t0d0, p_t1d0


def get_y_0_obs_y_1_obs(df: pd.DataFrame) -> Tuple[float, float]:
    y_0_obs = df.loc[df.t == 0]['Y_obs'].dropna()
    y_1_obs = df.loc[df.t == 1]['Y_obs'].dropna()
    return y_0_obs, y_1_obs


def get_q1_q2(p_t0d0: float, p_t1d0: float, pi_h: float) -> Tuple[float, float]:
    q1 = p_t0d0 / p_t1d0 - pi_h / p_t1d0
    q2 = 1 - pi_h / p_t0d0
    return q1, q2


################# zhang and rubin non parametric bounds ########################
def calc_non_parametric_zhang_rubin_given_arg(y_0_obs: np.ndarray, y_1_obs: np.ndarray, q1: float, q2: float) -> Tuple[
    float, float]:
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
    lower_pi = max(0.0, p_t0d0 - p_t1d0)
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

    return zhang_rubin_lb, zhang_rubin_ub


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


def plot_zhang_rubin_bounds_on_survivors(df: pd.DataFrame, zhang_rubin_bounds: List[Tuple[float, float]],
                                         title: str, y_title: str, plot_yi1_yi0_diff: bool = False,
                                         plot_graph_margin: bool = False, ylim_margin: int = 3,
                                         y0_dist_param: Dict[str, float] = y0_dist_param_default,
                                         y1_dist_param: Dict[str, float] = y1_dist_param_default):
    if 'stratum' in df.columns:
        df_as = df.loc[df.stratum == (Strata.AS.name if Strata.AS.name in df.stratum.values else Strata.AS)]
        if plot_yi1_yi0_diff:
            plt.scatter(df_as.x, (df_as.Y1 - df_as.Y0), label="True Y1 - Y0|AS", s=0.1)

        mu_y_0_x_as = df_as.mu0
        mu_y_1_x_as = df_as.mu1

    lb, up = zhang_rubin_bounds
    # print(f"lower bound: {lb}, upper bound: {up}, true value: {mu_y_1_x_as - mu_y_0_x_as}")
    plt.scatter([np.mean(inner_list) for inner_list in df.x] if isinstance(df.x.iloc[0], list) else list(df.x), lb, label="Lower bound", s=0.1)
    plt.scatter([np.mean(inner_list) for inner_list in df.x] if isinstance(df.x.iloc[0], list) else list(df.x), up, label="Upper bound", s=0.1)
    if 'stratum' in df.columns:
        plt.scatter([np.mean(inner_list) for inner_list in df_as.x] if isinstance(df_as.x.iloc[0], list) else list(df_as.x), mu_y_1_x_as - mu_y_0_x_as, label=r'$\mu_{y(1)|X,AS}-\mu_{y(0)|X,AS}$', s=0.1)

    plt.legend(markerscale=12)
    # plt.legend()
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel(y_title)
    if plot_graph_margin:
        plt.ylim((min(-ylim_margin, min(lb)), max(ylim_margin, max(up))))
    plt.grid(False)
    plt.show()
    return {'x':df.x, 'lb': lb, 'up': up, 'true value': mu_y_1_x_as - mu_y_0_x_as if 'stratum' in df.columns else None}


def plot_zhang_rubin_bounds_no_x(zr_bounds, title, y_title, plot_graph_margin: bool = False, margin=3,
                                 plot_horizontal_zero=False):
    bounds_for_plot = pd.DataFrame({'lb': zr_bounds['lb'], 'up': zr_bounds['up']})
    bounds_for_plot.sort_values(by='up', inplace=True)
    bounds_for_plot.reset_index(inplace=True)

    # what is the percentage of rows in bounds_for_plot that have positive/negative bounds?
    positive_bounds_cont = bounds_for_plot[(bounds_for_plot.lb > 0) & (bounds_for_plot.up > 0)].shape[0]
    negative_bounds_cont = bounds_for_plot[(bounds_for_plot.lb < 0) & (bounds_for_plot.up < 0)].shape[0]
    print(
        f"{positive_bounds_cont} positive bounds ({round(100 * positive_bounds_cont / bounds_for_plot.shape[0], 3)}%)")
    print(
        f"{negative_bounds_cont} negative bounds ({round(100 * negative_bounds_cont / bounds_for_plot.shape[0], 3)}%)")

    plt.scatter(list(bounds_for_plot.index), list(bounds_for_plot.lb), label="Lower bound", s=0.1)
    plt.scatter(list(bounds_for_plot.index), list(bounds_for_plot.up), label="Upper bound", s=0.1)
    if plot_horizontal_zero:
        plt.axhline(y=0, color='grey', linestyle='--', linewidth=1)
    plt.legend(markerscale=12)
    plt.title(title)
    plt.xlabel('Samples')
    plt.ylabel(y_title)
    if plot_graph_margin:
        plt.ylim((min(-margin, min(list(bounds_for_plot.lb))), max(margin, max(list(bounds_for_plot.up)))))
    plt.grid(False)
    plt.show()

################# zhang and rubin parametric bounds ########################
def calculate_integral(func, lb_integration: float, ub_integration: float):
    if lb_integration==ub_integration:
        return 0
    integral_result, estimate_absolute_error = integrate.quad(func=func, a=lb_integration, b=ub_integration)
    return integral_result


def cdf_s_normal(y: float) -> float:
    return 1 / (sqrt(2 * pi)) * exp(-1 / 2 * pow(y, 2))


def integrand_func(sigma: float, mu: float) -> Callable[[float, float], float]:
    return lambda y: y * (1 / (sigma * sqrt(2 * pi))) * exp(-1 / 2 * pow(((y - mu) / sigma), 2))


def calc_zhang_rubin_bounds_per_x(
        x: float, mu_y_0_x: float, mu_y_1_x: float, sigma_0: float, sigma_1: float,
        a0: float, b0: float, c0: float, a1: float, b1: float, c1: float,
        beta_d: List[float], plot_and_print: bool = False, pi_h_len_grid_search: int = 10,
        calc_non_parametric=False) -> Tuple[float, float]:
    beta_d0, beta_d1, beta_d2 = beta_d
    # print(f"mu0: {mu_y_0_x}, mu1: {mu_y_1_x}, sigma0: {sigma_0}, sigma1: {sigma_1}, beta: {beta_d}")

    # pi
    p_t0d0 = 1 - 1 / (1 + exp(-beta_d0 - beta_d2 * x))
    p_t1d0 = 1 - 1 / (1 + exp(-beta_d0 - beta_d1 - beta_d2 * x))
    lower_pi = max(0.0, p_t0d0 - p_t1d0)
    upper_pi = min(p_t0d0, 1 - p_t1d0)
    pi_h_list = np.linspace(lower_pi, upper_pi, num=pi_h_len_grid_search)
    if plot_and_print:
        plot_pi_h(lower_pi, upper_pi)

    # bounds
    zhang_rubin_lb_results = []
    zhang_rubin_ub_results = []
    for pi_h in pi_h_list:
        # Y1
        mu_1_p = a1 + b1 * x
        mu_1_as = mu_1_p + c1
        # QUESTION: Where do we account for the fact that this x resulted in EITHER AS or P? we don't have another x like this with the opposite strata. We are taking weighted average but where does the probability of being AS/P - in terms of Beta - is addressed?
        one_minus_alpha = 1 - p_t0d0 / p_t1d0 + pi_h / p_t1d0
        m = GaussianMixtureDistribution([stats.norm(loc=mu_1_p, scale=sigma_1), stats.norm(loc=mu_1_as, scale=sigma_1)],
                                        [one_minus_alpha, 1 - one_minus_alpha])

        ppf_0 = m.ppf(0)
        ppf_1_minus_weight = m.ppf(1 - one_minus_alpha)
        lb_frst_argmt_integral_p = calculate_integral(
            func=integrand_func(sigma_1, mu_1_p),
            lb_integration=ppf_0,
            ub_integration=ppf_1_minus_weight
        )
        lb_frst_argmt_integral_as = calculate_integral(
            func=integrand_func(sigma_1, mu_1_as),
            lb_integration=ppf_0,
            ub_integration=ppf_1_minus_weight
        )

        if lb_frst_argmt_integral_p==lb_frst_argmt_integral_as==0:
            lb_frst_argmt_integral=0
        else:
            lb_frst_argmt_integral = one_minus_alpha * (1/(1 - one_minus_alpha)) * lb_frst_argmt_integral_p + (1 - one_minus_alpha) * (1/(1 - one_minus_alpha)) * lb_frst_argmt_integral_as

        ppf_weight = m.ppf(one_minus_alpha)
        ppf_1 = m.ppf(1)
        ub_frst_argmt_integral_p = calculate_integral(
            func=integrand_func(sigma_1, mu_1_p),
            lb_integration=ppf_weight,
            ub_integration=ppf_1
        )
        ub_frst_argmt_integral_as = calculate_integral(
            func=integrand_func(sigma_1, mu_1_as),
            lb_integration=ppf_weight,
            ub_integration=ppf_1
        )

        if ub_frst_argmt_integral_p==ub_frst_argmt_integral_as==0:
            ub_frst_argmt_integral=0
        else:
            ub_frst_argmt_integral = one_minus_alpha * (1/(1 - one_minus_alpha)) * ub_frst_argmt_integral_p + (1 - one_minus_alpha) * (1/(1 - one_minus_alpha)) * ub_frst_argmt_integral_as

        # Y0
        mu_0_h = a0 + b0 * x
        mu_0_as = mu_0_h + c0
        # QUESTION: Where do we account for the fact that this x resulted in EITHER AS or H? we don't have another x like this with the opposite strata. We are taking weighted average but where does the probability of being AS/P - in terms of Beta - is addressed?
        one_minus_alpha = pi_h / p_t0d0
        m = GaussianMixtureDistribution([stats.norm(loc=mu_0_h, scale=sigma_0), stats.norm(loc=mu_0_as, scale=sigma_0)],
                                        [one_minus_alpha, 1 - one_minus_alpha])

        ppf_weight = m.ppf(one_minus_alpha)
        ppf_1 = m.ppf(1)
        lb_scnd_argmt_integral_h = calculate_integral(
            func=integrand_func(sigma_0, mu_0_h),
            lb_integration=ppf_weight,
            ub_integration=ppf_1
        )
        lb_scnd_argmt_integral_as = calculate_integral(
            func=integrand_func(sigma_0, mu_0_as),
            lb_integration=ppf_weight,
            ub_integration=ppf_1
        )

        if lb_scnd_argmt_integral_h== lb_scnd_argmt_integral_as==0:
            lb_scnd_argmt_integral=0
        else:
            lb_scnd_argmt_integral = one_minus_alpha * (1/(1 - one_minus_alpha)) * lb_scnd_argmt_integral_h + (1 - one_minus_alpha) * (1/(1 - one_minus_alpha)) * lb_scnd_argmt_integral_as

        ppf_0 = m.ppf(0)
        ppf_1_minus_weight = m.ppf(1 - one_minus_alpha)
        ub_scnd_argmt_integral_h = calculate_integral(
            func=integrand_func(sigma_0, mu_0_h),
            lb_integration=ppf_0,
            ub_integration=ppf_1_minus_weight
        )
        ub_scnd_argmt_integral_as = calculate_integral(
            func=integrand_func(sigma_0, mu_0_as),
            lb_integration=ppf_0,
            ub_integration=ppf_1_minus_weight
        )

        if ub_scnd_argmt_integral_h==ub_scnd_argmt_integral_as==0:
            ub_scnd_argmt_integral=0
        else:
            ub_scnd_argmt_integral = one_minus_alpha * (1/(1 - one_minus_alpha)) * ub_scnd_argmt_integral_h + (1 - one_minus_alpha) * (1/(1 - one_minus_alpha)) * ub_scnd_argmt_integral_as

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


def x_reshape(x, n_features):
    # X should be of shape (n_samples, n_features). If X is a scalar -> #2 dimension set to be 1
    return np.stack(x.values) if n_features > 1 else np.array(x).reshape(-1, 1)

def train_superquantile_model(df: pd.DataFrame) -> tuple[KernelSuperquantileRegressor, KernelSuperquantileRegressor]:
    df_survivors = df.loc[(df.D_obs==0)].copy()
    n_features = np.size(df.x[0])

    # Y_obs_1
    superquantile_model = KernelSuperquantileRegressor(
        kernel=RFKernel(
            RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=n_features,
                min_samples_leaf=min_samples_leaf,
                n_jobs=-2)
        ))
    df_obs_1 = df_survivors.loc[(df_survivors.t == 1)].copy()
    X = x_reshape(df_obs_1.x, n_features)
    Y = np.array(df_obs_1.Y_obs)
    trained_superquantile_model_treated = superquantile_model.fit(X, Y)

    # Y_obs_0
    superquantile_model = KernelSuperquantileRegressor(
        kernel=RFKernel(
            RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=n_features,
                min_samples_leaf=min_samples_leaf,
                n_jobs=-2)
        ))
    df_obs_0 = df_survivors.loc[(df_survivors.t == 0)].copy()
    X = x_reshape(df_obs_0.x, n_features)
    Y = np.array(df_obs_0.Y_obs)
    trained_superquantile_model_untreated = superquantile_model.fit(X, Y)

    return trained_superquantile_model_treated, trained_superquantile_model_untreated

def calc_zhang_rubin_bounds_using_cvar_est(df: pd.DataFrame,
                                           trained_superquantile_model_treated: KernelSuperquantileRegressor,
                                           trained_superquantile_model_untreated: KernelSuperquantileRegressor,
                                           pi_h_len_grid_search: int = 5,
                                           monotonicity_assumption: bool = False, ras_assumption: bool = False) -> \
                                           Tuple[np.array, np.array]:
    n_features = np.size(df.x[0])

    lower_pi_x = np.maximum(np.zeros(df.shape[0]), np.array(df.p_t0d0_x - df.p_t1d0_x))
    upper_pi_x = np.minimum(np.array(df.p_t0d0_x), 1 - np.array(df.p_t1d0_x))
    pi_h_grid_search_per_x = np.array([np.linspace(lower, upper, num=pi_h_len_grid_search) for lower, upper in zip(lower_pi_x, upper_pi_x)])
    # pi_h_grid_search_per_x holds a row per x with the grid values for pi_h (first is the lower bound pi_h, last is the upper bound pi_h, everything in between is the grid)
    grid_search_lb_frst_argmt_per_x = list()
    grid_search_lb_scnd_argmt_per_x = list()
    grid_search_ub_frst_argmt_per_x = list()
    grid_search_ub_scnd_argmt_per_x = list()


    # Y_obs_1
    for pi_h in pi_h_grid_search_per_x.T:
        if ras_assumption:
            X_tau = np.ones(len(pi_h))
        elif monotonicity_assumption:
            X_tau = np.array(df.p_t0d0_x / df.p_t1d0_x)
        else:
            X_tau = np.array(df.p_t0d0_x / df.p_t1d0_x - pi_h / df.p_t1d0_x)
        lb_frst_argmt_per_x = trained_superquantile_model_treated.predict(x_reshape(df.x, n_features), X_tau, tail='left')
        grid_search_lb_frst_argmt_per_x.append(lb_frst_argmt_per_x)

    for pi_h in pi_h_grid_search_per_x.T:
        if monotonicity_assumption:
            X_tau = 1 - np.array(df.p_t0d0_x / df.p_t1d0_x)
        else:
            X_tau = 1 - np.array(df.p_t0d0_x / df.p_t1d0_x) + np.array(pi_h / df.p_t1d0_x)
        ub_frst_argmt_per_x = trained_superquantile_model_treated.predict(x_reshape(df.x, n_features), X_tau, tail='right')
        grid_search_ub_frst_argmt_per_x.append(ub_frst_argmt_per_x)

    # Y_obs_0
    for pi_h in pi_h_grid_search_per_x.T:
        if monotonicity_assumption:
            X_tau = np.zeros(len(pi_h))
        else:
            X_tau = np.array(pi_h / df.p_t0d0_x)
        lb_scnd_argmt_per_x = trained_superquantile_model_untreated.predict(x_reshape(df.x, n_features), X_tau, tail='right')
        grid_search_lb_scnd_argmt_per_x.append(lb_scnd_argmt_per_x)

    for pi_h in pi_h_grid_search_per_x.T:
        if monotonicity_assumption or ras_assumption:
            X_tau = np.ones(len(pi_h))
        else:
            X_tau = 1 - np.array(pi_h / df.p_t0d0_x)
        ub_scnd_argmt_per_x = trained_superquantile_model_untreated.predict(x_reshape(df.x, n_features), X_tau, tail='left')
        grid_search_ub_scnd_argmt_per_x.append(ub_scnd_argmt_per_x)


    zhang_rubin_lb_results = np.array(grid_search_lb_frst_argmt_per_x) - np.array(grid_search_lb_scnd_argmt_per_x)
    zhang_rubin_ub_results = np.array(grid_search_ub_frst_argmt_per_x) - np.array(grid_search_ub_scnd_argmt_per_x)

    zhang_rubin_lb = zhang_rubin_lb_results.min(axis=0)
    zhang_rubin_ub = zhang_rubin_ub_results.max(axis=0)

    return (zhang_rubin_lb, zhang_rubin_ub)


def calc_zhang_rubin_bounds_analytically(df: pd.DataFrame) -> List[Tuple[float]]: #List of the lower bounds and upper bounds
    zhang_rubin_analytic_bounds = list()
    for index, row in df.iterrows():
        if index % 1000 == 0:
            print(f"row {index} out of {df.shape[0]}")
        bound_per_x = calc_zhang_rubin_bounds_per_x(x=row.x, mu_y_0_x=row.mu0, mu_y_1_x=row.mu1, sigma_0=row.sigma_0,
                                              sigma_1=row.sigma_1, a0=row.a0, b0=row.b0, c0=row.c0, a1=row.a1,
                                              b1=row.b1, c1=row.c1, beta_d=row.beta_d, pi_h_len_grid_search=5)

        zhang_rubin_analytic_bounds.append(bound_per_x)
    return list(zip(*zhang_rubin_analytic_bounds))
