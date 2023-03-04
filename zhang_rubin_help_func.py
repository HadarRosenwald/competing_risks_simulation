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
    y_0_obs = df.loc[df.t == 0]['Y0'].dropna()
    y_1_obs = df.loc[df.t == 1]['Y1'].dropna()
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
                                         y0_dist_param: Dict[str, float] = y0_dist_param_default,
                                         y1_dist_param: Dict[str, float] = y1_dist_param_default):
    df_plot_as = df.loc[df.stratum == Strata.AS.name]
    plt.scatter(df_plot_as.x, (df_plot_as.Y1 - df_plot_as.Y0), label="True Y1 - Y0|AS", s=0.1)

    mu_y_0_x = df.mu0
    mu_y_1_x = df.mu1

    lb, up = zhang_rubin_bounds
    # print(f"lower bound: {lb}, upper bound: {up}, true value: {mu_y_1_x - mu_y_0_x}")
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
    return {'x':df.x, 'lb': lb, 'up': up, 'true value': mu_y_1_x - mu_y_0_x}


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
        weight = 1 - p_t0d0 / p_t1d0 + pi_h / p_t1d0
        m = GaussianMixtureDistribution([stats.norm(loc=mu_1_p, scale=sigma_1), stats.norm(loc=mu_1_as, scale=sigma_1)],
                                        [weight, 1 - weight])

        ppf_0 = m.ppf(0)
        ppf_1_minus_weight = m.ppf(1 - weight)
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
            lb_frst_argmt_integral = weight * (1/(1 - weight)) * lb_frst_argmt_integral_p + (1 - weight) * (1/(1 - weight)) * lb_frst_argmt_integral_as

        ppf_weight = m.ppf(weight)
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
            ub_frst_argmt_integral = weight * (1/(1 - weight)) * ub_frst_argmt_integral_p + (1 - weight) * (1/(1 - weight)) * ub_frst_argmt_integral_as

        # Y0
        mu_0_h = a0 + b0 * x
        mu_0_as = mu_0_h + c0
        # QUESTION: Where do we account for the fact that this x resulted in EITHER AS or H? we don't have another x like this with the opposite strata. We are taking weighted average but where does the probability of being AS/P - in terms of Beta - is addressed?
        weight = pi_h / p_t0d0
        m = GaussianMixtureDistribution([stats.norm(loc=mu_0_h, scale=sigma_0), stats.norm(loc=mu_0_as, scale=sigma_0)],
                                        [weight, 1 - weight])

        ppf_pih_t1t0 = m.ppf(pi_h / p_t1d0)
        ppf_1 = m.ppf(1)
        lb_scnd_argmt_integral_h = calculate_integral(
            func=integrand_func(sigma_0, mu_0_h),
            lb_integration=ppf_pih_t1t0,
            ub_integration=ppf_1
        )
        lb_scnd_argmt_integral_as = calculate_integral(
            func=integrand_func(sigma_0, mu_0_as),
            lb_integration=ppf_pih_t1t0,
            ub_integration=ppf_1
        )

        if lb_scnd_argmt_integral_h== lb_scnd_argmt_integral_as==0:
            lb_scnd_argmt_integral=0
        else:
            lb_scnd_argmt_integral = weight * (1/(1 - pi_h / p_t1d0)) * lb_scnd_argmt_integral_h + (1 - weight) * (1/(1 - pi_h / p_t1d0)) * lb_scnd_argmt_integral_as

        ppf_0 = m.ppf(0)
        ppf_1_minus_pih_t1t0 = m.ppf(1 - pi_h / p_t1d0)
        ub_scnd_argmt_integral_h = calculate_integral(
            func=integrand_func(sigma_0, mu_0_h),
            lb_integration=ppf_0,
            ub_integration=ppf_1_minus_pih_t1t0
        )
        ub_scnd_argmt_integral_as = calculate_integral(
            func=integrand_func(sigma_0, mu_0_as),
            lb_integration=ppf_0,
            ub_integration=ppf_1_minus_pih_t1t0
        )

        if ub_scnd_argmt_integral_h==ub_scnd_argmt_integral_as==0:
            ub_scnd_argmt_integral=0
        else:
            ub_scnd_argmt_integral = weight * (1/(1 - pi_h / p_t1d0)) * ub_scnd_argmt_integral_h + (1 - weight) * (1/(1 - pi_h / p_t1d0)) * ub_scnd_argmt_integral_as

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


def calc_zhang_rubin_bounds_using_cvar_est(df: pd.DataFrame, pi_h_len_grid_search: int = 5) -> Tuple[np.array, np.array]:
    p_t0d0, p_t1d0 = calc_non_parametric_p_t0d0_p_t1d0(df)
    lower_pi = max(0.0, p_t0d0 - p_t1d0)
    upper_pi = min(p_t0d0, 1 - p_t1d0)
    pi_h_list = np.linspace(lower_pi, upper_pi, num=pi_h_len_grid_search)

    # bounds
    zhang_rubin_lb_results = []
    zhang_rubin_ub_results = []
    for pi_h in pi_h_list:

        # TODO re-considerate our train and predict data set. Currently only survivors. Is that the right way?

        df_survivors = df.loc[(df.D_obs==0)].copy()

        # Y_obs_1
        df_obs_1 = df_survivors.loc[(df.t==1)].copy()
        X = np.array(df_obs_1.x).reshape(-1, 1) # X should be of shape (n_samples, n_features). Since X is currently 1D -> #2 dimension set to be 1
        Y = np.array(df_obs_1.Y_obs)

        superquantile_model = KernelSuperquantileRegressor(
            kernel=RFKernel(
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    max_features=max_features,
                    min_samples_leaf=min_samples_leaf,
                    n_jobs=-2)
            ),
            tail='left')

        trained_superquantile_model = superquantile_model.fit(X, Y)

        X_tau = np.array(df_survivors.p_t0d0_x / df_survivors.p_t1d0_x - pi_h / df_survivors.p_t1d0_x)
        lb_frst_argmt_per_x = trained_superquantile_model.predict((np.array(df_survivors.x).reshape(-1, 1)), X_tau)

        superquantile_model = KernelSuperquantileRegressor(
            kernel=RFKernel(
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    max_features=max_features,
                    min_samples_leaf=min_samples_leaf,
                    n_jobs=-2)
            ),
            tail='right')

        trained_superquantile_model = superquantile_model.fit(X, Y)

        X_tau = 1- np.array(df_survivors.p_t0d0_x / df_survivors.p_t1d0_x) + np.array(pi_h / df_survivors.p_t1d0_x)
        ub_frst_argmt_per_x = trained_superquantile_model.predict(np.array(df_survivors.x).reshape(-1, 1), X_tau)

        # Y_obs_0
        df_obs_0 = df_survivors.loc[(df.t == 0)].copy()
        X = np.array(df_obs_0.x).reshape(-1, 1) # X should be of shape (n_samples, n_features). Since X is currently 1D -> #2 dimension set to be 1
        Y = np.array(df_obs_0.Y_obs)

        superquantile_model = KernelSuperquantileRegressor(
            kernel=RFKernel(
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    max_features=max_features,
                    min_samples_leaf=min_samples_leaf,
                    n_jobs=-2)
            ),
            tail='right')

        trained_superquantile_model = superquantile_model.fit(X, Y)

        X_tau = np.array(pi_h / df_survivors.p_t1d0_x)
        lb_scnd_argmt_per_x = trained_superquantile_model.predict(np.array(df_survivors.x).reshape(-1, 1), X_tau)

        superquantile_model = KernelSuperquantileRegressor(
            kernel=RFKernel(
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    max_features=max_features,
                    min_samples_leaf=min_samples_leaf,
                    n_jobs=-2)
            ),
            tail='left')

        trained_superquantile_model = superquantile_model.fit(X, Y)

        X_tau = 1- np.array(pi_h / df_survivors.p_t1d0_x)
        ub_scnd_argmt_per_x = trained_superquantile_model.predict(np.array(df_survivors.x).reshape(-1, 1), X_tau)


        zhang_rubin_lb = lb_frst_argmt_per_x - lb_scnd_argmt_per_x
        zhang_rubin_ub = ub_frst_argmt_per_x - ub_scnd_argmt_per_x

        zhang_rubin_lb_results.append(zhang_rubin_lb)
        zhang_rubin_ub_results.append(zhang_rubin_ub)

    zhang_rubin_lb = np.array(zhang_rubin_lb_results).min(axis=0) #TODO - zhang_rubin_lb_results is a matrix of #pi rows and #x column. Here we take the min result per x (it can be different pi's for different x's. Is that ok?)
    zhang_rubin_ub = np.array(zhang_rubin_ub_results).max(axis=0)

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
