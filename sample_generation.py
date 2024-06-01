from estimations import estimate_beta_d_from_realizations

from typing import List, Tuple
import pandas as pd
from consts import *
from distributions import *
from policy import policy_treat_by_zr_bounds, policy_treat_by_ignoring_trunc
from strata import Strata, get_strata
import pyreadstat
from scipy.stats import skew
from sklearn.linear_model import LogisticRegression, LinearRegression

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


def calc_logistic_proba(beta, x_row, t, omega=omega_default):
    bias = beta[0] + beta[1] * t
    x_row = np.array(x_row, ndmin=1)  # Ensures x_row is at least 1D
    weighted_features = sum(beta[i+2] * x_value for i, x_value in enumerate(x_row))
    logit = bias + weighted_features
    return 1 / (1 + np.exp(-omega * logit))


def calc_death_array_and_proba(x: np.array, beta_d,
                               omega: float = omega_default) -> Tuple[np.array, np.array, np.array, np.array]:
    D0_prob = [calc_logistic_proba(beta_d, x[i], t=0, omega=omega) for i in range(len(x))]
    D1_prob = [calc_logistic_proba(beta_d, x[i], t=1, omega=omega) for i in range(len(x))]

    D0 = BernoulliDist(n=x.size, param={'p': D0_prob}).sampled_vector
    D1 = BernoulliDist(n=x.size, param={'p': D1_prob}).sampled_vector

    p_t0d0_x = 1 - np.array(D0_prob)
    p_t1d0_x = 1 - np.array(D1_prob)

    return D0, D1, p_t0d0_x, p_t1d0_x


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

    variables_dict['t'] = BernoulliDist(n=population_size, param={'p': treatment_prob}) .sampled_vector

    D0, D1, p_t0d0_x, p_t1d0_x = calc_death_array_and_proba(variables_dict['x'], beta_d, omega)
    variables_dict['D0'] = D0
    variables_dict['D1'] = D1
    variables_dict['p_t0d0_x'] = p_t0d0_x
    variables_dict['p_t1d0_x'] = p_t1d0_x

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


def impute_numeric_column(column):
    skewness = skew(column.dropna())
    # If skewness is significant, use median, otherwise mean
    if abs(skewness) > 0.5:
        return column.fillna(column.median())
    else:
        return column.fillna(column.mean())

def data_adjustments(dataset: str) -> pd.DataFrame:
    if dataset not in ('lalonde', 'STAR'):
        # Placeholder for other RWD sources
        return None
    elif dataset == 'lalonde':
        control = pd.read_table('nsw_control.txt', header=None,
                                names=['t', 'age', 'educ', 'black', 'hispan', 'married', 'nodegree', 're75', 're78'],
                                sep='\s+')
        treated = pd.read_table('nsw_treated.txt', header=None,
                                names=['t', 'age', 'educ', 'black', 'hispan', 'married', 'nodegree', 're75', 're78'],
                                sep='\s+')
        df = pd.concat([control, treated], ignore_index=True)
        df_adjusted = pd.DataFrame(data={
            'x': df[['age', 'educ', 'black', 'hispan', 'married', 'nodegree', 're75']].values.tolist(),
            't': df.t.values.tolist(),
            'D_obs': df['re78'].apply(lambda D_obs: 1 if D_obs == 0 else 0).tolist(),
            'Y_obs': df['re78'].apply(lambda Y_obs: None if Y_obs == 0 else Y_obs).tolist(),
        })
    elif dataset == 'STAR':
        df, meta = pyreadstat.read_sav('STAR_Students.sav')
        dfk = df.loc[(df.FLAGSGK == 1)].copy()
        relevant_grades_list = ['g1treadss', 'g1wordskillss', 'g1tmathss', 'g1readbsraw', 'g1mathbsraw']

        # remove students that changed arm or have missing grades
        starg1_changed_from_small = (dfk.FLAGSG1 == 1) & (dfk.gkclasstype == 1.0) & (dfk.g1classtype != 1)
        starg1_changed_from_regular = (dfk.FLAGSG1 == 1) & (dfk.gkclasstype != 1.0) & (dfk.g1classtype == 1.0)
        starg1_no_grades = ((dfk.FLAGSG1 == 1) & dfk[relevant_grades_list].isnull().all(axis=1))
        dfk = dfk.loc[~(starg1_changed_from_small | starg1_changed_from_regular | starg1_no_grades)]

        dfk['freelunch'] = dfk['gkfreelunch'].apply(lambda x: 1 if x == 1.0 else (0 if not pd.isnull(x) else x))
        dfk['white'] = dfk['race'].apply(lambda x: 1 if x == 1.0 else (
            0 if not pd.isnull(x) else x))  # dfk.race.value_counts(dropna=False) <- 4234 white, 2058 black, 33 other
        dfk['age_in_1985'] = 1985 - dfk['birthyear']
        dfk['girl'] = dfk['gender'].apply(lambda x: 1 if x == 2.0 else (0 if not pd.isnull(x) else x))
        dfk['teacher_white'] = dfk['gktrace'].apply(lambda x: 1 if x == 1.0 else (0 if not pd.isnull(x) else x))
        # 'gktyears'
        dfk['teacher_master'] = dfk['gkthighdegree'].apply(
            lambda x: 1 if (not pd.isnull(x) and x > 2.0) else (0 if not pd.isnull(x) else x))
        dfk['teacher_girl'] = dfk['gktgen'].apply(lambda x: 1 if x == 2.0 else (0 if not pd.isnull(x) else x))

        dfk['t'] = dfk['gkclasstype'].apply(lambda x: 1 if x == 1.0 else (0 if not pd.isnull(x) else x))

        dfk['D'] = (dfk.FLAGSG1 == 0) | dfk[relevant_grades_list].isnull().all(axis=1).astype(int)
        # if we removed studets with missing grades, then dfk.FLAGSG1 == 0 is enough.

        dfk['avg_percentile'] = dfk[relevant_grades_list].rank(pct=True).mean(axis=1)

        # Fill na
        features_list = ['freelunch', 'white', 'age_in_1985', 'girl', 'teacher_white', 'gktyears', 'teacher_master',
                         'teacher_girl']
        numeric_features = ['age_in_1985', 'gktyears']
        categorical_features = [f for f in features_list if f not in numeric_features]
        dfk_features_imputed = dfk[features_list].copy()
        for feature in numeric_features:
            dfk_features_imputed[feature] = impute_numeric_column(dfk_features_imputed[feature])
        for feature in categorical_features:
            dfk_features_imputed[feature] = dfk_features_imputed[feature].fillna(
                dfk_features_imputed[feature].mode()[0])

        df_adjusted = pd.DataFrame(data={
            'x': dfk_features_imputed[features_list].values.tolist(),
            't': dfk.t.values.tolist(),
            'D_obs': dfk.D.values.tolist(),
            'Y_obs': dfk.avg_percentile.values.tolist(),
        })

    return df_adjusted


def simulate_counterfactual_D(df_arm, counter_clf, beta_d, beta_y):
    # simulate D0 with X, D1, Y1*D1 and vise versa
    att = np.hstack([np.ones(shape=(df_arm.shape[0], 1)),
                     np.vstack(df_arm.x),
                     df_arm[['D_obs', 'Y_obs']].fillna(0).to_numpy()])
    beta = np.concatenate(([counter_clf.intercept_[0]],
                           counter_clf.coef_.ravel(),
                           [beta_d, beta_y]))
    logits = np.dot(att, beta).astype(float)
    D_prob = 1 / (1 + np.exp(-logits))
    cf_D = BernoulliDist(n=len(D_prob), param={'p': D_prob}).sampled_vector
    return cf_D

def simulate_counterfactual_Y(df_arm, counter_rl, beta_d, beta_y, col):
    # simulate Y0 with X, D0, D1, Y1*D1 and vise versa
    att = np.hstack([np.ones(shape=(df_arm.shape[0], 1)),
                     np.vstack(df_arm.x),
                     df_arm[col].fillna(0).to_numpy()])
    beta = np.concatenate(([counter_rl.intercept_],
                           counter_rl.coef_.ravel(),
                           [beta_d, beta_y]))
    logits = np.dot(att, beta).astype(float)
    Y_cf = 1 / (1 + np.exp(-logits))
    return Y_cf


def simulate_counterfactuals(df, beta_d=1.0, beta_y=1.0):
    # simulating D0 for those with t=1

    treatment_df = df[df.t == 1].copy()
    control_df = df[df.t == 0].copy()

    control_clf = LogisticRegression(random_state=0).fit(np.vstack(control_df.x), control_df.D_obs.to_numpy())
    treatment_clf = LogisticRegression(random_state=0).fit(np.vstack(treatment_df.x), treatment_df.D_obs.to_numpy())

    # simulate D0 with X, D1, Y1 and vise versa.
    # The betas for X is driven from the classifiers, the betas for D and Y are set as input.
    treatment_df['D1'] = treatment_df['D_obs'].astype(int)
    treatment_df['D0'] = simulate_counterfactual_D(treatment_df, control_clf, beta_d, beta_y)
    treatment_df['stratum'] = treatment_df.apply(lambda row: get_strata(d0=row['D0'], d1=row['D1']), axis=1)

    control_df['D0'] = control_df['D_obs'].astype(int)
    control_df['D1'] = simulate_counterfactual_D(control_df, treatment_clf, beta_d, beta_y)
    control_df['stratum'] = control_df.apply(lambda row: get_strata(d0=row['D0'], d1=row['D1']), axis=1)

    df_cf = pd.concat([control_df, treatment_df])
    strata_counts = df_cf['stratum'].value_counts()
    strata_info = pd.DataFrame({'counts': strata_counts, 'percentage': 100*strata_counts / len(df_cf)})
    print(strata_info)

    control_df_s = control_df[control_df.D0 == 0]
    rl_att = np.hstack((np.vstack(control_df_s.x), control_df_s.D1.values.reshape(-1, 1)))
    control_lr = LinearRegression().fit(rl_att, control_df_s.Y_obs.to_numpy())
    treatment_df_s = treatment_df[treatment_df.D1 == 0]
    rl_att = np.hstack((np.vstack(treatment_df_s.x), treatment_df_s.D0.values.reshape(-1, 1)))
    treatment_lr = LinearRegression().fit(rl_att, treatment_df_s.Y_obs.to_numpy())

    treatment_df['Y1'] = treatment_df['Y_obs']
    treatment_df['Y0'] = simulate_counterfactual_Y(treatment_df, control_lr, beta_d, beta_y, ['D1', 'D0', 'Y1'])
    treatment_df.loc[treatment_df['D0'] == 1, 'Y0'] = None

    control_df['Y0'] = control_df['Y_obs']
    control_df['Y1'] = simulate_counterfactual_Y(control_df, treatment_lr, beta_d, beta_y, ['D0', 'D1', 'Y0'])
    control_df.loc[control_df['D1'] == 1, 'Y1'] = None

    df_cf = pd.concat([control_df, treatment_df])
    df_cf['mu0'] = df_cf['Y0'].mean()
    df_cf['mu1'] = df_cf['Y1'].mean()

    return df_cf


def adjust_data_with_beta_estimations(df):
    beta_d_hat = estimate_beta_d_from_realizations(
        true_beta_d_for_comparison=df.iloc[0].beta_d if 'beta_d' in df.columns else None,
        t=df.t.values,
        x=df.x.values,
        d_obs=df.D_obs.values
    )

    _, _, p_t0d0_x_hat, p_t1d0_x_hat = calc_death_array_and_proba(df.x.values, beta_d_hat)
    adjusted_df = df[['x','D_obs','Y_obs','t']].copy()
    adjusted_df['p_t0d0_x'] = p_t0d0_x_hat
    adjusted_df['p_t1d0_x'] = p_t1d0_x_hat
    return adjusted_df
