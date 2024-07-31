import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier

from strata import Strata


def policy_treat_by_D_cate(df: pd.DataFrame, cate_D_threshold: float = 0.55):
    features = pd.DataFrame(df.x.tolist(), columns=[f'x{i}' for i in range(len(df.x.iloc[0]))])
    features['t'] = df.t
    model = XGBClassifier(random_state=0, eval_metric='logloss')
    model.fit(features, df.D_obs)

    features['t'] = 1
    treated_probs = model.predict_proba(features)[:, 1]
    features['t'] = 0
    control_probs = model.predict_proba(features)[:, 1]

    df['cate_D'] = treated_probs - control_probs
    df['pi_cate_D'] = [0 if cate > cate_D_threshold else 1 for cate in df['cate_D']]
    df['pi_cate_D_Y_value'] = np.where(df['pi_cate_D'] == 1, df['Y1'], df['Y0'])
    df['pi_cate_D_D_value'] = np.where(df['pi_cate_D'] == 1, df['D1'], df['D0'])

    as_df = df.loc[df.stratum == Strata.AS].copy()
    return df.pi_cate_D_Y_value.mean(), 1-df.pi_cate_D_D_value.sum()/df.pi_cate_D_D_value.count(), as_df.pi_cate_D_Y_value.mean()


def policy_treat_by_ignoring_trunc(df: pd.DataFrame):
    df_s = df.loc[df.D_obs == 0].copy()
    df_s.reset_index(inplace=True)
    features = pd.DataFrame(df_s['x'].tolist(), columns=[f'x{i}' for i in range(len(df_s['x'][0]))])
    features['treatment'] = df_s['t']
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(features, df_s['Y_obs'])

    # inference - for all the samples (not just the truncated)
    features = pd.DataFrame(df['x'].tolist(), columns=[f'x{i}' for i in range(len(df['x'][0]))])
    features['treatment'] = 1
    Y_treatment_pred = rf.predict(features)
    features['treatment'] = 0
    Y_control_pred = rf.predict(features)

    df['cate_Y'] = Y_treatment_pred - Y_control_pred
    df['pi_cate_Y'] = [1 if cate > 0 else 0 for cate in df['cate_Y']]
    df['pi_cate_Y_Y_value'] = np.where(df['pi_cate_Y'] == 1, df['Y1'], df['Y0'])
    df['pi_cate_Y_D_value'] = np.where(df['pi_cate_Y'] == 1, df['D1'], df['D0'])

    as_df = df.loc[df.stratum == Strata.AS].copy()

    return df.pi_cate_Y_Y_value.mean(), 1-df.pi_cate_Y_D_value.sum()/df.pi_cate_Y_D_value.count(), as_df.pi_cate_Y_Y_value.mean()


def policy_treat_by_composite_outcome(df: pd.DataFrame):
    features = pd.DataFrame(df['x'].tolist(), columns=[f'x{i}' for i in range(len(df['x'][0]))])
    features['treatment'] = df['t']
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    label = df['Y_obs'].fillna(df['Y_obs'].mean()) # TODO fillna with 0
    rf.fit(features, label)

    features['treatment'] = 1
    Y_treatment_pred = rf.predict(features)
    features['treatment'] = 0
    Y_control_pred = rf.predict(features)

    df['cate_co'] = Y_treatment_pred - Y_control_pred
    df['pi_co'] = [1 if cate > 0 else 0 for cate in df['cate_co']]
    df['pi_co_Y_value'] = np.where(df['pi_co'] == 1, df['Y1'], df['Y0'])
    df['pi_co_D_value'] = np.where(df['pi_co'] == 1, df['D1'], df['D0'])

    df_as = df.loc[df.stratum == Strata.AS].copy()

    return df.pi_co_Y_value.mean(), 1-df.pi_co_D_value.sum()/df.pi_co_D_value.count(), df_as.pi_co_Y_value.mean()


def policy_treat_by_zr_bounds(df, lb, ub, lb_threshold=-0.25):
    # ub_threshold = 0.3

    df['zr_lb'] = lb
    df['pi_zr'] = [int(lb_x > lb_threshold) for lb_x in lb]
    df['pi_zr_Y_value'] = np.where(df['pi_zr'] == 1, df['Y1'], df['Y0'])
    df['pi_zr_D_value'] = np.where(df['pi_zr'] == 1, df['D1'], df['D0'])

    df_as = df.loc[df.stratum == Strata.AS].copy()

    return df.pi_zr_Y_value.mean(), 1-df.pi_zr_D_value.sum()/df.pi_zr_D_value.count(), df_as.pi_zr_Y_value.mean()
