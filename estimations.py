from typing import List, Optional

import pandas as pd
from numpy import random
from sklearn.linear_model import LogisticRegression

from consts import default_random_seed
from sample_generation import create_sample

random.seed(default_random_seed)

def estimate_beta_d_from_realizations(true_beta_d_for_estimation: List[float],
                                      df: Optional[pd.DataFrame] = None) -> List[float]:
    if type(df) != pd.DataFrame and df == None:
        df = create_sample(beta_d = true_beta_d_for_estimation)

    features = [[t_i, x_i] for t_i, x_i in zip(list(df.t), list(df.x))]
    # y = df['D0'].where(df['t'] == 0, df['D1']) # observed D
    y = df['D_obs']

    clf = LogisticRegression(random_state=0).fit(features, y)

    beta_d_hat = [round(float(clf.intercept_),2)] + [round(beta,2) for beta in list(clf.coef_[0])]
    print(f"beta_d_hat: {beta_d_hat}")
    print(f"(True beta_d: {true_beta_d_for_estimation})")

    # unique, counts = np.unique(clf.predict(features), return_counts=True)
    # print(f"values count for y_hat: {dict(zip(unique, counts))}")
    # unique, counts = np.unique(y, return_counts=True)
    # print(f"values count for : {dict(zip(unique, counts))}")

    return beta_d_hat