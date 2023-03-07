from typing import List
import numpy as np
from numpy import random
from sklearn.linear_model import LogisticRegression

from consts import default_random_seed

def estimate_beta_d_from_realizations(true_beta_d_for_comparison: List[float],
                                      t: np.array, x: np.array, d_obs: np.array) -> List[float]:
    random.seed(default_random_seed)

    features = [[t_i, x_i] for t_i, x_i in zip(t, x)]
    y = d_obs

    clf = LogisticRegression(random_state=0).fit(features, y)

    beta_d_hat = [round(float(clf.intercept_),2)] + [round(beta,2) for beta in list(clf.coef_[0])]
    print(f"beta_d_hat: {beta_d_hat}")
    print(f"(True beta_d: {true_beta_d_for_comparison})")

    # unique, counts = np.unique(clf.predict(features), return_counts=True)
    # print(f"values count for y_hat: {dict(zip(unique, counts))}")
    # unique, counts = np.unique(y, return_counts=True)
    # print(f"values count for : {dict(zip(unique, counts))}")

    return beta_d_hat