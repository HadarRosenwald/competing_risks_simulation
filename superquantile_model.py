import numpy as np

n_estimators = 100
max_depth = 7
max_features = 1 # max_features must be in (0, n_features]
min_samples_leaf = 10
class KernelSuperquantileRegressor:

    def __init__(self, kernel, tail='left'):
        self.kernel = kernel
        if tail not in ["left", "right"]:
            raise ValueError(
                f"The 'tail' parameter can only take values in ['left', 'right']. Got '{tail}' instead.")
        self.tail = tail

    def fit(self, X, Y):
        self.sorted_Y_idx = np.argsort(Y)
        self.sorted_Y = Y[self.sorted_Y_idx]
        self.kernel.fit(X[self.sorted_Y_idx], Y[self.sorted_Y_idx])
        return self

    def predict(self, X, X_tau):
        preds = np.empty(X.shape[0])
        sorted_weights = self.kernel.predict(X)
        for i, (x, tau) in enumerate(zip(X,X_tau)):
            if self.tail == "right":
                idx_tail = np.where((np.cumsum(sorted_weights[i]) >= tau) == True)[0]
                preds[i] = np.sum(self.sorted_Y[idx_tail] * sorted_weights[i][idx_tail]) / (1 - tau)
            else:
                idx_tail = np.where((np.cumsum(sorted_weights[i]) <= tau) == True)[0]
                preds[i] = np.sum(self.sorted_Y[idx_tail] * sorted_weights[i][idx_tail]) / tau
        return preds


class RFKernel:

    def __init__(self, rf):
        self.rf = rf

    def fit(self, X, Y):
        self.rf.fit(X, Y)
        self.train_leaf_map = self.rf.apply(X)

    def predict(self, X):
        weights = np.empty((X.shape[0], self.train_leaf_map.shape[0]))
        leaf_map = self.rf.apply(X)
        for i, x in enumerate(X):
            P = (self.train_leaf_map == leaf_map[[i]])
            weights[i] = (1. * P / P.sum(axis=0)).mean(axis=1)
        return weights