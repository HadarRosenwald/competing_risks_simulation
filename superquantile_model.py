import numpy as np

n_estimators = 100
max_depth = 7
min_samples_leaf = 10
class KernelSuperquantileRegressor:

    def __init__(self, kernel):
        self.kernel = kernel

    def fit(self, X, Y):
        self.sorted_Y_idx = np.argsort(Y)
        self.sorted_Y = Y[self.sorted_Y_idx]
        self.kernel.fit(X[self.sorted_Y_idx], Y[self.sorted_Y_idx])
        return self

    def predict(self, X, X_tau, tail='left'):
        """
        The superquantile, also known as CVaR, is a measure of the risk of extreme events  in the "tail" of the
        statistical distribution.
        """

        if tail not in ["left", "right"]:
            raise ValueError(
                f"The 'tail' parameter can only take values in ['left', 'right']. Got '{tail}' instead.")

        preds = np.empty(X.shape[0])
        sorted_weights = self.kernel.predict(X)
        for i, (x, tau) in enumerate(zip(X,X_tau)):
            if tail == "right":
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
        """
        Running regular random forest fit, and adding train_leaf_map that stores the locations of the leaf nodes for
        each sample in the training data.
        Explanation: 'apply' method applies trees in the forest to X, and returns leaf indices (in train_leaf_map);
        For each datapoint x in X and for each tree in the forest, return the index of the leaf x ends up in.
        A matrix of shape (n_samples, n_estimators).
        """
        self.rf.fit(X, Y)
        self.train_leaf_map = self.rf.apply(X)

    def predict(self, X):
        """
        Analougous to Non-parametric density estimation techniques, such as kernel density estimation (KDE).
        The weights in the model measure how "similar" or "close" a test sample is to the training samples, with
        similarity based on the structure of the decision trees in the random forest.
        When a test sample falls into the same leaves as many training samples, it suggests that the region of the
        feature space around this test sample is densely populated. And density implies probability (probability
        density functions are essentially histograms that have been normalized and smoothed). This density is
        indicative of the likelihood of observing a sample with features X and outcome Y (essentially ~P(Y|X)).
        If the training data are representative of the underlying distribution, then areas with higher concentrations
        of training samples (in terms of feature space) are areas where the outcome Y is more probable for given X
        values.


        Regular random forest regression's 'predict' would return the avg of the samples' values in the leaf where a
        new test sample landed. Here it works differently.

        weights[i] represents how "similar" the i'th test sample is to each of the training samples, according to the
        Random Forest model; Given the i'th test sample, the RFKernel generates a weight vector of the same size as
        the training data set. Each element in this weight vector corresponds to the similarity between each sample of
        the training set and the i'th test sample. The similarity is determined by how often the test sample and a
        training sample end up in the same leaves averaging all the trees in the Random Forest.

        Instead of predict returning a single value (average), it will provide a normalized weight vector indicating
        the similarity between each test sample and all training samples.


        train_leaf_map has the leaf indices for each survived with T=t. For all survived (any t) (=test set), we check
        element wise if train and predict agree on the leaf location for each x.

        P is a binary matrix of agreement between a certain test sample and all of the training samples.
        In each iteration i:
        P[j,k]=1 iff tree k had the same leaf mapping for sample j from the train set as the i'th test sample.

        P.sum(axis=0) <- for each tree, how many samples in train had the same leaf mapping like our i'th test sample.
        1.*P/P.sum(axis=0) <- for each tree (each column) we create a distribution of the ‘agreement’s; each [j, k]
        entry represents the proportion of "agreements" for the k-th tree that involve the j-th training sample.
        Then we average for each training sample across the different trees.
        1.*P/P.sum(axis=0) is a "normalized" version of P where each column(=tree) sum up to 1.
        Averaging the rows (across the different trees) - will also end up in a normalized array that will sum up to 1.

        weights[i] has an element for each sample in the training sample, and it represents the average proportion of
        "agreements" that training sample has with the current test sample across all trees. Each element quantifies
        the average "similarity" between a training sample and the current test sample as determined by all
        trees in the random forest.

        The weights essentially measure how often a given test sample falls into the same leaf nodes as the training
        samples. The logic being, if two samples often end up in the same leaves, they are similar and should have
        similar target values.
        """
        weights = np.empty((X.shape[0], self.train_leaf_map.shape[0]))
        leaf_map = self.rf.apply(X)
        for i, x in enumerate(X):  # for each test sample
            P = (self.train_leaf_map == leaf_map[[i]])  # which train mapping agree? in each column=tree.
            weights[i] = (1. * P / P.sum(axis=0)).mean(axis=1)
        return weights