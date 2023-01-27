import numpy as np


class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications (updates) in each epoch.

    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    # Train the perceptron
    def fit(self, X, y):
        """Fits training data.
        This function take the input of features,
        make predictions, update the weights n_iter times.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """

        # initializing the weights
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        # print("Weights:", self.w_)

        for _ in range(self.n_iter):
            error = 0
            for xi, target in zip(X, y):

                # 1. calculate y^
                y_pred = self.predict(xi)

                # 2. calculate Update
                # update = η * (y - y^)
                update = self.eta * (target - y_pred)

                # 3. Update the weights
                # Wi = Wi + Δ(Wi)       where  Δ(Wi) = update * Xi
                self.w_[1:] = self.w_[1:] + update * xi
                # print(self.w_[1:])

                # Xo = 1 => update * 1 = update
                self.w_[0] = self.w_[0] + update

                # update != 0  ==>  y^ != y  ==>  there is an error
                error += int(update != 0.0)

            self.errors_.append(error)

        return self

    # Weighted Sum
    def net_input(self, X):
        """
        The net_input() function returns the dot product
        i.e. the weighted sum: sum of w*x
        :param: X: an array of features (inputs)
        Returns:
            the dot product of W and X
        """
        # Z = sum (Wi * Xi) where Wo = 0 et Xo = 1
        return np.dot(X, self.w_[1:]) + self.w_[0]

    # Step Function
    def predict(self, X):
        """
        The activation function.
        Returns the predicted value
        :param X: an array of features
        :return:
            y_pred - the predicted value:
                0 if the weighted sum < 0
                1 otherwise
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
