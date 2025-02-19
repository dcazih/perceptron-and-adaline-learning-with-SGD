import numpy as np


class AdalineSGD:
    """ADAptive LInear NEuron classifier also with Mini-batch Stochastic Gradient Descent.

    Parameters
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    batch_size : int
        Number of training examples in each mini-batch.
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent cycles.
    random_state : int
        Random number generator seed for random weight initialization.

    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting.
    b_ : Scalar
        Bias unit after fitting.
    losses_ : list
        Mean squared error loss function value averaged over all training examples in each epoch.
    """

    def __init__(self, eta=0.05, n_iter=100, batch_size=32, shuffle=True, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.w_initialized = False
        self.losses_ = []

    def fit_SGD(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of examples
            and n_features is the number of features.
        y : array-like, shape = [n_examples]
            Target values.

        Returns
        -------
        self : object
        """
        self._initialize_weights(X.shape[1])
        self.losses_ = []

        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = []
            for xi, target in zip(X, y):
                losses.append(self._update_weights(xi, target))
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)

        return self

    def fit_mini_batch_SGD(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of examples
            and n_features is the number of features.
        y : array-like, shape = [n_examples]
            Target values.

        Returns
        -------
        self : object
        """
        self._initialize_weights(X.shape[1])

        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)

            losses = []
            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]
                loss = self._update_weightsMB(X_batch, y_batch)
                losses.append(loss)

            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)

        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights."""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])

        return self._update_weights(X, y)

    def _shuffle(self, X, y):
        """Shuffle training data."""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """Initialize weights to small random numbers."""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float_(0.)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights."""
        output = self.activation(self.net_input(xi))
        error = target - output
        self.w_ += self.eta * 2.0 * xi * error
        self.b_ += self.eta * 2.0 * error
        loss = error ** 2
        return loss

    def _update_weightsMB(self, X_batch, y_batch):
        """Apply Adaline learning rule to update the weights using mini-batches."""
        outputs = self.activation(self.net_input(X_batch))
        errors = y_batch - outputs

        # Compute gradient over batch
        self.w_ += self.eta * 2.0 * np.dot(X_batch.T, errors) / len(y_batch)
        self.b_ += self.eta * 2.0 * np.mean(errors)

        loss = np.mean(errors ** 2)
        return loss

    def net_input(self, X):
        """Compute net input."""
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        """Linear activation function."""
        return X

    def predict(self, X):
        """Return class labels after thresholding."""
        return np.where(self.net_input(X) >= 0.0, 1, 0)
