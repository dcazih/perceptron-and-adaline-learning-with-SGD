import numpy as np

class AdalineGD:
    """ADAptive LInear NEuron classifier.
    Parameters
    ------------
    eta : float
    Learning rate (between 0.0 and 1.0)
    n_iter : int
    Passes over the training dataset.
    random_state : int
    Random number generator seed for random weight initialization.
    Attributes
    -----------
    w_ : 1d-array
    Weights after fitting.
    b_ : Scalar
    Bias unit after fitting.
    losses_ : list
    Mean squared error loss function values in each epoch.
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.
        Training Simple Machine Learning Algorithms for Classification40
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        Training vectors, where n_examples
        is the number of examples and
        n_features is the number of features.
        y : array-like, shape = [n_examples]
        Target values.
        Returns
        -------
        self : object
        """
        rgen = np.random.RandomState(self.random_state) # Create random generator

        # Initialize Adaline's parameters
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.losses_ = [] # because adaline implements a loss function

        for i in range(self.n_iter):
            # Compute pre/activation functions
            net_input = self.net_input(X)
            output = self.activation(net_input)

            # Compute error
            errors = (y - output)

            # Update weights
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()

            # Update loss
            loss = (errors**2).mean()
            self.losses_.append(loss)

        return self

    # Pre-activation function
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    # Linear activation function
    def activation(self, X):
        """Compute linear activation fucntion"""
        return X

    # Threshold function
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)