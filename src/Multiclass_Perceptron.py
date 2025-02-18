import numpy as np
import os
import scipy
import sklearn
import matplotlib.pyplot as plt
import pandas as pd

# Perceptron model with the bias b_ absorbed
class Perceptron_bAbsorbed:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        # Make extra space for the now absorbed bias
        features = X.shape[1] + 1
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=features)
        # No Bias data field
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                xiBias = np.append(xi, 1) # Get bias term
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xiBias
                # No bias update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
        
    def net_input(self, X):
        # Last weight used as bias
        return np.dot(X, self.w_[:-1]) + self.w_[-1]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

# Read in iris data
s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(s, header=None, encoding='utf-8')

# Make variables of iris data
y = df.iloc[:, 4].values
X = df.iloc[:, 0:4].values

# Make Setosa=0, Versicolor=1, Virginica=2
class_mapping = {label: idx for idx, label in enumerate(np.unique(y))}
y_encoded = np.array([class_mapping[label] for label in y])

# Put each flower against the others
y_setosa = np.where(y_encoded == 0, 1, 0)      # Setosa vs. Others
y_versicolor = np.where(y_encoded == 1, 1, 0)  # Versicolor vs. Others
y_virginica = np.where(y_encoded == 2, 1, 0)   # Virginica vs. Others

# Train 3 different biases absorbed Perceptrons
ppn_setosa = Perceptron_bAbsorbed(eta=0.1, n_iter=500)
ppn_versicolor = Perceptron_bAbsorbed(eta=0.1, n_iter=500)
ppn_virginica = Perceptron_bAbsorbed(eta=0.1, n_iter=500)

# Fit all with x and y
ppn_setosa.fit(X, y_setosa)
ppn_versicolor.fit(X, y_versicolor)
ppn_virginica.fit(X, y_virginica)

# Make predictions from all Perceptrons
def predict_multiclass(X):
    # Predict class by choosing highest net input
    net_setosa = ppn_setosa.net_input(X)
    net_versicolor = ppn_versicolor.net_input(X)
    net_virginica = ppn_virginica.net_input(X)

    # Find the class with the highest activation
    return np.array([
        np.argmax([net_setosa[i], net_versicolor[i], net_virginica[i]])
        for i in range(len(X))
    ])

# Look at accuracy
y_pred = predict_multiclass(X)
accuracy = np.mean(y_pred == y_encoded) * 100
print(f"Training Accuracy: {accuracy:.2f}%")


