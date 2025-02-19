import numpy as np
import os
import scipy
import sklearn
import matplotlib.pyplot as plt
import pandas as pd

# Standard Perceptron
class Perceptron:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = 0 
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_  

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)
    

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
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
X = df.iloc[0:100, [0, 2]].values

# Create regular Perceptron and train it
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

# Create the bias absorbed Perceptron and train it
bA_ppn = Perceptron_bAbsorbed(eta=0.1, n_iter=10)
bA_ppn.fit(X, y)

# Plot the graphs to show comparison
plt.figure(figsize=(8,5))
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o', label="Regular Perceptron")
plt.plot(range(1, len(bA_ppn.errors_) + 1), bA_ppn.errors_, marker='x', linestyle='dashed', label="Bias Absorbed Perceptron")
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.title('Comparison of Learning Progress')
plt.legend()
plt.show()

from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')

# Plot decision boundaries for both models
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('Sepal length (cm)')
plt.ylabel('Petal length (cm)')
plt.title("Regular Perceptron")
plt.legend(loc='upper left')

plt.subplot(1, 2, 2)
plot_decision_regions(X, y, classifier=bA_ppn)
plt.xlabel('Sepal length (cm)')
plt.ylabel('Petal length (cm)')
plt.title("Bias Absorbed Perceptron")
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()