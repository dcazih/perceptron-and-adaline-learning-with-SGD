import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.Perceptron import Perceptron
from models.AdalineGD import AdalineGD

def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', edgecolor='k')
    plt.title(title)
    plt.show()


## Preprocess data

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data[:, [2, 3]] # Select only the petal length and petal width as features
y = iris.target # Extract the target labels (species)

# To see class labels
#print('Class labels:', np.unique(y)) # Iris-setosa, Iris-versicolor, and Iris-virginica

# Split iris data into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

# Standardize the features
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


#### Train models

ppn = Perceptron()
ppn.fit(X_train_std, y_train)
print()

ada = AdalineGD()
ada.fit(X_train_std, y_train)
print()


#### Plots to help compare the two models

# Misclassifications
print('Perceptron misclassified examples: %d' % (y_test != ppn.predict(X_test_std)).sum())
print('Adaline misclassified examples: %d' % (y_test != ada.predict(X_test_std)).sum())

# Plot convergence (errors and loss)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.title('Perceptron - Number of updates per epoch')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(ada.losses_) + 1), ada.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Adaline - Loss per epoch')

plt.show()

# Decision boundary plots
plot_decision_boundary(ppn, X_train_std, y_train, "Perceptron Decision Boundary")
plot_decision_boundary(ada, X_train_std, y_train, "Adaline Decision Boundary")

# Print margin (w norm inverse)
ppn_margin = 1 / np.linalg.norm(ppn.w_)
ada_margin = 1 / np.linalg.norm(ada.w_)
print(f"Perceptron margin: {ppn_margin:.4f}")
print(f"Adaline margin: {ada_margin:.4f}")
