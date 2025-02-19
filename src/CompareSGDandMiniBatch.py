import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.AdalineSGD import AdalineSGD


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


adaline1 = AdalineSGD()
adaline1.fit_SGD(X_train_std, y_train)

adaline2 = AdalineSGD()
adaline2.fit_mini_batch_SGD(X_train_std, y_train)

# Misclassifications
print('SGD misclassified examples: %d' % (y_test != adaline1.predict(X_test_std)).sum())
print('Mini-Batch SGD misclassified examples: %d' % (y_test != adaline2.predict(X_test_std)).sum())

# Plot convergence (errors and loss)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(adaline1.losses_) + 1), adaline1.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.title('SGD - Loss per epoch')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(adaline2.losses_) + 1), adaline2.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Mini-Batch SGD - Loss per epoch')

plt.show()

# Decision boundary plots
plot_decision_boundary(adaline1, X_train_std, y_train, "SGD Decision Boundary")
plot_decision_boundary(adaline2, X_train_std, y_train, "Mini-Batch SGD Decision Boundary")

# Print margin (w norm inverse)
sgd_margin = 1 / np.linalg.norm(adaline1.w_)
msgd_margin = 1 / np.linalg.norm(adaline2.w_)
print(f"SGD margin: {sgd_margin:.4f}")
print(f"Mini-Batch SGD margin: {msgd_margin:.4f}")
