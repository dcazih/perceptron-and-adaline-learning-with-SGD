from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np

# Load the Iris dataset
iris = datasets.load_iris()

# Select only the petal length and petal width as features
X = iris.data[:, [2, 3]]

# Extract the target labels (species)
y = iris.target

print('Class labels:', np.unique(y)) # Iris-setosa, Iris-versicolor, and Iris-virginica

# Split iris data into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

# Standardize the features
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)