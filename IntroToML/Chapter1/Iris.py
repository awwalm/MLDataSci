"""
A First Application: Classifying Iris Species.
This implements the KNN classifier for predicting the type of flower in the test set.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Iris dataset.
# load_iris() returns a `Bunch` object, which is similar to a dictionary
iris = load_iris()

# iris.keys() -> dict_keys(['DESCR', 'data', 'target_names', 'feature_names', 'target'])
print(iris.keys())

# DESCR holds a value containing a short description of the dataset
print(iris['DESCR'][:193] + "\n...")

# target_names is an array of strings containing the species of flower to be predicted
print(iris["target_names"])

# feature_names are a list of strings giving the description of each feature
print(iris['feature_names'])

# data contains the numeric measurements of sepal length, sepal width, petal length,
# and petal width in a NumPy array (rows -> flowers|columns-> four measurements)
print(type(iris['data']))
print(iris['data'].shape)

# First five samples
print(iris['data'][:5])

# target contains species of each of the flowers that were measured
# It is 1-D numpy array with one entry per flower, encoded as integers between 0 and 2
# (0 means Setosa, 1 means Versicolor and 2 means Virginica)
print(iris['target'])

# Split dataset into training and test set
# random_state parameter value 0f 0 allows the shuffling of dataset to be the same
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)

# Training (75%) and test set (25%)
print(X_train.shape)
print(X_test.shape)

# Preliminary visualization - pair plot which takes all pairs of two features
fig, ax = plt.subplots(3, 3, figsize=(15, 15))
plt.suptitle("iris_pairplot")
for i in range(3):
    for j in range(3):
        ax[i, j].scatter(X_train[:, j], X_train[:, i+1], c=y_train, s=60)
        ax[i, j].set_xticks(())
        ax[i, j].set_yticks(())
        if i == 2:
            ax[i, j].set_xlabel(iris['feature_names'][j])
        if j == 0:
            ax[i, j].set_ylabel(iris['feature_names'][i+1])
        if j > i:
            ax[i, j].set_visible(False)
plt.show()

# KNN model with neighbors set to 1 for now
# The `knn` object is knowledgeable of the training set
knn = KNeighborsClassifier(n_neighbors=1)

# To build a model on the training set, we call `fit` method of the `knn` object
knn.fit(X_train, y_train)
KNeighborsClassifier(
    algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None,
    n_jobs=1, n_neighbors=1, p=2, weights='uniform'
)

# It is now time to predict: Imagine we found an iris in the wild with a sepal length of 5cm,
# a sepal width of 2.9cm, a petal length of 1cm and a petal width of 0.2cm. What species of iris would this be?
X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new)
print(prediction)

# Model evaluation
y_pred = knn.predict(X_test)
print(y_pred, np.mean(y_pred == y_test), knn.score(X_test, y_test))

# input("press any key to close")
