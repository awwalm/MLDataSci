"""A First Application: Classifying Iris Species."""

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
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

# input("press any key to close")
