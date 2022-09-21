"""A First Application: Classifying Iris Species."""

from sklearn.datasets import load_iris

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
