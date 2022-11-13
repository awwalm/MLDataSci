"""
Boston Housing dataset. The task associated with this dataset is to predict the median value of homes
in several Boston neighborhoods in the 1970s, using information such as crime rate, proximity to the Charles River,
highway accessibility, and so on. The dataset contains 506 data points, described by 13 features.
"""

from sklearn.datasets import load_boston
from IntroToML.mglearn.mglearn import (
    datasets
)

boston = load_boston()
print("Data shape:{}".format(boston.data.shape))
X, y = datasets.load_extended_boston()
print("X.shape: {}".format(X.shape))
