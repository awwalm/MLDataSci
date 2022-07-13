# A simple program to demonsratea two-class classification

import matplotlib.pyplot as plt
from DataScience.mglearn import mglearn
from DataScience.mglearn.mglearn import datasets

X, y = datasets.make_forge()
plt.scatter(X[:, 0], X[:, 1], c=y, s=60, cmap=mglearn.cm2)
print("X.shape: %s" % (X.shape,))
