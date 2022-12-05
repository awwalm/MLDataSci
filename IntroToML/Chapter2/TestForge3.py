"""
Forge dataset with splitting process and decision boundary for all text data points in xy plane.
The following code produces the visualizations of the decision boundaries for one, three, and nine neighbors.
"""

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from IntroToML.mglearn.mglearn import (
    datasets,
    plots,
    discrete_scatter
)

# First, we split our data into a training and a test set
X, y = datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# The fit method returns the object self, so we can instantiate and fit in one line
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for n_neighbors, ax in zip([1, 3, 9], axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
