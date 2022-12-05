"""Forge dataset with splitting process.
"""

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from IntroToML.mglearn.mglearn import (
    datasets
)

# First, we split our data into a training and a test set
X, y = datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Import and instantiate class
clf = KNeighborsClassifier(n_neighbors=3)

# Fit classifier using the training set (storing the dataset to compute predictions)
clf.fit(X_train, y_train)

# Make predictions
print("Test set predictions: {}".format(clf.predict(X_test)))

# Print accuracy
print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))
