"""
The following code creates a scatter plot visualizing all the data points in the Wave dataset.
The wave dataset has one input feature and a continuous target variable (or response) that we want to model.
The plot created here shows the feature on the x-axis and the regression target (the output) on the y-axis.
"""

import matplotlib.pyplot as plt
from IntroToML.mglearn.mglearn import (
    datasets
)

# Generate dataset
X, y = datasets.make_wave(n_samples=40)

# Plot dataset
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Target")
print("X.shape: {}".format(X.shape))
plt.show()
