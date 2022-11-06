"""
The following code creates a scatter plot visualizing all the data points in the Forge dataset.
"""

import matplotlib.pyplot as plt
from IntroToML.mglearn.mglearn import (
    datasets,
    discrete_scatter
)

# Generate dataset
X, y = datasets.make_forge()

# Plot dataset
discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape: {}".format(X.shape))
plt.show()
