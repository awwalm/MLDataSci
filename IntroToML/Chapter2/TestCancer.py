"""
Wisconsin Breast Cancer dataset, which records clinical measurements of breast cancer tumors.
Each tumor is labeled as “benign” (for harmless tumors) or “malignant” (for cancerous tumors),
and the task is to learn to predict whether a tumor is malignant based on the measurements of the tissue.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
print(f"cancer.keys():\t\n{cancer.keys()}")
print(f"Shape of cancer data: {cancer.data.shape}")
print("Sample counts per class:\n{}".format({n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
print("Feature names:\n{}".format(cancer.feature_names))
print(cancer.DESCR)
