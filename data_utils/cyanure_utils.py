from collections import defaultdict

import numpy as np
from cyanure import BinaryClassifier, MultiClassifier, Regression
from sklearn.base import BaseEstimator
from sklearn.multioutput import MultiOutputClassifier


class SklearnBinaryClassifier(BinaryClassifier, BaseEstimator):
    def __init__(self, **params):
        super().__init__(**params)
        self.classes_ = 2


class CyanurePredictor(BaseEstimator):
    def __init__(self, task_type="multi_class", C=1.0, **params):
        if task_type == "multi_class":
            self.clf = MultiClassifier(loss="sqhinge", **params)
        elif task_type == "binary":
            self.clf = BinaryClassifier(loss="sqhinge", **params)
        elif task_type == "regression":
            self.clf = Regression(**params)
        elif task_type == "multi_label":
            clf = SklearnBinaryClassifier(loss="sqhinge", **params)
            self.clf = MultiOutputClassifier(clf, n_jobs=-1)

        self.task_type = task_type
        self.C = C
        self.label_map = None

    def fit(self, X, y, **params):
        lambd = 1 / (2 * len(y) * self.C)
        if self.task_type == "multi_class":
            self.label_map, y = np.unique(y, return_inverse=True)
        self.clf.fit(X, y, lambd=lambd, lambd2=lambd, verbose=False)

    def predict(self, X):
        if self.task_type == "multi_class" and self.label_map is not None:
            return self.label_map[self.clf.predict(X)]
        return self.clf.predict(X)
