import os
import sys
import warnings
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.multioutput import (ClassifierChain, MultiOutputClassifier,
                                 _available_if_estimator_has)
from sklearn.svm import LinearSVC
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import (_check_fit_params, check_is_fitted,
                                      has_fit_parameter)


def _fit_estimator(estimator, X, y, sample_weight=None, **fit_params):
    estimator = clone(estimator)
    if sample_weight is not None:
        estimator.fit(X, y, sample_weight=sample_weight, **fit_params)
    else:
        estimator.fit(X, y, **fit_params)
    return estimator


class LinearSVCChainEnsemble:
    def __init__(self, num_chains, scoring):
        self.num_chains = num_chains
        self.scoring = scoring

    def fit(self, X_tr, y_tr, X_val, y_val, C_min=-1, C_max=1):
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"
        test_split_index = [-1] * len(y_tr) + [0] * len(y_val)
        X_tr_val, y_tr_val = np.concatenate((X_tr, X_val), axis=0), np.concatenate(
            (y_tr, y_val)
        )
        splits = PredefinedSplit(test_fold=test_split_index)

        def get_chain(i):
            print("Getting chain", i)
            estimator = ClassifierChain(LinearSVC(), order="random", random_state=i)
            parameters = {
                "base_estimator__C": np.logspace(
                    C_min, C_max, (C_max - C_min + 1) * 2 - 1
                )
            }

            clf = GridSearchCV(
                estimator,
                parameters,
                scoring=self.scoring,
                cv=splits,
                refit=False,
                n_jobs=-1,
            )

            clf.fit(X_tr_val, y_tr_val)

            # print(pd.DataFrame.from_dict(clf.cv_results_).sort_values('rank_test_score'))
            print(clf.best_params_)
            estimator.set_params(**clf.best_params_)
            clf = estimator
            clf.fit(X_tr, y_tr)

            return clf

        chains = Parallel(n_jobs=-1)(
            delayed(get_chain)(i) for i in range(self.num_chains)
        )
        self.estimators = chains

        return self

    def decision_function(self, X):
        y = Parallel(n_jobs=-1)(
            delayed(e.decision_function)(X) for e in self.estimators
        )
        y = np.array(y).mean(axis=0)
        return y


class MyMultiOutputClassifier(MultiOutputClassifier):
    def fit(self, X, y, sample_weight=None, X_val=None, y_val=None, **fit_params):
        self._validate_params()

        if not hasattr(self.estimator, "fit"):
            raise ValueError("The base estimator should implement a fit method")

        y = self._validate_data(X="no_validation", y=y, multi_output=True)

        if is_classifier(self):
            check_classification_targets(y)

        if y.ndim == 1:
            raise ValueError(
                "y must have at least two dimensions for "
                "multi-output regression but has only one."
            )

        if sample_weight is not None and not has_fit_parameter(
            self.estimator, "sample_weight"
        ):
            raise ValueError("Underlying estimator does not support sample weights.")

        fit_params_validated = _check_fit_params(X, fit_params)

        if y_val is None:
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_estimator)(
                    self.estimator, X, y[:, i], sample_weight, **fit_params_validated
                )
                for i in range(y.shape[1])
            )
        else:
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_estimator)(
                    self.estimator,
                    X,
                    y[:, i],
                    sample_weight,
                    X_val=X_val,
                    y_val=y_val[:, i],
                    **fit_params_validated,
                )
                for i in range(y.shape[1])
            )

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_

        return self

    @_available_if_estimator_has("decision_function")
    def decision_function(self, X):
        check_is_fitted(self)
        y = Parallel(n_jobs=self.n_jobs)(
            delayed(e.decision_function)(X) for e in self.estimators_
        )
        return np.asarray(y).T

    @_available_if_estimator_has("predict_proba")
    def predict_proba(self, X):
        check_is_fitted(self)
        y = Parallel(n_jobs=self.n_jobs)(
            delayed(e.predict_proba)(X) for e in self.estimators_
        )
        if y[0].ndim > 1:
            y = [pred[:, 1] for pred in y]
        return np.asarray(y).T

    @property
    def best_config_train_time(self):
        check_is_fitted(self)
        return np.mean([e.best_config_train_time for e in self.estimators_])

    @property
    def best_config(self):
        check_is_fitted(self)
        return [e.best_config for e in self.estimators_]


class SklearnPredictor(BaseEstimator):
    def __init__(self, task_type="multi_class", **params):
        if task_type == "multi_class":
            self.clf = LinearSVC(**params)
        elif task_type == "binary":
            self.clf = LinearSVC(**params)
        elif task_type == "regression":
            self.clf = Ridge(**params)
        elif task_type == "multi_label":
            clf = LinearSVC(**params)
            self.clf = MyMultiOutputClassifier(clf, n_jobs=-1)

        self.task_type = task_type

    def fit(self, X, y, **params):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def decision_function(self, X):
        return self.clf.decision_function(X)

    def set_params(self, **params):
        self.clf.set_params(**params)
        return self

    # def get_params(self, deep=True):
    #     return self.clf.get_params(deep=deep)

    def get_grid(self):
        grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
        if self.task_type == "regression":
            grid = {"alpha": grid}
        else:
            grid = {"C": grid}

        if self.task_type == "multi_label":
            grid = {f"estimator__{key}": value for key, value in grid.items()}
        return grid
