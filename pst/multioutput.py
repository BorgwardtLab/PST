from sklearn.base import clone
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils import Bunch
from sklearn.utils.metadata_routing import _routing_enabled, process_routing
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import _check_fit_params, has_fit_parameter


def _fit_estimator(estimator, X, y, sample_weight=None, **fit_params):
    estimator = clone(estimator)
    if sample_weight is not None:
        estimator.fit(X, y, sample_weight=sample_weight, **fit_params)
    else:
        estimator.fit(X, y, **fit_params)
    return estimator


class MultiOutputClassifierWithVal(MultiOutputClassifier):
    def __init__(self, estimator, *, n_jobs=None):
        super().__init__(estimator, n_jobs=n_jobs)

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None, **fit_params):
        """Fit the model to data, separately for each output variable.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets. An indicator matrix turns on multilabel
            estimation.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If `None`, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.

        **fit_params : dict of string -> object
            Parameters passed to the ``estimator.fit`` method of each step.

            .. versionadded:: 0.23

        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        if _routing_enabled():
            routed_params = process_routing(
                obj=self,
                method="fit",
                other_params=fit_params,
                sample_weight=sample_weight,
            )
        else:
            if sample_weight is not None and not has_fit_parameter(
                self.estimator, "sample_weight"
            ):
                raise ValueError(
                    "Underlying estimator does not support sample weights."
                )

            fit_params_validated = _check_fit_params(X, fit_params)
            routed_params = Bunch(estimator=Bunch(fit=fit_params_validated))
            if sample_weight is not None:
                routed_params.estimator.fit["sample_weight"] = sample_weight

        if y_val is None:
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_estimator)(
                    self.estimator, X, y[:, i], **routed_params.estimator.fit
                )
                for i in range(y.shape[1])
            )
        else:
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_estimator)(
                    self.estimator,
                    X,
                    y[:, i],
                    X_val=X_val,
                    y_val=y_val[:, i],
                    **routed_params.estimator.fit
                )
                for i in range(y.shape[1])
            )

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_
        self.classes_ = [estimator.classes_ for estimator in self.estimators_]
        return self
