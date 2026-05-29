# filename: src/xgb_wrapper.py
# purpose:  sklearn-compatible XGBoost wrapper handling Cover_Type 1-7 label offset
# version:  1.0

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier


class XGBWrapper(BaseEstimator, ClassifierMixin):
    """
    Wraps XGBClassifier to handle Cover_Type labels 1-7.
    XGBoost expects 0-indexed labels (0-6).
    This class subtracts 1 on fit() and adds 1 back on predict().
    All other code (predictor.py, tests, API) uses standard sklearn API.
    """

    def __init__(self, **xgb_params):
        self.xgb_params = xgb_params
        self.model = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)  # [1, 2, 3, 4, 5, 6, 7]
        self.model = XGBClassifier(**self.xgb_params)
        self.model.fit(X, np.array(y) - 1)  # shift to 0-indexed
        return self

    def predict(self, X):
        return self.model.predict(X) + 1  # shift back to 1-7

    def predict_proba(self, X):
        # probabilities: index 0 = class 1, index 1 = class 2, etc.
        return self.model.predict_proba(X)

    def get_params(self, deep: bool = True) -> dict:
        return {**self.xgb_params}

    def set_params(self, **params):
        self.xgb_params.update(params)
        return self
