import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder,
                                   QuantileTransformer, StandardScaler)


def split_columns(X):
    cat = [c for c in X.columns if str(X[c].dtype) == "category"]
    num = [c for c in X.columns if c not in cat]
    return num, cat


def _obj(X):
    X = X.copy()
    for c in X.columns:
        if str(X[c].dtype) == "category":
            X[c] = X[c].astype(object)
    return X


class SkPrep:
    """Linear and tree preprocessing. kind: 'linear' or 'tree'."""

    def __init__(self, kind):
        assert kind in ("linear", "tree")
        self.kind = kind

    def fit(self, X):
        num, cat = split_columns(X)
        cat_imp = SimpleImputer(strategy="constant", fill_value="missing")
        if self.kind == "linear":
            num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                                 ("sc", StandardScaler())])
            cat_pipe = Pipeline([
                ("imp", cat_imp),
                ("enc", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                      min_frequency=10, sparse_output=True))])
            sparse_threshold = 0.3
        else:
            num_pipe = SimpleImputer(strategy="median")
            cat_pipe = Pipeline([
                ("imp", cat_imp),
                ("enc", OrdinalEncoder(handle_unknown="use_encoded_value",
                                       unknown_value=-1))])
            sparse_threshold = 0.0
        parts = []
        if num:
            parts.append(("num", num_pipe, num))
        if cat:
            parts.append(("cat", cat_pipe, cat))
        self.ct = ColumnTransformer(parts, sparse_threshold=sparse_threshold)
        self.ct.fit(_obj(X))
        return self

    def transform(self, X):
        return self.ct.transform(_obj(X))


class NeuralPrep:
    """Returns (X_num float32, X_cat int64). Category index 0 = unseen/unknown."""

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X):
        self.num, self.cat = split_columns(X)
        X = _obj(X)
        if self.num:
            self.imp = SimpleImputer(strategy="median").fit(X[self.num])
            self.qt = QuantileTransformer(
                n_quantiles=min(1000, len(X)), output_distribution="normal",
                random_state=self.seed).fit(self.imp.transform(X[self.num]))
        self.cat_imp = None
        self.cardinalities = []
        if self.cat:
            self.cat_imp = SimpleImputer(strategy="constant", fill_value="missing")
            Z = self.cat_imp.fit_transform(X[self.cat])
            self.enc = OrdinalEncoder(handle_unknown="use_encoded_value",
                                      unknown_value=-1).fit(Z)
            self.cardinalities = [len(c) + 1 for c in self.enc.categories_]
        return self

    def transform(self, X):
        X = _obj(X)
        n = len(X)
        if self.num:
            Xn = self.qt.transform(self.imp.transform(X[self.num])).astype(np.float32)
        else:
            Xn = np.zeros((n, 0), dtype=np.float32)
        if self.cat:
            Z = self.cat_imp.transform(X[self.cat])
            Xc = (self.enc.transform(Z) + 1).astype(np.int64)
        else:
            Xc = np.zeros((n, 0), dtype=np.int64)
        return Xn, Xc


def build_prep(kind, X_train, seed=0):
    if kind in ("linear", "tree"):
        return SkPrep(kind).fit(X_train)
    if kind == "neural":
        return NeuralPrep(seed).fit(X_train)
    raise ValueError(kind)


def encode_target(y, task):
    return np.log1p(y) if task == "regression" else y


def decode_pred(p, task):
    return np.expm1(p) if task == "regression" else p