from sklearn.linear_model import LogisticRegression, Ridge


class LinearModel:
    def __init__(self, params, task, seed, meta=None, cfg=None):
        self.task = task
        if task == "classification":
            self.m = LogisticRegression(C=params["C"], max_iter=2000, random_state=seed)
        else:
            self.m = Ridge(alpha=params["alpha"], random_state=seed)

    def fit(self, X, y, Xva=None, yva=None):
        self.m.fit(X, y)
        return self

    def predict(self, X):
        if self.task == "classification":
            return self.m.predict_proba(X)[:, 1]
        return self.m.predict(X)