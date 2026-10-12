from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class RFModel:
    def __init__(self, params, task, seed, meta=None, cfg=None):
        self.task = task
        kw = dict(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                  min_samples_leaf=params["min_samples_leaf"],
                  max_features=params["max_features"], n_jobs=-1, random_state=seed)
        self.m = (RandomForestClassifier(**kw) if task == "classification"
                  else RandomForestRegressor(**kw))

    def fit(self, X, y, Xva=None, yva=None):
        self.m.fit(X, y)
        return self

    def predict(self, X):
        if self.task == "classification":
            return self.m.predict_proba(X)[:, 1]
        return self.m.predict(X)