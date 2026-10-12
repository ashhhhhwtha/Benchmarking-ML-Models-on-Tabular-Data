import xgboost as xgb


class XGBModel:
    def __init__(self, params, task, seed, meta=None, cfg=None):
        self.task = task
        kw = dict(n_estimators=2000, learning_rate=params["learning_rate"],
                  max_depth=params["max_depth"], subsample=params["subsample"],
                  colsample_bytree=params["colsample_bytree"],
                  min_child_weight=params["min_child_weight"],
                  reg_alpha=params["reg_alpha"], reg_lambda=params["reg_lambda"],
                  tree_method="hist", early_stopping_rounds=50, n_jobs=8,
                  random_state=seed, verbosity=0)
        self.m = (xgb.XGBClassifier(eval_metric="auc", **kw) if task == "classification"
                  else xgb.XGBRegressor(eval_metric="rmse", **kw))

    def fit(self, X, y, Xva, yva):
        self.m.fit(X, y, eval_set=[(Xva, yva)], verbose=False)
        return self

    def predict(self, X):
        if self.task == "classification":
            return self.m.predict_proba(X)[:, 1]
        return self.m.predict(X)