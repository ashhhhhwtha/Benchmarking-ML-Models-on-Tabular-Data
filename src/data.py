from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

CACHE = Path("data_cache")


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _is_cat(s: pd.Series) -> bool:
    return str(s.dtype) in ("object", "category", "bool", "string")


def load_dataset(name, ds_cfg=None, glob_cfg=None):
    ds_cfg = ds_cfg or load_yaml("configs/datasets.yaml")["datasets"]
    glob_cfg = glob_cfg or load_yaml("configs/global.yaml")
    cfg = ds_cfg[name]

    kwargs = dict(data_id=cfg["openml_id"], as_frame=True, return_X_y=True,
                  data_home=str(CACHE), parser="auto")
    if cfg.get("target"):
        kwargs["target_column"] = cfg["target"]
    X, y = fetch_openml(**kwargs)
    X = X.copy()
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)
    drop = cfg.get("drop_columns") or []
    absent = [c for c in drop if c not in X.columns]
    if absent:
        raise KeyError(f"drop_columns not found in {name}: {absent}")
    X = X.drop(columns=drop)

    # Drop rows with missing target
    keep = y.notna().to_numpy()
    X, y = X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)

    # Dtypes: categoricals become string categories (NaN preserved); the rest numeric
    for c in X.columns:
        if _is_cat(X[c]):
            col = X[c].astype("object")
            X[c] = col.where(col.isna(), col.astype(str)).astype("category")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # Drop constant columns
    const = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    X = X.drop(columns=const)

    # Target
    if cfg["task"] == "classification":
        y = pd.Series(LabelEncoder().fit_transform(y.astype(str)), name="target")
    else:
        y = pd.to_numeric(y).astype(float).rename("target")

    # Optional fixed subsample (applies to every model on this dataset)
    max_rows = cfg.get("max_rows")
    if max_rows and len(X) > max_rows:
        rng = np.random.RandomState(glob_cfg["split_seed"])
        if cfg["task"] == "classification":
            idx = (pd.DataFrame({"i": np.arange(len(y)), "y": y})
                   .groupby("y", group_keys=False)
                   .apply(lambda g: g.sample(frac=max_rows / len(y), random_state=rng))["i"]
                   .to_numpy())
        else:
            idx = rng.choice(len(X), size=max_rows, replace=False)
        idx = np.sort(idx)
        X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)

    return X, y


def dataset_metadata(name, X, y, task):
    cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    row = {
        "dataset": name,
        "task": task,
        "rows": len(X),
        "features": X.shape[1],
        "numeric": len(num_cols),
        "categorical": len(cat_cols),
        "cat_share": round(len(cat_cols) / X.shape[1], 3),
        "max_cardinality": max([X[c].nunique() for c in cat_cols], default=0),
        "missing_share": round(float(X.isna().mean().mean()), 4),
    }
    if task == "classification":
        vc = y.value_counts(normalize=True)
        row["n_classes"] = int(y.nunique())
        row["majority_class_share"] = round(float(vc.iloc[0]), 3)
    return row


if __name__ == "__main__":
    ds_cfg = load_yaml("configs/datasets.yaml")["datasets"]
    glob_cfg = load_yaml("configs/global.yaml")
    rows = []
    for name, cfg in ds_cfg.items():
        print(f"loading {name} (id {cfg['openml_id']}) ...", flush=True)
        X, y = load_dataset(name, ds_cfg, glob_cfg)
        print(f"  shape={X.shape}  target_dtype={y.dtype}  first_cols={list(X.columns[:5])}")
        rows.append(dataset_metadata(name, X, y, cfg["task"]))
    meta = pd.DataFrame(rows)
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    meta.to_csv("results/tables/dataset_metadata.csv", index=False)
    print(meta.to_string(index=False))