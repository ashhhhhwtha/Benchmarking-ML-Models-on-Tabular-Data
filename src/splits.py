import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from src.data import load_dataset, load_yaml


def make_split(name, force=False):
    glob = load_yaml("configs/global.yaml")
    ds = load_yaml("configs/datasets.yaml")["datasets"]
    path = Path(glob["paths"]["splits"]) / f"{name}.json"
    if path.exists() and not force:
        return json.loads(path.read_text())

    X, y = load_dataset(name, ds, glob)
    r = glob["split_ratios"]
    seed = glob["split_seed"]
    idx = np.arange(len(y))
    clf = ds[name]["task"] == "classification"

    train_idx, rest = train_test_split(
        idx, train_size=r["train"], random_state=seed,
        stratify=y if clf else None)
    val_idx, test_idx = train_test_split(
        rest, test_size=r["test"] / (r["val"] + r["test"]), random_state=seed,
        stratify=y.iloc[rest] if clf else None)

    split = {
        "n_rows": int(len(y)),
        "train": sorted(map(int, train_idx)),
        "val": sorted(map(int, val_idx)),
        "test": sorted(map(int, test_idx)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split))
    return split


def load_split(name):
    glob = load_yaml("configs/global.yaml")
    path = Path(glob["paths"]["splits"]) / f"{name}.json"
    s = json.loads(path.read_text())
    return {k: np.array(v) for k, v in s.items() if k != "n_rows"}


def get_train_subset(train_idx, fraction, seed):
    """Nested subsets: same seed gives the same permutation, so smaller fractions are prefixes."""
    perm = np.random.RandomState(seed).permutation(train_idx)
    n = int(round(fraction * len(train_idx)))
    return np.sort(perm[:n])


if __name__ == "__main__":
    ds = load_yaml("configs/datasets.yaml")["datasets"]
    for name in ds:
        s = make_split(name)
        print(f"{name:14s} rows={s['n_rows']:6d} train={len(s['train']):6d} "
              f"val={len(s['val']):6d} test={len(s['test']):6d}")