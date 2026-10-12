import numpy as np
from src.data import load_yaml
from src.splits import load_split, get_train_subset

NAMES = list(load_yaml("configs/datasets.yaml")["datasets"])


def test_disjoint_and_complete():
    for n in NAMES:
        s = load_split(n)
        a, b, c = set(s["train"]), set(s["val"]), set(s["test"])
        assert not (a & b) and not (a & c) and not (b & c)
        assert len(a) + len(b) + len(c) == len(set.union(a, b, c))


def test_subsets_nested_and_from_train():
    for n in NAMES:
        tr = load_split(n)["train"]
        for seed in [0, 1, 2]:
            s10 = set(get_train_subset(tr, 0.1, seed))
            s25 = set(get_train_subset(tr, 0.25, seed))
            s50 = set(get_train_subset(tr, 0.5, seed))
            assert s10 <= s25 <= s50 <= set(tr)
        assert len(get_train_subset(tr, 1.0, 0)) == len(tr)