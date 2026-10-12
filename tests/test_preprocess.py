import numpy as np
import scipy.sparse as sp
from src.data import load_dataset
from src.splits import load_split
from src.preprocess import build_prep

NAMES = ["amazon", "kick", "brazil_houses"]


def _has_nan(A):
    return np.isnan(A.data).any() if sp.issparse(A) else np.isnan(A).any()


def test_sklearn_preps():
    for name in NAMES:
        X, y = load_dataset(name)
        s = load_split(name)
        Xtr, Xva = X.iloc[s["train"]], X.iloc[s["val"]]
        for kind in ["linear", "tree"]:
            p = build_prep(kind, Xtr)
            A, B = p.transform(Xtr), p.transform(Xva)
            assert A.shape[0] == len(Xtr) and B.shape[0] == len(Xva)
            assert A.shape[1] == B.shape[1]
            assert not _has_nan(A) and not _has_nan(B)


def test_neural_prep():
    for name in NAMES:
        X, y = load_dataset(name)
        s = load_split(name)
        Xtr, Xva = X.iloc[s["train"]], X.iloc[s["val"]]
        p = build_prep("neural", Xtr, seed=0)
        xn, xc = p.transform(Xva)
        assert not np.isnan(xn).any()
        assert xc.min(initial=0) >= 0
        for j, card in enumerate(p.cardinalities):
          assert xc[:, j].max() < card