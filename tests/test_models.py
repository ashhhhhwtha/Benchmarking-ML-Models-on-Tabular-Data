import time
from functools import lru_cache
import numpy as np
import pytest
from src.data import load_dataset, load_yaml
from src.splits import load_split
from src.preprocess import build_prep, encode_target
from src.models import REGISTRY, make_model
from src.tune import get_space, sample_params

MODELS = list(REGISTRY)
DATASETS = ["amazon", "kick", "brazil_houses", "california"]


@lru_cache(maxsize=None)
def _data(name):
    return load_dataset(name), load_split(name)


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("model", MODELS)
def test_model_fits_and_predicts(model, dataset, n=800):
    ds_cfg = load_yaml("configs/datasets.yaml")["datasets"][dataset]
    task = ds_cfg["task"]
    cfg = load_yaml("configs/global.yaml")
    cfg["neural"]["max_epochs"] = 2
    spaces = load_yaml("configs/search_spaces.yaml")
    (X, y), s = _data(dataset)
    tr, va = s["train"][:n], s["val"][:n]
    prep = build_prep(REGISTRY[model][1], X.iloc[tr], seed=0)
    Xtr, Xva = prep.transform(X.iloc[tr]), prep.transform(X.iloc[va])
    ytr = encode_target(y.iloc[tr].to_numpy(), task)
    yva = encode_target(y.iloc[va].to_numpy(), task)
    meta = ({"n_num": len(prep.num), "cardinalities": prep.cardinalities}
            if REGISTRY[model][1] == "neural" else {})
    params = sample_params(get_space(spaces, model, task), np.random.RandomState(0))
    t0 = time.time()
    m = make_model(model, params, task, 0, meta, cfg).fit(Xtr, ytr, Xva, yva)
    pred = np.asarray(m.predict(Xva))
    print(f"{model:14s} {dataset:14s} {time.time() - t0:6.1f}s")
    assert pred.shape == (len(va),) and np.isfinite(pred).all()
    if task == "classification":
        assert pred.min() >= 0 and pred.max() <= 1