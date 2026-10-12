import time
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from src.preprocess import decode_pred

# Bounds on the log1p scale for regression predictions. Applied to every model.
# exp(16) is about 8.9M, above any target in these datasets.
LOG_LO, LOG_HI = 0.0, 16.0


def get_space(spaces, model, task):
    s = spaces[model]
    return s.get(task, s)


def sample_params(space, rng):
    out = {}
    for k, s in space.items():
        t = s["type"]
        if t == "loguniform":
            out[k] = float(np.exp(rng.uniform(np.log(s["low"]), np.log(s["high"]))))
        elif t == "uniform":
            out[k] = float(rng.uniform(s["low"], s["high"]))
        elif t == "int":
            out[k] = int(rng.randint(s["low"], s["high"] + 1))
        elif t == "choice":
            out[k] = s["values"][rng.randint(len(s["values"]))]
        else:
            raise ValueError(f"unknown type {t} for {k}")
    return out


def metrics(pred, y_raw, task, bounds=None):
    pred = np.asarray(pred, dtype=float)
    if not np.isfinite(pred).all():
        raise ValueError("non-finite predictions")
    if task == "classification":
        return {"auc": float(roc_auc_score(y_raw, pred)),
                "accuracy": float(accuracy_score(y_raw, pred >= 0.5))}
    lo, hi = bounds if bounds is not None else (LOG_LO, LOG_HI)
    p_log = np.clip(pred, lo, hi)
    p = decode_pred(p_log, task)
    return {"rmse": float(np.sqrt(np.mean((p - y_raw) ** 2))),
            "rmse_log": float(np.sqrt(np.mean((p_log - np.log1p(y_raw)) ** 2)))}


def select_value(m, task):
    """Higher is better."""
    return m["auc"] if task == "classification" else -m["rmse"]


def random_search(task, space, seed, n_trials, make_fn, Xtr, ytr, Xva, yva, yva_raw, bounds=None):
    rng = np.random.RandomState(seed)
    best, log = None, []
    t_start = time.time()
    for i in range(n_trials):
        params = sample_params(space, rng)  # drawn before the try block on purpose
        t0 = time.time()
        m, met, val, err, fit_t = None, None, -np.inf, None, None
        try:
            m = make_fn(params).fit(Xtr, ytr, Xva, yva)
            fit_t = time.time() - t0
            met = metrics(m.predict(Xva), yva_raw, task, bounds)
            val = select_value(met, task)
        except Exception as e:
            m, err = None, repr(e)[:200]
        log.append({"trial": i, "params": params, "val": met, "error": err,
                    "time_s": round(time.time() - t0, 2),
                    "timed_out": bool(getattr(m, "timed_out", False)),
                    "epochs": getattr(m, "epochs_run", None)})
        if m is not None and (best is None or val > best["val"]):
            best = {"val": val, "metrics": met, "params": params, "model": m,
                    "fit_time_s": fit_t, "trial": i}
    if best is None:
        raise RuntimeError("all trials failed; first error: "
                           + str(next(l["error"] for l in log if l["error"])))
    return best, log, time.time() - t_start