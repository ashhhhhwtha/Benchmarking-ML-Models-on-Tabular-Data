import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path
import numpy as np
from src.data import load_dataset, load_yaml
from src.env_info import get_env_info
from src.models import REGISTRY, make_model
from src.preprocess import build_prep, encode_target
from src.seeding import set_global_seed
from src.splits import get_train_subset, load_split
from src.tune import get_space, metrics, random_search


def config_hash():
    h = hashlib.sha256()
    for p in ["configs/global.yaml", "configs/search_spaces.yaml", "configs/datasets.yaml"]:
        h.update(Path(p).read_text().replace("\r\n", "\n").encode())
    return h.hexdigest()[:12]


def out_path(out_dir, dataset, model, fraction, seed):
    return Path(out_dir) / f"{dataset}__{model}__f{fraction}__s{seed}.json"


def run_job(dataset, model, fraction, seed, out_dir, n_trials=None, force=False):
    """Returns the result dict, or None if the job was already done."""
    glob = load_yaml("configs/global.yaml")
    ds_cfg = load_yaml("configs/datasets.yaml")["datasets"]
    spaces = load_yaml("configs/search_spaces.yaml")
    task = ds_cfg[dataset]["task"]
    path = out_path(out_dir, dataset, model, fraction, seed)
    if path.exists() and not force:
        return None

    set_global_seed(seed)
    t_job = time.time()
    X, y = load_dataset(dataset, ds_cfg, glob)
    y_raw = y.to_numpy()
    s = load_split(dataset)
    tr = get_train_subset(s["train"], fraction, seed)
    va, te = s["val"], s["test"]

    kind = REGISTRY[model][1]
    prep = build_prep(kind, X.iloc[tr], seed=seed)  # fitted on the training subset only
    Xtr, Xva, Xte = (prep.transform(X.iloc[i]) for i in (tr, va, te))
    ytr = encode_target(y_raw[tr], task)
    bounds = (float(ytr.min()), float(ytr.max())) if task == "regression" else None
    yva = encode_target(y_raw[va], task)
    meta = ({"n_num": len(prep.num), "cardinalities": prep.cardinalities}
            if kind == "neural" else {})

    n_trials = n_trials or glob["n_trials"]
    space = get_space(spaces, model, task)
    best, log, tune_time = random_search(
        task, space, seed, n_trials,
        lambda p: make_model(model, p, task, seed, meta, glob),
        Xtr, ytr, Xva, yva, y_raw[va], bounds)

    test = metrics(best["model"].predict(Xte), y_raw[te], task, bounds)  # test set used once, here
    res = {
        "dataset": dataset, "model": model, "fraction": fraction, "seed": seed,
        "task": task, "n_train": int(len(tr)), "n_trials": n_trials,
        "n_trials_failed": sum(1 for l in log if l["error"]),
        "n_trials_timed_out": sum(1 for l in log if l["timed_out"]),
        "best_trial": best["trial"],
        "val_metrics": best["metrics"],
        "test_roc_auc": test.get("auc"), "test_accuracy": test.get("accuracy"),
        "test_rmse": test.get("rmse"),
        "test_rmse_log": test.get("rmse_log"),
        "best_params": best["params"],
        "tune_time_s": round(tune_time, 2),
        "fit_time_s": round(best["fit_time_s"], 2),
        "job_time_s": round(time.time() - t_job, 2),
        "epochs_run": getattr(best["model"], "epochs_run", None),
        "device": "cpu", "env": get_env_info(), "config_hash": config_hash(),
        "trials": log,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, default=str))
    tmp.replace(path)
    return res


def main():
    glob = load_yaml("configs/global.yaml")
    ds_all = list(load_yaml("configs/datasets.yaml")["datasets"])
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=ds_all)
    ap.add_argument("--models", nargs="*", default=list(REGISTRY))
    ap.add_argument("--fractions", nargs="*", type=float, default=glob["fractions"])
    ap.add_argument("--seeds", nargs="*", type=int, default=glob["seeds"])
    ap.add_argument("--smoke", action="store_true", help="2 trials, writes to results/smoke")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    out_dir = "results/smoke" if a.smoke else glob["paths"]["raw_results"]
    n_trials = 2 if a.smoke else None
    failed = []
    for model in a.models:
        for dataset in a.datasets:
            for fr in a.fractions:
                for seed in a.seeds:
                    tag = f"{dataset:14s} {model:14s} f{fr:<5} s{seed}"
                    try:
                        r = run_job(dataset, model, fr, seed, out_dir, n_trials, a.force)
                    except Exception:
                        failed.append(tag)
                        print("FAILED ", tag)
                        traceback.print_exc()
                        continue
                    if r is None:
                        print("skip   ", tag)
                        continue
                    score = (f"auc={r['test_roc_auc']:.4f}" if r["task"] == "classification"
                             else f"rmse={r['test_rmse']:.2f}")
                    print(f"done   {tag} {score}  tune={r['tune_time_s']}s  "
                          f"failed_trials={r['n_trials_failed']}  "
                          f"timed_out={r['n_trials_timed_out']}", flush=True)
    if failed:
        print(f"\n{len(failed)} jobs failed:")
        for f in failed:
            print("  ", f)


if __name__ == "__main__":
    main()