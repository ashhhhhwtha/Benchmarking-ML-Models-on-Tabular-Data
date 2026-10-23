import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data import dataset_metadata, load_dataset, load_yaml

RAW, TAB, FIG = Path("results/raw"), Path("results/tables"), Path("results/figures")
MODELS = ["linear", "rf", "xgb", "mlp", "fttransformer"]
TREES, NEURAL = ["rf", "xgb"], ["mlp", "fttransformer"]
COLORS = {"linear": "#7f7f7f", "rf": "#2ca02c", "xgb": "#1f77b4",
          "mlp": "#ff7f0e", "fttransformer": "#d62728"}


def load_results():
    rows = []
    for f in glob.glob(str(RAW / "*.json")):
        r = json.load(open(f))
        clf = r["task"] == "classification"
        rows.append({
            "dataset": r["dataset"], "model": r["model"], "fraction": r["fraction"],
            "seed": r["seed"], "task": r["task"], "n_train": r["n_train"],
            "score": r["test_roc_auc"] if clf else r["test_rmse"],
            "accuracy": r["test_accuracy"], "rmse_log": r["test_rmse_log"],
            "fit_time_s": r["fit_time_s"],
            "median_trial_s": float(np.median([t["time_s"] for t in r["trials"]])),
            "timed_out": r["n_trials_timed_out"],
        })
    return pd.DataFrame(rows)


def fmt(m, s, task):
    s = 0.0 if pd.isna(s) else s
    return f"{m:.4f} ± {s:.4f}" if task == "classification" else f"{m:,.0f} ± {s:,.0f}"


def to_md(df):
    d = df.reset_index()
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(lines)


def better(a, b, higher):
    return a > b if higher else a < b


def verdict(sub, higher):
    """sub: rows with model, mean, std. A win needs a gap larger than the pooled std."""
    sub = sub.assign(std=sub["std"].fillna(0.0)).sort_values("mean", ascending=not higher)
    b = sub.iloc[0]
    tied = [r.model for r in sub.itertuples()
            if abs(r.mean - b["mean"]) <= np.sqrt((r.std ** 2 + b["std"] ** 2) / 2)]
    gap, pooled = np.nan, np.nan
    if len(sub) > 1:
        s2 = sub.iloc[1]
        gap = abs(b["mean"] - s2["mean"])
        pooled = np.sqrt((b["std"] ** 2 + s2["std"] ** 2) / 2)
    label = b["model"] if len(tied) == 1 else "tie: " + ", ".join(tied)
    return label, tied, gap, pooled


def tree_vs_neural(sub, higher):
    t = sub[sub.model.isin(TREES)].sort_values("mean", ascending=not higher).iloc[0]
    n = sub[sub.model.isin(NEURAL)].sort_values("mean", ascending=not higher).iloc[0]
    gap = abs(t["mean"] - n["mean"])
    pooled = np.sqrt((np.nan_to_num(t["std"]) ** 2 + np.nan_to_num(n["std"]) ** 2) / 2)
    if gap <= pooled:
        out = "tie"
    else:
        out = "trees" if better(t["mean"], n["mean"], higher) else "neural"
    return {"best_tree": t["model"], "best_neural": n["model"], "gap": gap,
            "pooled_std": pooled, "gap_in_std": gap / pooled if pooled > 0 else np.inf,
            "outcome": out}


def data_needed(summary, ds, model, task, adjusted=False):
    s = summary[(summary.dataset == ds) & (summary.model == model)].sort_values("fraction")
    if len(s) < 2:
        return "100% only"
    full = s[s.fraction == 1.0]["mean"].iloc[0]
    for r in s.itertuples():
        if task == "classification":
            ratio = (r.mean - 0.5) / (full - 0.5) if adjusted else r.mean / full
        else:
            if adjusted:
                return "n/a"
            ratio = full / r.mean
        if ratio >= 0.95:
            return f"{int(r.fraction * 100)}%"
    return "not reached"


def plot_curves(summary, datasets, tasks):
    FIG.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        fig, ax = plt.subplots(figsize=(6, 4))
        for m in MODELS:
            s = summary[(summary.dataset == ds) & (summary.model == m)].sort_values("fraction")
            if s.empty:
                continue
            ax.errorbar(s.fraction * 100, s["mean"], yerr=s["std"].fillna(0),
                        marker="o", capsize=3, label=m, color=COLORS[m])
        ax.set_xlabel("Training data (%)")
        ax.set_ylabel("Test ROC-AUC" if tasks[ds] == "classification" else "Test RMSE")
        ax.set_xticks([10, 25, 50, 100])
        ax.set_title(ds)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / f"curve_{ds}.png", dpi=150)
        plt.close(fig)


def plot_time_vs_score(summary, df, tasks):
    full = summary[summary.fraction == 1.0].copy()
    t = df[df.fraction == 1.0].groupby(["dataset", "model"])["median_trial_s"].mean()
    full["time"] = [t[(r.dataset, r.model)] for r in full.itertuples()]
    rel = []
    for r in full.itertuples():
        grp = full[full.dataset == r.dataset]["mean"]
        rel.append(r.mean / grp.max() if tasks[r.dataset] == "classification"
                   else grp.min() / r.mean)
    full["relative_score"] = rel
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in MODELS:
        s = full[full.model == m]
        ax.scatter(s["time"], s["relative_score"], label=m, color=COLORS[m])
    ax.set_xscale("log")
    ax.set_xlabel("Median seconds per trial (100% data)")
    ax.set_ylabel("Relative Score (1.0 = Best)" )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "time_vs_score.png", dpi=150)
    plt.close(fig)


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml("configs/datasets.yaml")["datasets"]
    datasets = list(cfg)
    tasks = {d: cfg[d]["task"] for d in datasets}
    df = load_results()
    print(f"{len(df)} jobs loaded")

    summary = (df.groupby(["dataset", "model", "fraction"])
                 .agg(mean=("score", "mean"), std=("score", "std"), n=("score", "size"),
                      accuracy=("accuracy", "mean"), rmse_log=("rmse_log", "mean"),
                      median_trial_s=("median_trial_s", "mean"))
                 .reset_index())
    summary.to_csv(TAB / "summary_all.csv", index=False)
    short = summary[summary.n < 3]
    if len(short):
        print("WARNING: cells with fewer than 3 seeds:\n", short[["dataset", "model", "fraction", "n"]])

    # Main table (100% data)
    full = summary[summary.fraction == 1.0]
    cells, verdicts, tvn = {}, [], []
    for ds in datasets:
        sub = full[full.dataset == ds]
        higher = tasks[ds] == "classification"
        label, tied, gap, pooled = verdict(sub, higher)
        verdicts.append({"dataset": ds, "result": label, "gap": gap, "pooled_std": pooled})
        tv = tree_vs_neural(sub, higher)
        tv["dataset"] = ds
        tvn.append(tv)
        for r in sub.itertuples():
            txt = fmt(r.mean, r.std, tasks[ds])
            if r.model in tied:
                txt = f"**{txt}**" if len(tied) == 1 else f"{txt} †"
            cells[(r.model, ds)] = txt
    main = pd.DataFrame({ds: {m: cells.get((m, ds), "") for m in MODELS} for ds in datasets})
    main.index.name = "model"
    main.to_csv(TAB / "main_table.csv")
    (TAB / "main_table.md").write_text(
        to_md(main) + "\n\nClassification: ROC-AUC (higher is better). Regression: RMSE "
        "(lower is better). Mean ± std over 3 seeds. Bold = clear winner (gap larger than "
        "pooled std). † = tied within pooled std.\n")
    vdf = pd.DataFrame(verdicts).set_index("dataset")
    vdf.to_csv(TAB / "winners.csv")
    tdf = pd.DataFrame(tvn).set_index("dataset")
    tdf.to_csv(TAB / "trees_vs_neural.csv")
    print("\n" + to_md(main) + "\n")
    print(to_md(vdf.round(4)) + "\n")
    print(to_md(tdf.round(4)) + "\n")
    print("Trees vs neural outcomes:", tdf["outcome"].value_counts().to_dict())

    # Secondary metrics at 100%
    sec = full.pivot(index="model", columns="dataset", values="accuracy")[
        [d for d in datasets if tasks[d] == "classification"]]
    seclog = full.pivot(index="model", columns="dataset", values="rmse_log")[
        [d for d in datasets if tasks[d] == "regression"]]
    sec.round(4).to_csv(TAB / "accuracy_100.csv")
    seclog.round(4).to_csv(TAB / "rmse_log_100.csv")

    # Data needed for 95%
    for adjusted, name in [(False, "data_needed_95.md"), (True, "data_needed_95_chance_adjusted.md")]:
        t = pd.DataFrame({ds: {m: data_needed(summary, ds, m, tasks[ds], adjusted)
                               for m in MODELS} for ds in datasets})
        t.index.name = "model"
        (TAB / name).write_text(to_md(t) + "\n")
        print(name + "\n" + to_md(t) + "\n")

    # Timing
    tm = df[df.fraction == 1.0].groupby(["model", "dataset"])["median_trial_s"].mean().unstack()
    tm = tm.reindex(index=MODELS, columns=datasets).round(1)
    (TAB / "median_trial_seconds_100.md").write_text(to_md(tm) + "\n")
    print("Median seconds per trial at 100% data\n" + to_md(tm) + "\n")

    # Dataset properties vs winner
    meta = []
    for ds in datasets:
        X, y = load_dataset(ds)
        m = dataset_metadata(ds, X, y, tasks[ds])
        meta.append({"dataset": ds, "rows": m["rows"], "features": m["features"],
                     "cat_share": m["cat_share"], "result": vdf.loc[ds, "result"],
                     "trees_vs_neural": tdf.loc[ds, "outcome"],
                     "gap_in_std": round(float(tdf.loc[ds, "gap_in_std"]), 2)})
    mdf = pd.DataFrame(meta).set_index("dataset")
    (TAB / "dataset_properties_vs_winner.md").write_text(to_md(mdf) + "\n")
    print(to_md(mdf) + "\n")

    plot_curves(summary, datasets, tasks)
    plot_time_vs_score(summary, df, tasks)
    print("Tables in results/tables, figures in results/figures")


if __name__ == "__main__":
    main()