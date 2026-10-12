import json
from src.run import run_job, out_path


def test_run_job_and_idempotent(tmp_path):
    r = run_job("brazil_houses", "linear", 0.1, 0, tmp_path, n_trials=2)
    assert r["test_rmse"] > 0 and r["n_train"] > 0
    assert r["task"] == "regression" and r["test_roc_auc"] is None
    saved = json.loads(out_path(tmp_path, "brazil_houses", "linear", 0.1, 0).read_text())
    assert saved["config_hash"] == r["config_hash"]
    assert run_job("brazil_houses", "linear", 0.1, 0, tmp_path, n_trials=2) is None