# Tabular benchmark: tree ensembles vs. neural networks

This repo compares five model families on six public OpenML tabular datasets under equal conditions: the same splits, the same tuning budget and the same seeds for every model. It also measures how performance changes as training data shrinks.

**Headline result.** XGBoost clearly beat every other model on 3 of 6 datasets (Electricity, Kick, California). On the other 3 (Amazon, Brazil Houses, Allstate) the best models are within about 2 seed standard deviations of each other, so they are reported as ties or near-ties. No neural model clearly won on any dataset. These conclusions come from 6 datasets, 3 seeds, one split per dataset and a CPU-limited neural budget (see Limitations).

## Scope of what was run

| Model | Fractions of training data | Jobs |
| --- | --- | --- |
| Linear / logistic regression | 10%, 25%, 50%, 100% | 72 |
| Random Forest | 10%, 25%, 50%, 100% | 72 |
| XGBoost | 10%, 25%, 50%, 100% | 72 |
| MLP (PyTorch) | **100% only** | 18 |
| FT-Transformer | **100% only** | 18 |

Total: 252 jobs, 7,560 tuning trials, no failed trials. The MLP and FT-Transformer were not run at 10%, 25% and 50% because of CPU compute limits, so their sample-efficiency curves are single points. The sample-efficiency results (research question 2) cover linear, Random Forest and XGBoost only.

## Setup

- Windows laptop, CPU only: 8 logical cores, about 16 GB RAM, no GPU.
- PyTorch 2.14.1 (CPU build). Exact versions are in `requirements.lock.txt`; `requirements.txt` is the readable list.
- Create an environment, then `pip install -r requirements.txt` (or the lock file).

## Reproduce

```
python -m src.reproduce --fast          # 2-trial check of all five models on one small dataset (minutes)
python -m src.reproduce --use-cached    # rebuild every table and figure from the committed results/raw
python -m src.reproduce                 # full rerun at 100% data: all models, 6 datasets, 3 seeds
```

The full rerun takes on the order of a day on the laptop above. The tree and linear fraction runs (10%, 25%, 50%) are run separately:

```
python -m src.run --models linear rf xgb
python -m src.run --models mlp fttransformer --fractions 1.0
```

Jobs are idempotent: a job whose result file exists is skipped, so an interrupted run resumes by rerunning the same command. Tests: `python -m pytest tests -q` (25 tests).

## Datasets

| Name | OpenML ID | Task | Rows used | Features | Categorical share |
| --- | --- | --- | --- | --- | --- |
| amazon | 4135 | Classification | 32,769 | 9 | 100% (max 7,518 categories) |
| electricity | 151 | Classification | 45,312 | 8 | 12.5% |
| kick | 41162 | Classification | 40,000 (stratified cap) | 30 | 53% |
| california | 537 | Regression | 20,640 | 8 | 0% |
| brazil_houses | 42688 | Regression | 10,685 | 8 | 50% |
| allstate | 42571 | Regression | 10,000 (random cap) | 130 | 89% |

Cleaning, identical for every model: rows with a missing target are dropped, text columns become categorical, constant columns are dropped, classification targets become 0/1. No feature engineering.

Judgment calls (all configured in `configs/datasets.yaml`):

- **Brazil Houses:** the four cost columns (HOA, rent, property tax, fire insurance) make up the target, so they were dropped. Rows with target above 50,000 (7 rows) were removed because one extreme row produced 96.6% of the squared error in the test set.
- **Kick:** `BYRNO` (buyer number) and `VNZIP1` (ZIP code) were dropped as ID-like numeric columns.
- **Kick and Allstate:** capped at 40,000 and 10,000 rows so FT-Transformer is feasible on a CPU. The same subsample is used for every model.
- **Electricity:** rows are time-ordered but the split is random. This was kept and is listed as a limitation.

## Splits and seeds

- One fixed split per dataset: 60% train, 20% validation, 20% test, split seed 12345, stratified for classification. Row indices are committed in `splits/`.
- Subsets for 10%, 25% and 50% are drawn from the training portion only and are nested within a seed (10% inside 25% inside 50%).
- Seeds 0, 1, 2 control the subset draw, model initialization and the random-search sampler. They do **not** change the split.
- The test set is evaluated once per job, after the best trial is chosen on validation.
- `src/seeding.py` fixes Python, NumPy and PyTorch randomness and enables deterministic PyTorch. Scikit-learn and XGBoost receive explicit seeds.

## Preprocessing (fitted on the job's training subset only)

| | Linear | Random Forest, XGBoost | MLP, FT-Transformer |
| --- | --- | --- | --- |
| Missing numerics | Median | Median | Median |
| Numeric scaling | Standardize | None | Quantile transform to normal |
| Missing categoricals | Own category | Own category | Own category |
| Categorical encoding | One-hot (categories under 10 training rows grouped) | Ordinal, unseen = -1 | Integer index with learned embeddings, 0 = unseen |

Regression targets are trained on log(1 + y). Predictions are converted back and RMSE is computed on the original scale. For every model, log-scale predictions are clipped to the training subset's target range, which prevents a single extrapolated prediction from exploding RMSE. Log-scale RMSE is also stored as a secondary metric.

## Tuning

30 random-search trials per model, per dataset, per fraction, per seed, scored on the validation set (ROC-AUC, or original-scale RMSE). The best trial's already-trained model is reused with no retraining, for every model. Search spaces are in `configs/search_spaces.yaml` and were frozen before the real runs.

| Model | Searched |
| --- | --- |
| Linear | `C` 1e-4 to 100, or Ridge `alpha` 1e-3 to 1000 (log scale), L2 only |
| Random Forest | trees 100 to 500; max depth {4, 8, 12, 16, 24, none}; min leaf 1 to 20; max features {sqrt, 0.5, 1.0} |
| XGBoost | learning rate 0.01 to 0.3; depth 2 to 10; subsample and column subsample 0.5 to 1; min child weight 1 to 20; L1 1e-8 to 10; L2 1e-3 to 10; up to 2000 trees with early stopping (50 rounds) |
| MLP | 1 to 4 layers; width {64, 128, 256, 512}; dropout 0 to 0.5; learning rate 1e-4 to 3e-3; weight decay 1e-6 to 1e-2; batch {128, 256, 512} |
| FT-Transformer | 1 to 2 blocks; width {32, 64}; 4 heads; attention, feed-forward and residual dropout; learning rate 1e-4 to 3e-3; weight decay 1e-6 to 1e-3; batch {256, 512} |

Neural models use AdamW, early stopping on validation loss (patience 10, maximum 100 epochs) and a per-trial wall-clock cap: 60 s for the MLP and 90 s for FT-Transformer. A capped trial keeps its best validation state and still counts as one of the 30. The FT-Transformer search space is narrower than the MLP's because of CPU limits.

Neural time caps are wall-clock, so laptop sleep distorts them. Any neural job with a single trial longer than 1.3 times its cap was deleted and rerun; the final set passes that check.

## Metrics

- Classification: ROC-AUC (primary), accuracy (secondary; uninformative on imbalanced data such as Amazon at 94% negatives).
- Regression: RMSE on the original scale (primary), RMSE on the log scale (secondary).
- A model counts as a winner only if its mean beats the runner-up by more than the pooled standard deviation across seeds. Otherwise the result is a tie.
- Training time is reported as the median seconds per trial, which a stray sleep cannot distort.

## Results

All numbers are test-set mean ± standard deviation over 3 seeds at 100% of training data. Classification columns are ROC-AUC (higher is better); regression columns are RMSE (lower is better). Bold = clear winner. † = within pooled std of the best.

| model | amazon | electricity | kick | california | brazil_houses | allstate |
| --- | --- | --- | --- | --- | --- | --- |
| linear | 0.8192 ± 0.0003 | 0.8344 ± 0.0000 | 0.7572 ± 0.0002 | 71,169 ± 0 | 3,241 ± 0 | 2,108 ± 2 |
| rf | 0.8171 ± 0.0032 | 0.9696 ± 0.0040 | 0.7509 ± 0.0019 | 53,459 ± 491 | 2,475 ± 3 † | 2,140 ± 6 |
| xgb | 0.8188 ± 0.0064 | **0.9791 ± 0.0005** | **0.7659 ± 0.0014** | **49,007 ± 128** | 2,478 ± 35 † | 2,008 ± 11 † |
| mlp | 0.8200 ± 0.0133 † | 0.9311 ± 0.0036 | 0.7523 ± 0.0036 | 52,689 ± 151 | 2,576 ± 24 | 2,032 ± 47 † |
| fttransformer | 0.8268 ± 0.0037 † | 0.9149 ± 0.0021 | 0.7526 ± 0.0020 | 52,261 ± 628 | 2,491 ± 18 | 2,026 ± 2 |

Linear's std of 0.0000 on Electricity and 0 on California does not mean zero uncertainty: at 100% data the training set and the deterministic linear fit are identical across seeds, and seeds never change the split. Standard deviations with 3 seeds are themselves rough.

### Q1. Do tree ensembles beat neural models under an equal tuning budget?

Partly. The best tree model (XGBoost or Random Forest) was clearly ahead of the best neural model on three datasets, by 7 to 19 pooled standard deviations:

| Dataset | Best tree | Best neural | Gap in std | Reading |
| --- | --- | --- | --- | --- |
| electricity | XGBoost 0.979 | MLP 0.931 | 18.7 | Trees clearly ahead |
| kick | XGBoost 0.766 | FT-Transformer 0.753 | 7.8 | Trees clearly ahead |
| california | XGBoost 49,007 | FT-Transformer 52,261 | 7.2 | Trees clearly ahead |
| allstate | XGBoost 2,008 | FT-Transformer 2,026 | 2.2 | Near-tie |
| brazil_houses | Random Forest 2,475 | FT-Transformer 2,491 | 1.3 | Tie |
| amazon | XGBoost 0.819 | FT-Transformer 0.827 | 1.5 | Near-tie; all five models span 0.817 to 0.827 |

The linear baseline is strong. On Kick it beats Random Forest, the MLP and FT-Transformer and is 0.009 AUC below XGBoost. On Allstate it beats Random Forest.

### Q2. How does performance change with training data? (linear, Random Forest, XGBoost only)

- Electricity, Kick, Brazil Houses and Allstate: Random Forest and XGBoost reach 95% of their full-data score at 10% to 25% of the training data (ratio of score to full-data score, no interpolation).
- California is the one dataset where more data keeps helping the trees: XGBoost needs 50% and Random Forest 100%.
- Amazon: every model needs 100%, and all curves are still rising. It has one column with 7,518 categories.
- Kick has a ranking change: linear is ahead of XGBoost at 10% and 25%, and XGBoost overtakes it near 50%.
- Linear on Brazil Houses gets worse as data grows (RMSE about 2,960 at 10%, 3,240 at 100%). The cause was not investigated.
- The plain 95% ratio is lenient for AUC because 0.5 is the floor. `results/tables/data_needed_95_chance_adjusted.md` gives the stricter classification version.
- The MLP and FT-Transformer have no curve; they were run at 100% only.

Learning curves are in `results/figures/curve_<dataset>.png`. Full tables are in `results/tables/`.

### Q3. Which dataset properties predict the winner?

| Dataset | Rows | Features | Categorical share | Result |
| --- | --- | --- | --- | --- |
| electricity | 45,312 | 8 | 12.5% | XGBoost |
| kick | 40,000 | 30 | 53% | XGBoost |
| california | 20,640 | 8 | 0% | XGBoost |
| brazil_houses | 10,685 | 8 | 50% | Tie: Random Forest, XGBoost |
| allstate | 10,000 | 130 | 89% | Tie: XGBoost, MLP |
| amazon | 32,769 | 9 | 100% | Tie: all models within about 0.01 AUC |

The clear tree wins are on the larger datasets (20,000 to 45,000 rows) with mostly numeric or mixed features. The near-ties are on the smallest dataset and on the two datasets dominated by categorical columns. Size and categorical share are confounded here, and six datasets are too few for a regression or a significance claim, so this is a descriptive pattern only.

### Training time

Median seconds per trial at 100% data: linear under 1 s; Random Forest 0.8 to 8.6 s; XGBoost 1.7 to 11.8 s; MLP 17 to 60 s; FT-Transformer 25 to 91 s. See `results/figures/time_vs_score.png`. Most FT-Transformer medians sit at the 90 s cap.

## Limitations

- **Six datasets** chosen for variety, not sampled at random, limit how far the conclusions generalize.
- **The neural models were compute-limited.** Most FT-Transformer trials hit the 90 s cap, and on Electricity every MLP and FT-Transformer trial was at or near its cap. The FT-Transformer search space was narrowed, and 30 trials may under-tune it and the MLP. The tree advantage on Electricity (18.7 std) partly reflects this and the random split on time-ordered data. A GPU and a larger budget could change the neural results.
- **The neural models ran at 100% data only**, so the sample-efficiency question is answered for linear, Random Forest and XGBoost.
- **Three seeds** give rough variance estimates, so the tie rule is approximate.
- **One split per dataset.** Seeds do not vary the split, so reported standard deviations understate total uncertainty.
- **Row caps and cutoffs** (Kick 40,000, Allstate 10,000, Brazil Houses target at most 50,000) and **dropped columns** change those datasets from their public form. They apply equally to every model.
- **Electricity** is time-ordered but split randomly, which favors models that interpolate between neighboring timestamps.
- **Validation is used for both tuning and early stopping**, so validation scores are slightly optimistic. The test set is used once per job.
- **Early stopping uses log-scale loss** for regression, while tuning selects on original-scale RMSE. The mismatch applies to the neural models only.
- **The 95% measure** uses only four tested fractions and no interpolation, and it is lenient for AUC.
- **CPU timing** is not comparable with GPU timing, and neural results depend on machine speed because the caps are wall-clock. GPU results can also differ slightly across hardware even with fixed seeds.

## Repo layout

```
configs/    global.yaml, datasets.yaml, search_spaces.yaml
src/        data.py, splits.py, preprocess.py, tune.py, run.py, analyze.py, reproduce.py, seeding.py, env_info.py
src/models/ linear.py, rf.py, gbdt.py, mlp.py, fttransformer.py, nn_train.py
splits/     committed row indices, one JSON per dataset
results/    raw/ (one JSON per job), tables/, figures/
tests/      25 tests
```

Each result file stores test and validation metrics, best hyperparameters, all 30 trial logs, timing, library versions, the git commit and a hash of the three config files.