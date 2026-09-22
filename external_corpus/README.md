# Pre-registered external validation corpus (camera-ready, #1613)

`CORPUS.json` freezes the selection rule and the resulting 37 datasets; the commit
that added it is the pre-registration timestamp. Before that commit the only attack
results computed on any of them were a smoke test of the code on five datasets
(banknote-authentication, wilt, pc1, qsar-biodeg, car; loss-threshold attack with RF
and XGBoost only, 10 of their 45 grid cells), run after the rule was fixed and not
used for any decision. `datasets.txt` is the same list, one slug per
line, read by the slurm array scripts. `main_corpus_training_table.csv` is the
31-dataset training table (DPRI features, delta-bar, Risk(D) at seed 42, per-cell
AUCs), so evaluation on the cluster needs no local `results/`.

Rule: OpenML-CC18 classification datasets not already in the main corpus, with
n >= 1000 rows and <= 2000 attributes; all of them, ordered by OpenML data_id.
Preprocessing as in the main corpus (one-hot, missing -> 0, 30k subsample, standardize).

## Cluster run (conda env `dpri`, same as the rebuttal grid)

```bash
git pull
python experiments/external_corpus.py download        # login node, internet; ~37 OpenML fetches -> data/external/
sbatch slurm/external_features_array.sh               # 37 CPU tasks: DPRI features + delta-bar -> results/external/features/
sbatch slurm/external_grid_gpu.sh                     # 111 GPU tasks: MLP x {loss, shadow, lira}
sbatch slurm/external_grid_cpu.sh                     # 222 CPU tasks: {xgboost, rf} x {loss, shadow, lira}
# when everything is done (tasks skip existing outputs, so resubmit failed arrays freely):
python experiments/external_corpus.py evaluate        # -> results/external/EXTERNAL_VALIDATION.md + external_validation.json
```

Partial progress can be inspected any time with `python experiments/external_corpus.py
evaluate --min_configs 1` (datasets with an incomplete grid are then included with
the mean over their finished cells; the final number must use the default, 9/9).

What to send back: `results/external/EXTERNAL_VALIDATION.md`, `external_validation.json`,
and the folders `results/external/features/` and `results/external/mia_grid/` (small JSON files).

## Local alternative

`bash scripts/run_external_local.sh` runs download, features and the whole grid with
4 parallel workers (skips finished cells; LiRA on the larger datasets takes hours).
