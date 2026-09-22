# Artifact companion files for S&P 2027 paper #1613 (DPRI)

`rebuttal_analysis/` — outputs of `experiments/rebuttal_experiments.py` on the
31-dataset corpus (the numbers reported in the camera-ready): per-fold subset
selection log (`selection.json`, `nested.json`), headline uncertainty
(`bootstrap.json`), alternative risk targets (`robustness.json`), the
eta-squared decomposition (`anova.json`), the theorem factor as a predictor
(`formula.json`), the failure analysis (`failures.json`, `labelproxy.json`),
sensitivity and encoding checks (`sensitivity.json`, `encoding.json`,
`distances.json`, `density_floor.json`), the benchmark-recipe experiment
(`recipe.json`), DP-SGD (`dp.json`, `dp_dp_lira.json`), triage
(`triage*.json`), the generated summary `REBUTTAL_NUMBERS.md`, the ground
truth `ground_truth_risk.csv` and the DPRI features `dpri_features.csv`.

`camera_ready/` — outputs of `experiments/camera_ready_pending.py`: the
model-free checks added for the camera-ready (W1 vs nearest-neighbour
spacing, the Gaussian-kernel two-factor statistic, class-count controls,
classical data-complexity measures, encoding robustness of the scale-free
uniqueness, and the corpus-size curve).

Regenerate everything with `bash scripts/run_local_pipeline.sh all` followed by
`python experiments/camera_ready_pending.py --stage all` (data are downloaded
by `scripts/download_data.py`; the Purchase100/Texas100 release archive must
be placed in `data/raw/`).
