# External validation on the pre-registered corpus

Rule: OpenML-CC18 (study 99) classification datasets not in the main corpus (data_id not in [6, 15, 24, 28, 31, 32, 36, 37, 44, 54, 59, 151, 182, 554, 1053, 1067, 1120, 1461, 1476, 1478, 1486, 1489, 1590, 40996] and name not in ['fashion-mnist', 'mnist_784', 'segment']), NumberOfInstances >= 1000, NumberOfFeatures - 1 <= 2000; all of them, no further selection; ordered by data_id.
Selected at (UTC): 2026-09-22T01:59:11+00:00; 37 datasets.

Datasets with features and >= 9/9 attack configurations: 37 of 37 (features done: 37; grid cells done: 333/333).

## Out-of-sample (fit on the 31 main-corpus datasets, predict the external corpus)

| index | subset chosen on the 31 (LOO rho) | Spearman on external | 95% CI (Fisher) | permutation p |
|---|---|---|---|---|
| pre-specified index (paper candidates) | ['uniqueness_mean', 'density_mean', 'cluster_sep', 'log_nfeatures'] (0.589) | 0.575 | [0.31, 0.76] | 0.0005 |
| two-factor index (candidates + delta-bar offered) | ['uniqueness_mean', 'density_mean', 'cluster_sep', 'disagree5'] (0.830) | 0.679 | [0.46, 0.82] | 0.0001 |
| fixed {u, rho, S, log d, delta-bar} (no selection) | ['uniqueness_mean', 'density_mean', 'cluster_sep', 'log_nfeatures', 'disagree5'] | 0.681 | [0.46, 0.82] | |

## Standalone Spearman with Risk(D) on the external corpus

| feature | external | main corpus |
|---|---|---|
| uniqueness_mean | 0.455 | 0.532 |
| density_mean | -0.451 | -0.600 |
| outlier_mean | 0.388 | 0.282 |
| entropy | 0.215 | -0.044 |
| cluster_sep | -0.268 | -0.500 |
| log_nfeatures | 0.306 | 0.350 |
| disagree5 | 0.591 | 0.706 |
| disagree20 | 0.756 | 0.778 |

## Nested CV (paper protocol) within the external corpus and on the pooled corpus

| candidates | external only | pooled (31 + external) |
|---|---|---|
| pre-specified index (paper candidates) | 0.515 | 0.584 |
| two-factor index (candidates + delta-bar offered) | 0.716 | 0.829 |

## External corpus: features and Risk(D)

| dataset | n | d | C | u | rho | S | delta5 | Risk(D) | pred (two-factor) | pred (pre-specified) |
|---|---|---|---|---|---|---|---|---|---|---|
| madelon | 2600 | 500 | 2 | 28.940 | 0.035 | 0.001 | 0.476 | 0.962 | 0.778 | 0.727 |
| numerai286 | 30000 | 21 | 2 | 2.197 | 0.468 | 0.001 | 0.494 | 0.832 | 0.736 | 0.698 |
| gesturephasesegmenta | 9873 | 32 | 5 | 2.373 | 1.347 | -0.154 | 0.463 | 0.819 | 0.711 | 0.672 |
| devnagariscript | 30000 | 1024 | 46 | 22.925 | 0.045 | -0.002 | 0.214 | 0.807 | 0.735 | 0.715 |
| cmc | 1473 | 31 | 3 | 2.022 | 1.131 | -0.026 | 0.561 | 0.787 | 0.712 | 0.670 |
| bioresponse | 3751 | 1776 | 2 | 28.695 | 0.051 | 0.006 | 0.296 | 0.786 | 0.752 | 0.712 |
| miceprotein | 1080 | 77 | 8 | 4.497 | 0.241 | 0.018 | 0.018 | 0.770 | 0.606 | 0.695 |
| mfeatfourier | 2000 | 76 | 10 | 6.984 | 0.152 | 0.038 | 0.243 | 0.762 | 0.726 | 0.692 |
| semeion | 1593 | 256 | 10 | 15.385 | 0.066 | 0.050 | 0.137 | 0.762 | 0.694 | 0.696 |
| mfeatkarhunen | 2000 | 64 | 10 | 6.766 | 0.151 | 0.070 | 0.070 | 0.740 | 0.653 | 0.689 |
| firstordertheorempro | 6118 | 51 | 6 | 1.302 | 5.225 | -0.099 | 0.489 | 0.738 | 0.683 | 0.631 |
| steelplatesfault | 1941 | 27 | 7 | 1.978 | 0.728 | 0.014 | 0.308 | 0.731 | 0.704 | 0.681 |
| mfeatzernike | 2000 | 47 | 10 | 3.781 | 0.277 | 0.061 | 0.213 | 0.729 | 0.699 | 0.679 |
| isolet | 7797 | 617 | 26 | 19.555 | 0.052 | 0.047 | 0.204 | 0.722 | 0.723 | 0.694 |
| cnae9 | 1080 | 856 | 9 | 22.958 | 0.075 | -0.096 | 0.223 | 0.714 | 0.741 | 0.723 |
| splice | 3190 | 347 | 3 | 18.550 | 0.057 | 0.008 | 0.314 | 0.704 | 0.756 | 0.718 |
| dna | 3186 | 540 | 3 | 21.922 | 0.048 | 0.005 | 0.317 | 0.692 | 0.756 | 0.718 |
| mfeatpixel | 2000 | 240 | 10 | 11.357 | 0.090 | 0.115 | 0.037 | 0.666 | 0.611 | 0.660 |
| connect4 | 30000 | 168 | 3 | 5.040 | 0.201 | -0.008 | 0.322 | 0.665 | 0.740 | 0.703 |
| qsarbiodeg | 1055 | 41 | 2 | 3.093 | 0.445 | 0.017 | 0.198 | 0.660 | 0.687 | 0.691 |
| mfeatfactors | 2000 | 216 | 10 | 7.934 | 0.132 | 0.159 | 0.049 | 0.656 | 0.625 | 0.651 |
| churn | 5000 | 37 | 2 | 3.552 | 0.296 | 0.112 | 0.147 | 0.643 | 0.660 | 0.656 |
| mfeatmorphological | 2000 | 6 | 10 | 0.176 | 8.969 | 0.221 | 0.307 | 0.641 | 0.611 | 0.544 |
| car | 1728 | 27 | 4 | 3.000 | 0.333 | -0.005 | 0.297 | 0.605 | 0.716 | 0.701 |
| texture | 5500 | 40 | 11 | 1.147 | 0.980 | 0.194 | 0.025 | 0.604 | 0.550 | 0.592 |
| pc4 | 1458 | 37 | 2 | 2.131 | 1.028 | 0.131 | 0.141 | 0.602 | 0.628 | 0.608 |
| pc3 | 1563 | 37 | 2 | 1.939 | 1.007 | 0.108 | 0.136 | 0.601 | 0.635 | 0.627 |
| krvskp | 3196 | 109 | 2 | 4.456 | 0.241 | 0.031 | 0.122 | 0.592 | 0.670 | 0.684 |
| wallrobotnavigation | 5456 | 24 | 4 | 1.666 | 1.126 | -0.018 | 0.172 | 0.592 | 0.650 | 0.667 |
| ozonelevel8hr | 2534 | 72 | 2 | 4.767 | 0.234 | -0.037 | 0.082 | 0.588 | 0.655 | 0.711 |
| junglechess2pcsrawen | 30000 | 6 | 3 | 0.486 | 2.084 | 0.016 | 0.275 | 0.587 | 0.652 | 0.640 |
| pc1 | 1109 | 21 | 2 | 1.106 | 3.124 | 0.418 | 0.094 | 0.568 | 0.581 | 0.562 |
| banknoteauthenticati | 1372 | 4 | 2 | 0.264 | 4.295 | 0.201 | 0.002 | 0.555 | 0.491 | 0.566 |
| phishingwebsites | 11055 | 98 | 2 | 3.852 | 0.535 | 0.054 | 0.056 | 0.555 | 0.622 | 0.665 |
| internetadvertisemen | 3279 | 4668 | 2 | 23.718 | 0.138 | 0.146 | 0.051 | 0.552 | 0.636 | 0.649 |
| sick | 3772 | 75 | 2 | 1.554 | 1.568 | -0.047 | 0.052 | 0.548 | 0.586 | 0.642 |
| wilt | 4839 | 5 | 2 | 0.431 | 3.024 | -0.141 | 0.041 | 0.546 | 0.537 | 0.657 |