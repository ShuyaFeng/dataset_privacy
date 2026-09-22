# Camera-ready pending checks (#1613) — model-free statistics on the 31-dataset corpus

## (A3') W1 between the member and non-member halves vs the nearest-neighbour spacing

W1 from the optimal matching of 3000-point subsamples of each half (3 draws, seed 42 split).

| comparison | Spearman |
|---|---|
| W1 vs u-bar (paper feature, k=5, full data) | 0.981 |
| W1 vs u-bar_5 on the same member subsample | 0.995 |
| W1 vs mean nearest non-member distance (k=1) on the same subsamples | 0.988 |

## W1 as a predictor

| statistic | Spearman with Risk(D) | p | n |
|---|---|---|---|
| W1 (3000-point halves) | 0.487 | 5.45e-03 | 31 |
| u-bar (paper) | 0.532 | 2.07e-03 | 31 |

| dataset | W1 | u-bar_5 (subsample) | u_out,1 (subsample) | u-bar (paper) |
|---|---|---|---|---|
| adult | 1.441 | 1.728 | 1.184 | 1.017 |
| compas | 0.460 | 0.551 | 0.293 | 0.424 |
| purchase100 | 29.828 | 29.429 | 28.685 | 28.032 |
| texas100 | 75.138 | 63.945 | 58.769 | 55.727 |
| nhanes | 1.377 | 1.664 | 1.172 | 1.465 |
| movielens | 0.291 | 0.285 | 0.150 | 0.228 |
| gowalla | 0.331 | 0.362 | 0.217 | 0.142 |
| covtype | 1.976 | 2.046 | 1.189 | 1.050 |
| digits | 4.498 | 5.174 | 3.977 | 4.717 |
| creditg | 6.573 | 7.492 | 6.049 | 6.894 |
| spambase | 3.257 | 3.771 | 2.392 | 3.222 |
| mushroom | 4.011 | 5.036 | 3.271 | 3.758 |
| electricity | 0.746 | 0.872 | 0.463 | 0.418 |
| letter | 1.425 | 1.748 | 1.174 | 1.189 |
| optdigits | 4.055 | 4.585 | 3.622 | 4.245 |
| pendigits | 1.008 | 1.184 | 0.823 | 0.927 |
| satimage | 1.451 | 1.620 | 1.320 | 1.475 |
| segment | 0.818 | 1.022 | 0.524 | 0.859 |
| vehicle | 1.521 | 1.917 | 1.334 | 1.638 |
| ionosphere | 3.688 | 3.595 | 2.909 | 3.229 |
| phoneme | 0.320 | 0.438 | 0.217 | 0.349 |
| bankmarketing | 3.348 | 4.076 | 2.618 | 2.413 |
| magic | 0.913 | 1.085 | 0.798 | 0.811 |
| nomao | 4.718 | 5.591 | 3.457 | 3.235 |
| har | 13.708 | 14.675 | 12.695 | 13.600 |
| gasdrift | 1.650 | 1.838 | 1.045 | 1.070 |
| mnist | 18.464 | 19.744 | 16.709 | 16.201 |
| fashionmnist | 17.322 | 17.930 | 15.533 | 15.309 |
| jm1 | 0.678 | 0.794 | 0.536 | 0.642 |
| kc1 | 0.762 | 0.897 | 0.579 | 0.742 |
| breastw | 0.930 | 1.113 | 0.769 | 0.978 |

## Proposition 2's own statistic on the corpus

| statistic | Spearman with Risk(D) | p | n |
|---|---|---|---|
| Delta-bar = mean delta_i*iota_i, Gaussian kernel, h_median_r5 | 0.259 | 1.60e-01 | 31 |
|   kernel-weighted delta-bar, h_median_r5 | 0.834 | 5.69e-09 | 31 |
|   iota-bar, h_median_r5 | -0.550 | 1.36e-03 | 31 |
| Delta-bar = mean delta_i*iota_i, Gaussian kernel, h_median_pairwise | 0.830 | 7.60e-09 | 31 |
|   kernel-weighted delta-bar, h_median_pairwise | 0.786 | 1.58e-07 | 31 |
|   iota-bar, h_median_pairwise | 0.425 | 1.73e-02 | 31 |
| Delta-bar = mean delta_i*iota_i, Gaussian kernel, h_half_median_pairwise | 0.827 | 1.01e-08 | 31 |
|   kernel-weighted delta-bar, h_half_median_pairwise | 0.779 | 2.50e-07 | 31 |
|   iota-bar, h_half_median_pairwise | 0.463 | 8.73e-03 | 31 |
| k-NN-5 label disagreement (uniform kernel) | 0.699 | 1.23e-05 | 31 |

| dataset | h=median r5 | W_mean | tail weight at k=50 | Delta-bar | delta-bar | iota-bar |
|---|---|---|---|---|---|---|
| adult | 0.983 | 20.30 | 0.298 | 0.0228 | 0.220 | 0.109 |
| compas | 0.325 | 16.74 | 0.171 | 0.0724 | 0.406 | 0.172 |
| purchase100 | 28.670 | 29.86 | 0.589 | 0.0284 | 0.870 | 0.033 |
| texas100 | 46.009 | 25.25 | 0.487 | 0.1056 | 0.815 | 0.122 |
| nhanes | 1.343 | 20.64 | 0.303 | 0.0271 | 0.333 | 0.073 |
| movielens | 0.167 | 13.77 | 0.115 | 0.0098 | 0.076 | 0.157 |
| gowalla | 0.095 | 20.61 | 0.328 | 0.0276 | 0.086 | 0.269 |
| covtype | 0.934 | 17.47 | 0.211 | 0.0265 | 0.329 | 0.086 |
| digits | 4.385 | 22.69 | 0.358 | 0.0135 | 0.161 | 0.058 |
| creditg | 6.752 | 25.62 | 0.456 | 0.0148 | 0.355 | 0.040 |
| spambase | 2.778 | 22.20 | 0.363 | 0.0145 | 0.172 | 0.108 |
| mushroom | 3.662 | 19.36 | 0.265 | 0.0001 | 0.001 | 0.062 |
| electricity | 0.373 | 16.70 | 0.167 | 0.0251 | 0.258 | 0.105 |
| letter | 1.168 | 21.00 | 0.287 | 0.0167 | 0.237 | 0.058 |
| optdigits | 3.952 | 23.88 | 0.399 | 0.0071 | 0.087 | 0.053 |
| pendigits | 0.849 | 19.87 | 0.277 | 0.0039 | 0.025 | 0.074 |
| satimage | 1.331 | 23.21 | 0.397 | 0.0134 | 0.152 | 0.067 |
| segment | 0.627 | 15.42 | 0.141 | 0.0220 | 0.106 | 0.143 |
| vehicle | 1.580 | 20.61 | 0.276 | 0.0249 | 0.457 | 0.058 |
| ionosphere | 2.087 | 16.72 | 0.228 | 0.1113 | 0.225 | 0.251 |
| phoneme | 0.306 | 15.07 | 0.137 | 0.0248 | 0.190 | 0.111 |
| bankmarketing | 1.863 | 16.87 | 0.212 | 0.0404 | 0.138 | 0.177 |
| magic | 0.639 | 20.22 | 0.316 | 0.0197 | 0.224 | 0.122 |
| nomao | 2.230 | 17.74 | 0.240 | 0.0222 | 0.072 | 0.216 |
| har | 12.529 | 25.59 | 0.472 | 0.0084 | 0.164 | 0.051 |
| gasdrift | 0.601 | 15.72 | 0.171 | 0.0053 | 0.021 | 0.200 |
| mnist | 14.749 | 25.74 | 0.470 | 0.0151 | 0.165 | 0.064 |
| fashionmnist | 13.707 | 25.16 | 0.461 | 0.0159 | 0.250 | 0.061 |
| jm1 | 0.282 | 21.62 | 0.366 | 0.0934 | 0.260 | 0.248 |
| kc1 | 0.262 | 21.42 | 0.353 | 0.1015 | 0.178 | 0.305 |
| breastw | 0.570 | 20.77 | 0.349 | 0.0357 | 0.046 | 0.356 |

## Number of classes and chance-normalized label disagreement

| statistic | Spearman with Risk(D) | p | n |
|---|---|---|---|
| log C (number of classes) | 0.427 | 1.66e-02 | 31 |
| chance disagreement 1 - sum p_c^2 | 0.335 | 6.54e-02 | 31 |
| k-NN-5 label disagreement (rebuttal raw) | 0.706 | 9.10e-06 | 31 |
| disagreement / chance | 0.527 | 2.30e-03 | 31 |
| disagreement - chance | 0.210 | 2.56e-01 | 31 |
| normalized class entropy (balance) | -0.019 | 9.18e-01 | 31 |

| nested-CV run | Spearman |
|---|---|
| paper candidates | 0.589 |
| + log C offered | 0.592 |
| + disagreement offered (rebuttal) | 0.818 |
| + disagreement + log C offered | 0.877 |
| + chance-normalized disagreement offered | 0.759 |
| log C alone (LOO) | 0.034 |
| disagreement alone (LOO) | 0.698 |
| chance-normalized disagreement alone (LOO) | 0.467 |
| + N2 offered | 0.873 |
| + N2 + disagreement offered | 0.848 |
| N2 alone (LOO) | 0.860 |
| N1 alone (LOO) | 0.673 |
| N3 alone (LOO) | 0.674 |
| N2 + disagreement (LOO, fixed) | 0.839 |

## Classical data-complexity measures (5,000-row subsample)

| statistic | Spearman with Risk(D) | p | n |
|---|---|---|---|
| N1 (fraction of MST boundary points) | 0.698 | 1.27e-05 | 31 |
| N2 (intra/inter-class NN distance ratio) | 0.860 | 5.42e-10 | 31 |
| N3 (leave-one-out 1-NN error) | 0.689 | 1.84e-05 | 31 |
| F1 (1/(1+max Fisher ratio); lower = easier) | 0.039 | 8.35e-01 | 31 |
| N3 on the full data (rebuttal nn1_error) | 0.641 | 1.03e-04 | 31 |

## Encoding robustness of the scale-free uniqueness (six re-encoded datasets, 30k subsample)

Rank agreement between integer and one-hot encodings: u 0.600 (rebuttal: 0.486); u-tilde 0.600.

| dataset | u int | u one-hot | u~ int | u~ one-hot |
|---|---|---|---|---|
| adult | 1.121 | 2.915 | 0.231 | 0.269 |
| compas | 0.424 | 0.450 | 0.127 | 0.095 |
| mushroom | 0.835 | 3.758 | 0.132 | 0.284 |
| creditg | 3.514 | 6.894 | 0.574 | 0.642 |
| bankmarketing | 1.338 | 2.417 | 0.262 | 0.258 |
| nomao | 2.791 | 3.230 | 0.214 | 0.201 |

| headline (nested CV, paper candidates) | Spearman |
|---|---|
| paper features (full-data run) | 0.589 |
| six datasets replaced by their submitted-encoding features from the 30k re-encoding run | 0.549 |
| six datasets replaced by their alternative-encoding features (30k run) | 0.686 |

## Corpus-size curve (1000 random subsets per size, seed 0)

| m | median | mean | 5th pct | 95th pct | P(rho<0) |
|---|---|---|---|---|---|
| 7 | 0.179 | 0.138 | -0.714 | 0.857 | 0.374 |
| 10 | 0.333 | 0.273 | -0.442 | 0.819 | 0.249 |
| 15 | 0.468 | 0.417 | -0.097 | 0.779 | 0.083 |
| 20 | 0.543 | 0.523 | 0.216 | 0.741 | 0.007 |
| 25 | 0.580 | 0.575 | 0.410 | 0.718 | 0.000 |
