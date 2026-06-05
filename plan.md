# Research Plan: What Makes a Dataset Privacy-Sensitive?
**Target: IEEE S&P (Oakland) 2027**
**Last updated: 2026-06-03 (session 3)**

---

## Paper Goal

Prove that dataset-level structural statistics can predict aggregate membership inference risk before any model is trained. Introduce DPRI (Dataset Privacy Risk Index), establish theoretical bounds, expose benchmark bias in MIA literature, and provide actionable security implications.

**Proposed title:** *What Makes a Dataset Privacy-Sensitive? Towards Predicting Privacy Risk Before Model Training*

---

## Phase Overview

| Phase | Goal | Status |
|---|---|---|
| 1 | Resolve fatal flaws (ground truth, threat model, Feldman positioning) | IN PROGRESS (1.1 running on cluster; 1.2, 1.3 DONE) |
| 2 | Theoretical framework (formal bounds) | DONE (draft) |
| 3 | Experimental design and execution | CODE DONE — waiting on cluster results |
| 4 | Security narrative (Oakland-specific) | CODE DONE (CLI tool); narrative needs cluster results |
| 5 | Paper writing and submission | TODO |

---

## Phase 1: Resolve Fatal Flaws

**Why this comes first:** Without solving these, the paper will be desk-rejected or receive fatal reviews regardless of experimental quality.

### Task 1.1 — Resolve the Ground Truth Circularity

**Problem:** We want to predict privacy risk *before* training any model. But the current ground truth is `Risk(D) = AUC_MIA`, which is model-dependent. This contradicts the model-free framing.

**Plan:**
- Adopt Path B (pragmatic, achievable in timeline):
  - For each dataset, run 3 MIA attacks × 3 model families
    - Attacks: LiRA (Carlini 2022), loss-threshold, shadow model
    - Models: MLP, XGBoost, Random Forest
  - Define `Risk(D)` = upper envelope of AUC scores across this attack/model grid
  - Explicitly reframe the claim: DPRI predicts *cross-model-family average risk*, not a model-free absolute

**Success criteria:**
- Risk(D) is stable across attack/model combinations (variance < 0.05 within each dataset)
- The reframing is logically consistent and defensible in a reviewer rebuttal

**If successful:** Proceed to Task 1.2. Document the grid results as Table 1 in the paper.

**If unsuccessful (high variance across models/attacks):**
- Investigate *why* AUC varies — is it attack strength or model capacity?
- If variance comes from attack strength: standardize on LiRA only (strongest), and argue LiRA represents the adversary's best case
- If variance comes from model capacity: add model capacity as a covariate in the regression, and update H1 accordingly
- If variance is irreducible: pivot to Path A (theoretical Bayes-optimal bound) — this would delay timeline by ~2 months but produce a stronger paper

---

### Task 1.2 — Write Formal Threat Model

**Plan:**
Write a precise threat model covering:
- **Adversary goal:** Determine whether a target sample x was in the training set D
- **Adversary capability:** Black-box query access to trained model f (primary); optionally white-box (secondary experiment)
- **Adversary knowledge:** Knows data domain distribution P_data; does not know exact D
- **Defender goal:** Before training, compute DPRI(D) to quantify and anticipate membership inference risk
- **Scope:** Centralized training setting; FL and DP settings treated as extensions in Section 7

**Success criteria:**
- Threat model fits on half a page
- Covers both offensive use (attacker uses DPRI to select targets) and defensive use (data holder uses DPRI pre-training)
- Consistent with all experiments in Phase 3

**If successful:** Proceed to Task 1.3.

**If threat model is contested in review:**
- Prepare rebuttal argument: our threat model is strictly a generalization of Shokri et al. 2017 (already accepted at Oakland)
- Consider adding a second threat model variant for white-box access

---

### Task 1.3 — Write Feldman 2020 Positioning Statement

**Problem:** Feldman (2020) showed rare/atypical samples are memorized more. This overlaps with our H2. Reviewers will ask "what's new over Feldman?"

**Plan:**
Draft a 3-sentence positioning statement:
> "Feldman (2020) analyzes *sample-level* memorization of *individual* rare examples, observable only after training. We study a complementary and prior question: can *dataset-level* distributional statistics predict *aggregate* privacy risk before any training occurs? DPRI operates at the dataset level, produces a single pre-training risk score, and is actionable without a trained model."

Key differentiators to emphasize:
1. Unit of analysis: sample-level (Feldman) vs. dataset-level (us)
2. Timing: post-training observation (Feldman) vs. pre-training prediction (us)
3. Output: memorization certificate (Feldman) vs. deployable risk score (us)

**Success criteria:**
- The statement can be inserted verbatim into Introduction and Related Work
- No reviewer could argue we simply "repackage" Feldman

**If positioning is weak:**
- Add an experiment that Feldman cannot do: show DPRI predicts risk for datasets where *no* individual sample is rare (high density datasets) — proving DPRI captures aggregate structure beyond individual outlier effects

---

## Phase 2: Theoretical Framework

**Why this is required for Oakland:** Empirical correlation alone (R² > 0.8) is a measurement paper (→ PETS). Oakland expects a formal theorem with proof.

### Task 2.1 — Derive Formal Upper Bound

**Starting point:** Yeom et al. (2018) proved:
$$\text{Adv}(\text{MIA}) \leq \sqrt{\frac{1}{2} \text{KL}(P_{\text{in}} \| P_{\text{out}})}$$

**Plan:**
- Step 1: Express KL(P_in || P_out) in terms of local density ρ(x) and uniqueness u(x)
- Step 2: Show that for a sample x with low ρ(x) (low density) and high u(x) (high uniqueness), the KL term is bounded below by a function of these quantities
- Step 3: Derive: Adv(MIA) ≤ f(ρ, u) where f is explicit and computable from data before training
- Target form: `Adv(MIA) ≤ C · (u(x) / ρ(x)^α)` for constants C, α to be determined

**Mathematical tools needed:**
- Information geometry (KL divergence under local density approximation)
- Possibly: results from statistical learning theory (covering numbers, VC dimension)

**Success criteria:**
- A formal theorem with a complete proof (no hand-waving)
- The bound is computable from data alone (no trained model required)
- Proof fits in ~1 page in the appendix

**If bound is too loose (trivially large RHS):**
- Pursue a tighter bound using the Bayes-optimal MIA advantage formulation from Sablayrolles et al. (2019)
- Alternatively: prove a *lower* bound — show that high-DPRI datasets must have high MIA advantage (this is harder but also more interesting)
- Last resort: reframe as "theoretical motivation" rather than a tight theorem, and compensate with stronger empirics

**If proof is intractable in timeline:**
- Consult with a theory-focused collaborator (flag this risk early)
- If still intractable: submit to PETS/CCS first, use review feedback to strengthen theory for Oakland resubmission

---

### Task 2.2 — Prove Bound Tightness (or Acknowledge Gap)

**Plan:**
- Construct a synthetic dataset family where the bound is tight (RHS ≈ actual Adv(MIA))
- Show this tightness holds for real datasets with known distribution structure

**Success criteria:**
- At least one case where bound is tight within factor of 2
- Clear discussion of when/why gap exists for other cases

**If tightness cannot be shown:**
- Explicitly state gap and give intuitive explanation
- Add experiment showing bound is predictively useful even if loose (e.g., ranking datasets by bound matches ranking by actual AUC)

---

## Phase 3: Experimental Design and Execution

### Task 3.1 — Dataset Selection

**Required datasets (8 minimum, 4 domains):**

| Domain | Dataset | Purpose |
|---|---|---|
| Tabular | Adult | Classic MIA baseline |
| Tabular | COMPAS | Classic MIA baseline |
| Tabular | Texas100 | Benchmark — expect benchmark bias finding |
| Tabular | Purchase100 | Benchmark — expect benchmark bias finding |
| Medical | MIMIC-III (public subset) or NHANES | High-stakes domain |
| Recommendation | MovieLens-1M | Sparse matrix, different structure |
| Mobility | Gowalla or Geolife | High individual uniqueness |
| Image (optional) | CIFAR-10 subset | Cross-modality validation |

**Success criteria:**
- All datasets publicly available and downloadable with documented provenance
- Preprocessing pipeline is deterministic and reproducible

**If MIMIC-III access is unavailable:**
- Substitute with CDC NHANES (fully public, health domain)
- Or use the Breast Cancer Wisconsin dataset (public, medical context)

---

### Task 3.2 — MIA Attack Implementation

**Required attacks:**
1. **LiRA** (Carlini et al. 2022) — primary, strongest attack
2. **Loss-threshold attack** — baseline
3. **Shadow model attack** (Shokri et al. 2017) — for comparison

**Required models:**
- MLP (2 hidden layers, tuned per dataset)
- XGBoost
- Random Forest

Total experiment grid: 8 datasets × 3 attacks × 3 models = 72 configurations

**Success criteria:**
- All 72 configurations produce valid AUC scores
- Compute time is feasible (estimate: ~2 weeks on a single GPU server for non-image data)
- Results are logged and reproducible via fixed random seeds

**If LiRA is computationally prohibitive:**
- LiRA requires training 64+ shadow models per configuration — this may be expensive
- Mitigation: use LiRA on 4 representative datasets; use loss-threshold on all 8
- Document this limitation explicitly

---

### Task 3.3 — DPRI Feature Computation

**Features to compute for each dataset:**

| Feature | Method | Library |
|---|---|---|
| Sample Uniqueness u(x) | k-NN distance (k=5) | scikit-learn |
| Local Density ρ(x) | k-NN density estimate | scikit-learn |
| Outlier Score | LOF + Isolation Forest (ensemble) | scikit-learn |
| Feature Entropy H(X) | Per-feature Shannon entropy | scipy |
| Cluster Separation S | Silhouette score on class labels | scikit-learn |

**Aggregate to dataset level:**
- Mean, median, 90th percentile of per-sample scores
- Final DPRI = weighted combination (weights learned via regression in Task 3.4)

**Success criteria:**
- Features computed for all 8 datasets
- Feature values are stable across random subsampling (bootstrap test)

**If features are unstable (high bootstrap variance):**
- Increase k in k-NN to reduce noise
- Aggregate using median instead of mean (more robust to outliers)

---

### Task 3.4 — Regression and Validation

**Plan:**
- Input: DPRI features (5 per dataset, mean + 90th pct = 10 values)
- Output: Risk(D) = upper envelope AUC from Task 1.1
- Model: Linear regression (interpretable) + Random Forest regression (non-linear check)
- Metric: R², Spearman rank correlation
- Validation: Leave-one-dataset-out cross-validation

**Success criteria:**
- R² > 0.75 (acceptable), > 0.85 (strong), > 0.90 (excellent)
- Spearman correlation > 0.8
- Linear model performs comparably to RF (suggests linear relationship, more interpretable)

**If R² < 0.75:**
- Diagnose: which datasets are outliers? Why?
- Add interaction terms between features
- Consider that image data may need separate treatment (add modality as covariate)
- If R² still low: the hypothesis may be wrong — investigate whether model capacity is a confound we missed. This is a valid negative finding.

---

### Task 3.5 — Key Finding Validation

**Finding 1: Data structure explains more variance than model architecture**
- Experiment: For fixed dataset, vary model architecture (depth, width). Measure AUC variance.
- For fixed model, vary dataset. Measure AUC variance.
- Compare: Var(AUC | dataset changes) vs. Var(AUC | model changes)
- Target: dataset explains > 60% of total variance

**Finding 2: Benchmark bias**
- Compute DPRI for Purchase100 and Texas100
- Compare to real-world datasets using KS test and Wasserstein distance
- Show benchmark datasets lie in the tail of DPRI distribution
- Implication: MIA papers using only benchmarks may report inflated risk numbers

**Finding 3: DP calibration depends on DPRI**
- For each dataset, apply DP-SGD with epsilon ∈ {0.1, 1, 10}
- Measure AUC drop relative to non-DP baseline
- Show that DP benefit (AUC drop) correlates with DPRI: high-DPRI datasets benefit more
- Implication: choosing epsilon without knowing DPRI may under/over-protect

**If any finding does not hold:**
- Finding 1 failure: reframe as "model architecture is not the *primary* driver" and show dataset effect is *significant*, even if not dominant
- Finding 2 failure: still report the distribution comparison — even no significant difference is interesting (it would mean benchmarks are representative)
- Finding 3 failure: investigate whether the relationship is non-monotone; a U-shaped relationship would also be a novel finding

---

### Task 3.6 — Ablation Study

- Remove each DPRI feature one at a time
- Measure R² drop
- Target: each feature contributes meaningfully (R² drop > 0.03 when removed)

**If a feature is redundant (R² drop < 0.01):**
- Remove it from DPRI to simplify the index
- Update the paper to reflect reduced feature set

---

## Phase 4: Oakland Security Narrative

### Task 4.1 — Adversarial Use of DPRI

**Plan:**
Demonstrate that an attacker can use DPRI as a *reconnaissance tool*:
- Attacker computes DPRI on candidate target datasets
- Prioritizes attacking high-DPRI datasets
- Show this strategy improves attack efficiency vs. random target selection

**Experiment:**
- Simulate attacker with budget of B MIA queries
- Compare: random dataset selection vs. DPRI-guided selection
- Metric: AUC achieved per unit of query budget

**Success criteria:**
- DPRI-guided selection achieves higher AUC with fewer queries
- Effect is statistically significant

**If effect is small:**
- Frame as "DPRI provides information gain" rather than "DPRI enables a new attack"
- Still include as a security implication subsection

---

### Task 4.2 — Policy Implication

Write a 1-paragraph policy recommendation:
- Organizations publishing datasets should compute DPRI before release
- High-DPRI datasets require stronger anonymization or DP guarantees
- Regulatory frameworks (GDPR, HIPAA) should incorporate dataset-level risk metrics, not just epsilon values

This goes in the Conclusion/Discussion section.

---

### Task 4.3 — Open-Source Tool

**Plan:**
- Implement DPRI as a Python package (CLI + library)
- Input: raw dataset (CSV or numpy array)
- Output: DPRI score, per-feature breakdown, risk category (Low/Medium/High)
- Host on GitHub, include in paper as artifact

**Success criteria:**
- Tool runs on all 8 experimental datasets and reproduces paper results
- README includes quickstart example
- Code is clean enough for public release

**Timeline:** Implement alongside Phase 3 experiments (reuse the same codebase)

---

## Phase 5: Paper Writing

### Structure (10 pages + references)

| Section | Pages | Key content |
|---|---|---|
| Abstract | 0.25 | 3 findings stated explicitly |
| 1. Introduction | 1.5 | Problem, gap, contributions (4 bullets) |
| 2. Background & Threat Model | 1.0 | MIA background, formal threat model |
| 3. Theoretical Analysis | 2.0 | DPRI motivation, bound proof |
| 4. DPRI Definition | 1.0 | 5 features with formulas |
| 5. Experimental Setup | 0.5 | Datasets, models, attacks |
| 6. Results | 2.5 | R², 3 findings, ablation |
| 7. Security Implications | 0.5 | Attacker view, policy, DP calibration |
| 8. Related Work | 0.5 | Feldman, Yeom, Sablayrolles, others |
| 9. Conclusion | 0.25 | Summary + future work |

### Writing Order

1. Experiments first (Phase 3 must be complete)
2. Theory section (Phase 2 results inserted)
3. Introduction last (write after knowing what you actually found)
4. Abstract absolute last

### Submission Checklist

- [ ] All claims have either a theorem or an experiment supporting them
- [ ] No claim says "model-free" without qualification (see Task 1.1)
- [ ] Threat model appears in Section 2
- [ ] Feldman positioning appears in Introduction and Related Work
- [ ] All datasets are public and cited
- [ ] Code is available (GitHub link in paper)
- [ ] Reproducibility appendix included
- [ ] Proofread by at least 2 non-author readers

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Ground truth variance is high across models | Medium | Fatal | Switch to LiRA-only ground truth (adversarial best case) |
| Theoretical bound is too loose | Medium | High | Reframe as motivation; compensate with stronger empirics |
| R² < 0.75 | Low-Medium | High | Add features, investigate outlier datasets |
| MIMIC-III access denied | Medium | Low | Use NHANES or WBCD as substitute |
| LiRA too expensive to run | Medium | Medium | Run LiRA on 4 datasets, loss-threshold on all 8 |
| Feldman overlap too strong | Low | High | Add dataset-level experiment Feldman cannot do |
| Paper rejected at Oakland | Medium | Medium | Revise and target CCS or USENIX Security |

---

## Related Work to Read (in order)

1. Yeom et al. (2018) — *Privacy Risk in Machine Learning: Analyzing Connections to Overfitting*
2. Feldman (2020) — *Does Learning Require Memorization? A Short Tale About a Long Tail*
3. Carlini et al. (2022) — *Membership Inference Attacks from First Principles* (LiRA)
4. Shokri et al. (2017) — *Membership Inference Attacks against Machine Learning Models*
5. Sablayrolles et al. (2019) — *White-box vs Black-box: Bayes Optimal Strategies for Membership Inference*
6. Homer et al. (2008) — *Resolving Individuals Contributing Trace Amounts of DNA to Highly Complex Mixtures*
7. Song & Mittal (2021) — *Systematic Evaluation of Privacy Risks of Machine Learning Models*

---

## Current Status

**Active task:** Phase 1 Task 1.1 — fixing LiRA bug, regenerating ground truth

### First MIA grid run (2026-06-03) — REVEALED A BUG

First full 63-config grid completed. `check_ground_truth_variance.py` output:
- Within-dataset AUC std: texas100=0.239, purchase100=0.223, adult=0.217 (5/7 FAILED <0.05 threshold)
- **LiRA mean AUC = 0.4819 < 0.5** ← impossible for a correct attack

**Root cause:** `attack_lira` used `true_labels.append(1 if j < half else 0)` —
an arbitrary membership label unrelated to actual shadow-model membership.
Score (mean in_conf - mean out_conf) was uninformative w.r.t. this label → AUC ~0.5.

**Why this also caused the variance failure:** within each dataset, the 3 LiRA
configs sat at ~0.48 while the 6 loss/shadow configs sat at ~0.9, inflating
within-dataset std. Verified: texas100 min_auc=0.4755 is exactly the LiRA value.
Fixing LiRA should collapse the variance.

**Fix applied (this session):** rewrote `attack_lira` as proper online LiRA —
per-(target,sample) Gaussian likelihood ratio in logit space, with ground-truth
membership labels. loss_threshold and shadow_model verified correct, NOT changed.

**This is bug-fixing, not p-hacking:** AUC < 0.5 is an objective implementation
error; a correct attack cannot do worse than random.

### Next actions (on cluster, in order)
1. `git pull`
2. Delete only the LiRA results: `rm results/mia_grid/*__lira__*.json`
3. Re-submit grid (skip-if-exists reruns only the 21 LiRA configs):
   `sbatch slurm/mia_gpu_array.sh && sbatch slurm/mia_cpu_array.sh`
4. After completion: re-run `check_ground_truth_variance.py`
   - If within-dataset std now <0.05 for most datasets → ground truth is clean, proceed
   - If still high →走 plan.md Task 1.1 mitigation (model capacity covariate / LiRA-only)
5. In parallel, `run_dpri.py` can run now (independent of ground truth) — but note
   it did NOT appear in all_analysis.txt, so it may have crashed or still be running.
   Check separately.
6. Once ground truth clean + DPRI done: `run_regression.py` then `run_findings.py`

### Completed (code on GitHub)
- All Phase 1-4 code + full paper draft with \TODO{} placeholders
- See git log for details

### Remaining TODO
- Confirm ground truth stability after LiRA fix
- Debug why run_dpri.py produced no output
- Fill paper \TODO{} placeholders once results are clean
- Theory: flesh out full proof of Theorem 1, tightness experiment (Task 2.2)
