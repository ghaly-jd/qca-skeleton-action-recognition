# Phase 1 Findings — Methodology Lockdown

**Date:** 2026-06-04  
**Dataset:** MSR Action3D (566 sequences, 20 classes, cross-subject split)  
**Seeds:** 10 (seeds 0–9)  
**Summary table:** `results/tables/phase1_main_summary.csv`  
**Full detail table:** `results/tables/phase1_full_detail.csv`  
**Figure:** `results/figures/phase1_full_data_comparison.png`

---

## 1. Best Feature Mode Per Method

| Method | Best Feature Mode | Acc Mean | Acc Std | Macro-F1 | Wilcoxon p vs raw_dtw |
|---|---|---|---|---|---|
| **local_sdtw_exact** | position_velocity | **0.8764** | 0.0000 | 0.8576 | 0.0020 |
| **local_sdtw_swap** | position_velocity | **0.8727** | 0.0054 | 0.8521 | 0.0020 |
| pca_dtw | position_velocity | 0.8545 | 0.0000 | 0.8358 | — (tied) |
| raw_dtw | position_velocity | 0.8545 | 0.0000 | 0.8365 | — (baseline) |
| kdtw | bone_velocity | 0.8404 | 0.0011 | 0.8219 | 0.0020 |
| gak | bone_vectors | 0.7695 | 0.0204 | 0.7619 | 0.0020 |
| mlp | bone_velocity | 0.7462 | 0.0244 | 0.7364 | 0.0020 |
| random_forest | bone_vectors | 0.7407 | 0.0087 | 0.7204 | 0.0020 |
| lstm | bone_velocity | 0.7120 | 0.0192 | 0.6886 | 0.0020 |

7 baselines + 2 Q-SDTW variants (local_sdtw_exact and local_sdtw_swap), all 10 seeds. Parameters selected as best acc_mean per (method, feature_mode) across the full parameter grid.

**Key finding:** `position_velocity` dominates for subspace and DTW methods. Learned baselines (MLP, LSTM, RF) benefit from `bone_velocity` or `bone_vectors` because bone-vector features provide view-invariant geometry that aids learned pattern matching. Velocity-only features are notably poor for DTW-family methods (raw_dtw velocity: 0.600) because the Euclidean frame distance between velocity vectors is unstable.

---

## 2. Full-Data SWAP Local-SDTW Results

**Does SWAP Local-SDTW show a statistically significant edge over raw_dtw at full data?**

Feature mode `position_velocity`, window=15, stride=5, rank=2, shots=2048:

| Comparison | Delta Acc | Wilcoxon p |
|---|---|---|
| Exact Local-SDTW vs raw_dtw | +2.18pp | **0.0020** |
| SWAP Local-SDTW vs raw_dtw  | +1.82pp | **0.0020** (W=0: all 10 seeds positive) |

SWAP@w=15 is only **0.37pp below exact@w=15** (0.8727 vs 0.8764) — within shot noise at shots=2048. Both methods use the same Phase 2 configuration; these are directly comparable.

**Caution on full-data Wilcoxon for deterministic classifiers:**  
raw_dtw, exact Local-SDTW, pca_dtw, and kdtw show **zero seed-variance** on the fixed cross-subject split. Their 10 per-seed accuracy values are all identical, making the Wilcoxon test degenerate (all differences equal the same constant → p ≈ 2^{-10}). These p-values confirm the methods produce different outcomes but do not capture sampling variance. SWAP is exempt from this caveat: std=0.0054 and W=0 reflect a genuine stochastic test. Phase 2 few-shot episodes restore across-episode variance for all methods.

**Shot-noise regularization signal (preliminary, do not cite):**  
`bone_velocity` rank=3 with SWAP@512 and SWAP@2048 both outperform the deterministic exact baseline by ~0.7–0.8pp at full data. This is interesting, but the exact baseline has zero seed-variance (std=0.0000), so the Wilcoxon here tests "is SWAP's mean above the exact constant" — which is valid, but the effect has not been shown to generalize beyond this single fixed split. Do not treat this as a population claim or mention it in any abstract draft. Phase 3 Step 3.3 will test this properly with matched Gaussian/uniform noise baselines across many episodes.

---

## 3. Position-Velocity Breakdown (All Methods)

At the dominant feature mode `position_velocity`:

| Method | Acc Mean | Acc Std | Wilcoxon p |
|---|---|---|---|
| local_sdtw_exact | 0.8764 | 0.0000 | 0.0020 |
| local_sdtw_swap | 0.8727 | 0.0054 | 0.0020 |
| pca_dtw | 0.8545 | 0.0000 | — (tied) |
| raw_dtw | 0.8545 | 0.0000 | — |
| kdtw | 0.8185 | 0.0011 | 0.0020 |
| random_forest | 0.7200 | 0.0069 | 0.0020 |
| mlp | 0.7011 | 0.0240 | 0.0020 |
| lstm | 0.6960 | 0.0217 | 0.0020 |
| gak | 0.6731 | 0.0268 | 0.0020 |

Ranking is stable: subspace methods > DTW-family > learned baselines. PCA+DTW adds nothing over raw_dtw at position_velocity **at the best window (none)**. Investigation confirmed PCA is running correctly — k=16 reduces 120→16 dims with 90.4% explained variance and produces genuinely different distance matrices (max Δ=0.34). The tied accuracy at `window=none` is real: both methods classify exactly 235/275 correctly via different distances but identical 1-NN assignments. The finding "PCA adds nothing" is correct; it is a genuine null result, not a code bug. Notably, pca_dtw does diverge from raw_dtw at `window=0.1` and `window=0.2` (k=16 is slightly worse, k=32 slightly better), so the PCA effect is present but zero-sum across the window sweep.

---

## 4. Few-Shot Probe Results (Step 1.23)

**Protocol:** K=1, 20-way, 50 episodes, 1 seed (seed=0), feature=position_velocity, window=15, stride=5, rank=2, shots=2048 for SWAP, all 566 MSR sequences pooled.

| Method | Acc Mean | Acc Std† | Gap vs raw_dtw |
|---|---|---|---|
| raw_dtw | 0.6090 | 0.0644 | — |
| local_sdtw_exact | 0.6622 | 0.0509 | **+5.32pp** |
| local_sdtw_swap (shots=2048) | 0.6586 | 0.0505 | **+4.96pp** |

†**Acc Std is across-episode std (50 episodes, 1 seed), not across-seed std.** These numbers give the within-seed episode variance, not the across-seed reproducibility. They are appropriate for "does a gap exist" but cannot be cited as population confidence intervals. Phase 2 runs 5 seeds × 100 episodes and reports across-seed std as the primary uncertainty estimate.

Both methods exceed the acceptance threshold of ≥2–3pp gap. The exact method touches the Phase 2 headline target of +5pp.

SWAP is 0.36pp below exact at 1 seed — within expected shot-noise variance. At shots=2048 with rank=2 and D=120 features, estimator variance is small and will average out over Phase 2's 5 seeds × 100 episodes.

**Random baseline for 20-way 1-shot:** 5% (1/20). Raw DTW at 60.9% confirms the task is well-conditioned.

---

## 5. Go / No-Go Decision

**Decision: GO — proceed to Phase 2.**

Criteria met:
- [x] Full-data Exact Local-SDTW beats raw_dtw by +2.18pp (p=0.0020)
- [x] Full-data SWAP Local-SDTW beats raw_dtw by +1.82pp (p=0.0020, W=0) at window=15
- [x] 1-shot probe gap ≥ 2–3pp: Exact +5.32pp, SWAP +4.96pp ✓
- [x] All baselines meet sanity floors (KDTW 84%, GAK 77%, MLP 75%, RF 74%, LSTM 71%)
- [x] 10 seeds per configuration throughout
- [x] Shot-noise regularization signal visible (preliminary)

**Phase 2 configuration:**
- Method: Q-SDTW = SWAP Local-SDTW
- Feature mode: `position_velocity`
- Parameters: window=15, stride=5, rank=2, shots=2048
- K values: 1, 2, 3, 5
- Episodes: 100 per seed, 5 seeds

**Window=15 justification (fully closed):** SWAP@w=15 was run post-Phase-1-gate on the full 291×275 split, 10 seeds (`results/raw/local_sdtw_swap_w15_phase1.csv`). Result: mean=0.8727, std=0.0054, W=0 (all 10 seeds beat raw_dtw), p=0.0020. SWAP@w=15 is only 0.37pp below exact@w=15 and +1.27pp above SWAP@w=10. The Phase 2 config (window=15) now has its own direct full-data evidence.

**Strategic note — real competitor is Exact Local-SDTW, not raw_dtw:**  
Exact Local-SDTW is a fully classical method (SVD per window + DTW) that already beats raw_dtw by +2.18pp at full data and +5.32pp at K=1. A reviewer will ask: "why not just use Exact Local-SDTW everywhere?" The Phase 2 and Phase 3 answer must be: (a) the shot-noise acts as a structured regularizer in the few-shot regime (tested in Step 3.3), and (b) the Hadamard variant enables signed overlaps (Step 3.5–3.8). If neither effect materializes, Exact Local-SDTW becomes the stronger story. Phase 2's acceptance criterion should be: Q-SDTW beats **Exact** Local-SDTW by a meaningful margin at K=1 with p < 0.05, not just raw_dtw.

**Fallback note:** If Phase 2 Wilcoxon at K=1 fails the +5pp / p<0.05 target, pivot to cross-dataset transfer + early recognition as the primary contribution (per Risk 1 in RESEARCH_PROCESS_V2).
