# Research Process V2: Few-Shot Q-SDTW for Skeleton Action Recognition

**Last updated:** 2026-05-27
**Status:** Active plan, replaces prior open-ended quantum-investigation framing.
**Supersedes:** `RESEARCH_PROCESS_quantum_canonical_angles_HAR.md`
**Project name:** `qca-skeleton-action-recognition` (existing repo)

---

## 0. Read This First

This document is the working plan for turning the existing MSR Action3D quantum-skeleton work into a publishable conference paper at IEEE QCE 2026 (or equivalent). It is written so a coding agent (Claude Code) can execute it phase by phase, and so a human collaborator can audit every claim, every experiment, and every numerical target.

The plan has four sequential phases. **Do not start Phase 2 until Phase 1 acceptance criteria are met.** Each phase has a hard "done" gate that must produce a specific artifact on disk.

This plan is *deliberately* narrower than the original research roadmap. We are not chasing a general quantum-advantage claim on full-data MSR Action3D. We are chasing a specific, defensible result in a specific regime where geometric and quantum kernel methods can plausibly win.

---

## 1. Strategic Pivot: What Changed and Why

### 1.1 What the prior plan got wrong

The original `RESEARCH_PLAN_quantum_canonical_angles_HAR.md` framed the work as "investigating quantum overlap estimation for HAR." That framing has three review-killing problems:

1. It contains no concrete success criterion. A reviewer cannot fail or accept "investigation."
2. The headline result (SWAP Local-SDTW 0.8521 vs PCA+DTW 0.8473 on MSR Action3D, one seed, no statistical test) is statistically indistinguishable from noise.
3. Modern deep skeleton models reach 95–99% on MSR Action3D. Beating those at full-data is unwinnable. Quietly conceding that point while still claiming "quantum HAR advantage" reads as evasive.

### 1.2 The new claim

> Few-shot skeleton action recognition is the regime where deep learning collapses and where geometric/kernel methods remain competitive. We propose **Q-SDTW**, a quantum-estimated local motion-subspace dynamic time warping method, and demonstrate that at 1–5 examples per class, Q-SDTW outperforms both strong classical DTW-family baselines and deep learning few-shot methods (ProtoNet over ST-GCN, fine-tuned ST-GCN, MLP) on MSR Action3D, UTKinect-Action3D, and UTD-MHAD. We provide mechanism evidence that the shot-noise of SWAP-test overlap estimation acts as a structured implicit regularizer at low data, and we provide cross-dataset transfer evidence that the quantum-compatible motion-subspace geometry generalizes more reliably than learned features.

This claim is:

- **Defensible.** The few-shot regime is where deep nets genuinely struggle; published numbers exist to anchor expectations.
- **Specific.** Three datasets, K ∈ {1, 2, 3, 5} shots, named baselines.
- **Falsifiable.** If Q-SDTW does not beat the baselines by a statistically significant margin, the claim fails. We will know before paper writing whether we have a paper.
- **Honest about quantum scope.** Shot-noise-as-regularizer is a mechanism story, not an asymptotic-speedup claim.

### 1.3 What is in scope

- Few-shot recognition: K ∈ {1, 2, 3, 5}, 20-way for MSR Action3D, full N-way for UTKinect and UTD-MHAD.
- Multi-dataset evaluation (MSR + UTKinect + UTD-MHAD).
- Motion features (position, velocity, acceleration, bone vectors).
- SWAP-test (squared overlap) AND Hadamard-test (signed overlap) variants.
- Shot-noise-as-regularizer mechanism analysis.
- Cross-dataset transfer (train kernel hyperparameters on MSR, evaluate on others).
- Early recognition curves.
- Strong classical AND strong deep learning baselines.
- Multi-seed evaluation with paired non-parametric statistical tests.

### 1.4 What is out of scope

- Beating SOTA at full-data MSR Action3D. Not the target.
- Real quantum hardware runs. Simulator only; we make no NISQ-hardware claims.
- Provable quantum-advantage / classical-hardness arguments. Mentioned in limitations; future work.
- NTU-RGB+D full benchmark. Optional stress test, not a deliverable.
- VQD/QPCA baselines. Optional.
- Noise-injection robustness (Gaussian joint noise, frame drop). Optional Phase 3 stretch.

---

## 2. Concrete Numerical Targets

A reviewer will look for these. The plan succeeds only if these numbers materialize.

| Quantity | Target | Comment |
| --- | --- | --- |
| Q-SDTW 1-shot accuracy vs best classical baseline (MSR) | **+5 pp**, p < 0.05 | Paired Wilcoxon across ≥5 seeds × ≥100 episodes |
| Q-SDTW 1-shot accuracy vs ProtoNet/ST-GCN (MSR) | **+10 pp**, p < 0.01 | Deep models are weak at 1-shot; gap should be large |
| Q-SDTW 5-shot accuracy vs best classical (MSR) | **+2 pp**, p < 0.05 | Gap shrinks as K grows, by design |
| Cross-dataset transfer (MSR → UTKinect, 5-shot) | Q-SDTW within 5 pp of in-domain | Demonstrates generalization |
| Shot-noise-as-regularizer effect | Quantum > exact at K=1, gap < 2 pp at K=10 | Inversion at low K |
| Statistical reporting | mean ± std over ≥5 seeds, Wilcoxon p-values | Non-negotiable |

If at the end of Phase 2 we do not see the 1-shot and 5-shot MSR numbers, we stop and reassess. Do not proceed to writing.

---

## 3. The Paper's Anatomy

Four contributions, in order of importance:

1. **Few-shot Q-SDTW**: a quantum-estimated local motion-subspace kernel with DTW alignment, plug-and-play at any K, demonstrated to beat both classical DTW-family and deep few-shot baselines on three datasets.
2. **Shot-noise-as-regularizer**: empirical and theoretical analysis showing that SWAP-test stochasticity has favorable variance structure (vs naive Gaussian noise injection) at low K.
3. **Signed-overlap variant** via Hadamard test: extends prior squared-overlap formulations and recovers orientation-sensitive geometry.
4. **Cross-dataset and early-recognition generalization** evidence.

A reader who finishes the paper should be able to state: "Q-SDTW is a method, here is when it helps, here is why it helps, here are the limits."

---

## 4. Phase 1 — Methodology Lockdown

**Duration:** 1–2 weeks. **Most painful phase. Do not skip.**

The current pipeline is built on single-seed evaluations and the wrong feature set. Every downstream claim depends on fixing this.

### 4.1 Goals

- Reproducible multi-seed evaluation with paired statistical testing.
- Full-data evaluation pipeline (no subset shortcuts) for all classical, exact, and quantum methods.
- Motion features (position, velocity, position+velocity, bone vectors, bone vectors+velocity) implemented and benchmarked.
- All current best results re-run with the above and reported as mean ± std with p-values vs Raw DTW.
- Strong classical baselines added.

### 4.2 Tasks

#### 4.2.1 Refactor result-writing for multi-seed mean ± std + Wilcoxon

- Update `src/eval/result_writer.py` to require: `dataset`, `method`, `feature_mode`, `seed`, `parameters` (dict), `accuracy`, `macro_f1`, `runtime_sec`, `git_commit`, `timestamp`.
- Add `src/eval/statistics.py` with `paired_wilcoxon(per_seed_a, per_seed_b)` and `mean_std(values)`.
- Add `scripts/aggregate_results.py` that takes a raw CSV, groups by `(method, feature_mode, parameters)`, and emits a summary CSV with `acc_mean`, `acc_std`, `f1_mean`, `f1_std`, `n_seeds`, plus pairwise Wilcoxon p-values against Raw DTW.

**Done when:** `python scripts/aggregate_results.py results/raw/dtw_baselines.csv` produces `results/tables/dtw_baselines_summary_v2.csv` with the new schema and per-method p-values.

#### 4.2.2 Implement motion features

Add `src/features/motion_features.py` with these functions, each taking `X ∈ R^{T x 60}` and returning a transformed sequence:

- `position(X) -> X` (identity; baseline).
- `velocity(X) -> ΔX` (first differences; pad first frame).
- `acceleration(X) -> Δ²X` (second differences; pad first two frames).
- `position_velocity(X) -> concat(X, velocity(X), axis=-1)` → returns `R^{T x 120}`.
- `bone_vectors(X) -> B` (compute bone vectors from MSR/UTKinect joint hierarchy; returns `R^{T x (n_bones * 3)}`).
- `bone_velocity(X) -> concat(bone_vectors(X), velocity(bone_vectors(X)))`.

Bone hierarchy for MSR (20 joints) is documented in `configs/msr_action3d.yaml`; add the bone edge list there. Same for UTKinect/UTD-MHAD when added.

**Done when:** `tests/test_motion_features.py` passes with sanity checks (output shapes, first-frame padding, bone-vector orthogonality to root translation).

#### 4.2.3 Re-run all existing experiments with seeds 0–9, full data, every feature mode

Re-run with 10 seeds (current is 5; bump for tighter confidence intervals). Feature modes: `position`, `velocity`, `position_velocity`, `bone_vectors`, `bone_velocity`.

Methods to re-run:

- Raw DTW
- PCA+DTW (k ∈ {16, 32})
- Exact canonical angles, rank ∈ {2, 4, 6}, chordal distance
- **Exact Local-SDTW** (the sliding-window subspace + DTW method) — this needs to be its own clean implementation in `src/distances/local_sdtw.py`; do not rely on the prior smoke-run code path. Use sliding windows of size 5–15 frames, stride 1 or window/2.
- **SWAP Local-SDTW** (the quantum-estimated version), shots ∈ {128, 512, 2048}, ranks ∈ {2, 3}.

All under cross-subject seeded splits. All on full MSR Action3D (566 sequences, no subset).

**Done when:** `results/tables/phase1_main_summary.csv` exists with one row per (method, feature_mode, parameters, dataset), columns: `acc_mean`, `acc_std`, `f1_mean`, `f1_std`, `n_seeds`, `wilcoxon_p_vs_raw_dtw`, `runtime_mean`.

#### 4.2.4 Add strong classical baselines

Add the following in `src/baselines/`:

- `mlp.py`: simple MLP on flattened-and-pooled sequence features (mean pool over time + max pool). 2 hidden layers, 128 units, ReLU. Cross-entropy, Adam. Train per seed.
- `lstm.py`: single-layer LSTM, hidden 64, on per-frame features.
- `random_forest.py`: scikit-learn RandomForest on hand-crafted features (mean joint position, std, range, first/last frame, joint-wise correlations). Trees=500.
- `kdtw.py`: Kernel DTW (Marteau 2009) implementation. Use existing library or implement directly; not deep but kernel-based — important as a "kernel without quantum" baseline.
- (Optional but strongly recommended) `gak.py`: Global Alignment Kernel (Cuturi 2011). Time-series kernel SOTA in the classical-kernel literature.

These run via a new script `scripts/09_run_classical_baselines.py` with the same multi-seed protocol.

**Done when:** Each baseline contributes rows to `results/tables/phase1_main_summary.csv` and at least one is competitive with PCA+DTW (~84% on MSR position-only) so we know they are correctly implemented.

#### 4.2.5 Verify the quantum pipeline on full data

The prior "best" SWAP Local-SDTW number was on a subset. Re-run on full MSR Action3D with 5 seeds, shots ∈ {128, 512, 2048}, ranks ∈ {2, 3}, best feature mode from the position/velocity/bone sweep.

This is expensive but unavoidable. Expect overnight runtime per (rank, shot, seed) configuration with current implementation. **Profile and optimize first:**

- Cache subspace bases per sequence; compute once, reuse across pairs.
- Vectorize the shot-sampling (use binomial sampling, not loops).
- Parallelize across seeds with multiprocessing if not done already.

**Done when:** `results/raw/phase1_quantum_full_data.csv` exists with all (method, feature_mode, rank, shots, seed) combinations, and `results/tables/phase1_main_summary.csv` is updated with the SWAP rows including Wilcoxon p-values vs Raw DTW.

#### 4.2.6 Sanity check — pre-flight

Before declaring Phase 1 done, the following must be true:

- The exact Local-SDTW number must be reproducible from the new clean implementation and within ±0.5 pp of the prior reported 0.847.
- The SWAP Local-SDTW number at shots=2048 must match Exact Local-SDTW within 1 pp (high prediction agreement: ≥ 90%). If it does not, the quantum estimator has a bug.
- All baselines must reach reasonable accuracies (MLP > 65%, LSTM > 70%, RF > 70%, KDTW > 80%). If not, the baseline is broken.

### 4.3 Phase 1 acceptance criteria

Phase 1 is done when:

1. `results/tables/phase1_main_summary.csv` exists with all required rows.
2. Every row has `n_seeds ≥ 5` (≥ 10 preferred).
3. Every method has a `wilcoxon_p_vs_raw_dtw` value.
4. `tests/` passes 100% with no skipped quantum-overlap tests.
5. The figure `results/figures/phase1_full_data_comparison.png` exists, showing all methods × all feature modes as a grouped bar chart with error bars.
6. A short write-up `results/REPORTS/phase1_findings.md` documents which method+feature combination is currently best, whether SWAP Local-SDTW retains any edge at full data with statistical significance, and what we have learned.

If after Phase 1 the SWAP Local-SDTW *does* have a statistically significant edge at full data, that becomes a secondary contribution. If it does not, it gets folded into the few-shot story as "comparable at full data, better at low data" — which is the expected and acceptable outcome.

---

## 5. Phase 2 — Few-Shot Centerpiece

**Duration:** 2–3 weeks. **The paper depends on this phase.**

### 5.1 Goals

- Episode-based few-shot evaluation protocol implemented cleanly.
- Q-SDTW evaluated at K ∈ {1, 2, 3, 5} on MSR Action3D.
- Classical and deep learning few-shot baselines implemented and evaluated.
- Statistical significance established for the headline claim.

### 5.2 Few-shot protocol

Define a *K-shot N-way episode* as:

1. Sample N classes from the test classes (for MSR, all 20 classes by default).
2. For each class, sample K examples as the support set, plus Q query examples (Q=15 by default).
3. Classify each query against the support set using the method's similarity function with 1-NN (or prototype-based for ProtoNet).
4. Record accuracy and macro-F1 per episode.

Episode aggregation:

- 100 episodes per seed.
- 5 seeds.
- Report mean ± 95% confidence interval across 500 episodes.
- Use the *same episodes* (same RNG seed for episode sampling) across methods so that within-episode paired tests are valid.

Implementation:

- New module `src/eval/few_shot.py` with `FewShotProtocol(n_way, k_shot, n_query, n_episodes, episode_seed)`.
- `protocol.generate_episodes(dataset)` yields a list of `(support_X, support_y, query_X, query_y)` tuples deterministically.
- `protocol.evaluate(method, episodes)` returns per-episode accuracy and macro-F1 arrays.

**Done when:** `pytest tests/test_few_shot.py` confirms episodes are reproducible given a seed and that the same episodes are returned for two different method evaluations.

### 5.3 Methods to evaluate in the few-shot setting

For each method, evaluate with K ∈ {1, 2, 3, 5}, best feature mode from Phase 1.

**Classical / kernel baselines:**

- Raw DTW (1-NN)
- PCA+DTW (1-NN)
- KDTW (1-NN; kernel-distance form)
- GAK (1-NN)
- Exact Local-SDTW (1-NN)
- Exact canonical-angle (whole-sequence subspace)

**Deep learning few-shot baselines:**

These three are non-negotiable for the paper:

- **ProtoNet-STGCN**: a pre-trained ST-GCN backbone (pre-trained on a subset of NTU-RGB+D or the train half of MSR), used as a feature extractor; query is classified by nearest prototype in the support set.
- **Fine-tuned ST-GCN**: a small ST-GCN trained from scratch on K-shot support set only (overfitting baseline; expected to be weak).
- **MLP-features ProtoNet**: ProtoNet over MLP-encoded features.

Use the `mmaction2` or `pyskl` library for ST-GCN where possible to avoid re-implementation. If integration is hard, use a published 3-layer ST-GCN reference implementation.

**Quantum / hybrid methods:**

- Q-SDTW = SWAP Local-SDTW, shots ∈ {512, 2048}, rank ∈ {2, 3}, best feature mode.
- (Stretch) Q-SDTW-H = Hadamard-test Local-SDTW (signed overlaps; see Phase 3).

### 5.4 Tasks

1. Implement `src/eval/few_shot.py` (protocol).
2. Implement `src/baselines/protonet.py` and a small ST-GCN backbone in `src/baselines/stgcn.py`.
3. Implement `src/baselines/finetune_stgcn.py` (overfitting baseline).
4. Add `scripts/10_run_few_shot.py` that takes `--method`, `--k-shot`, `--dataset` and writes to `results/raw/few_shot_<dataset>.csv`.
5. Run all (method × K × seed) combinations on MSR Action3D first.
6. Generate `results/figures/few_shot_curve_msr.png`: accuracy vs K, one line per method, with confidence bands.

### 5.5 Phase 2 acceptance criteria

1. `results/tables/few_shot_msr_summary.csv` exists with all required rows.
2. Q-SDTW beats the best classical few-shot baseline at K=1 by ≥ 5 pp with Wilcoxon p < 0.05.
3. Q-SDTW beats ProtoNet-STGCN at K=1 by ≥ 10 pp with Wilcoxon p < 0.01.
4. The figure `results/figures/few_shot_curve_msr.png` exists and is publication-quality.
5. `results/REPORTS/phase2_findings.md` documents the headline result.

**If 5.5.2 or 5.5.3 fail, stop and reassess before Phase 3.** Either the method does not have the advantage we hypothesized, or the protocol/baselines have a bug. Investigate before committing more time.

---

## 6. Phase 3 — Mechanism + Generalization

**Duration:** 2–3 weeks.

### 6.1 Goals

- Explain *why* Q-SDTW works at low data (shot-noise-as-regularizer ablation).
- Add the Hadamard-test signed-overlap variant.
- Establish cross-dataset transfer.
- Add early-recognition curves.

### 6.2 Shot-noise-as-regularizer ablation

This is the mechanism contribution. We want to show that *the noise from SWAP-test sampling has favorable structure* compared to naive Gaussian noise injection on exact distances.

Experimental conditions (all at K=1 MSR Action3D, 100 episodes × 5 seeds):

1. **Exact Local-SDTW** (deterministic, no noise) — reference upper bound.
2. **SWAP Local-SDTW @ shots=2048** — low quantum noise.
3. **SWAP Local-SDTW @ shots=128** — high quantum noise.
4. **Exact + Gaussian noise σ_low** — Gaussian noise on exact distances with variance matched to SWAP@2048.
5. **Exact + Gaussian noise σ_high** — Gaussian noise matched to SWAP@128.
6. **Exact + uniform noise** matched.
7. **Exact + dropout** on subspace bases (drop k% of basis vectors per pair).

Expected pattern: in the K=1 regime, conditions 2 and 3 outperform condition 1 by a small margin; conditions 4–7 do not match the gain, indicating that the variance structure of SWAP shot noise is special, not generic noise.

Output: `results/figures/shot_noise_regularizer.png` and a brief theoretical note in `results/REPORTS/phase3_mechanism.md` explaining the variance structure of SWAP estimator (variance ∝ p(1-p)/N where p is the true overlap, so the noise is *signal-dependent* and *bounded*).

### 6.3 Hadamard test (signed overlaps)

Implement `src/quantum/hadamard_test.py` for signed inner-product estimation. Add `Q-SDTW-H` as a method variant.

Test against orientation-sensitive action pairs (e.g., "raise arm left" vs "raise arm right" if present in the dataset; if not, construct synthetic mirrored sequences as a proof-of-concept).

Optional Phase 3 deliverable — if time permits.

### 6.4 Cross-dataset transfer

Run Q-SDTW with hyperparameters tuned on MSR Action3D *only*, evaluated zero-tuning on:

- UTKinect-Action3D
- UTD-MHAD

Compare against baselines under the same protocol. The story: motion-subspace geometry transfers; learned features do not.

Tasks:

- Add `src/data/utkinect_loader.py` and `src/data/utdmhad_loader.py` (skeleton-only modality for UTD-MHAD).
- Process and validate both datasets via the existing `scripts/00_prepare_dataset.py` / `scripts/01_validate_dataset.py`.
- Run few-shot evaluation on each.

### 6.5 Early recognition

For each test sequence, evaluate accuracy when only the first p% of frames is observed, for p ∈ {20, 30, 40, 50, 70, 100}. K=5 setting, MSR Action3D.

Compare Q-SDTW, Exact Local-SDTW, Raw DTW, ProtoNet-STGCN.

Output: `results/figures/early_recognition_curve.png`.

### 6.6 Phase 3 acceptance criteria

1. `results/figures/shot_noise_regularizer.png` exists and shows the expected pattern.
2. UTKinect and UTD-MHAD few-shot results in `results/tables/few_shot_<dataset>_summary.csv`.
3. Early recognition curve figure exists.
4. `results/REPORTS/phase3_findings.md` summarizes mechanism and transfer findings.

---

## 7. Phase 4 — Paper + Polish

**Duration:** 2–3 weeks. **Includes paper writing.**

### 7.1 Goals

- All figures and tables in publication-ready form.
- Confusion matrices and failure analysis.
- Paper draft complete.

### 7.2 Final figures

Required:

- F1: Method overview (skeleton → windows → subspaces → SWAP-test → DTW alignment). Hand-drawn or TikZ.
- F2: Few-shot accuracy curves (one panel per dataset; methods overlaid).
- F3: Shot-noise-as-regularizer comparison bar chart.
- F4: Cross-dataset transfer heatmap (rows: source dataset; columns: target; cells: accuracy).
- F5: Early recognition curves.
- F6: Confusion matrix comparison (Raw DTW vs Q-SDTW at K=1, MSR).
- F7: (Optional) accuracy-vs-shots ablation.
- F8: (Optional) rank ablation.

### 7.3 Failure analysis

Pick 4–6 failure cases where Q-SDTW misclassifies and analyze: which class pairs are confused, why, and whether they are also confused by Raw DTW.

### 7.4 Paper writing

Write in this order (the prior plan got this right):

1. Method
2. Experiments
3. Results
4. Related work
5. Introduction
6. Abstract
7. Limitations

Limitations section MUST include:

- No quantum hardware results; simulator only.
- No classical-hardness argument; shot-noise-as-regularizer is empirical.
- SWAP gives squared overlap, not signed (Hadamard variant partially addresses).
- We do not beat deep models at full-data MSR Action3D; the advantage is regime-specific.

### 7.5 Phase 4 acceptance criteria

1. `paper/main.pdf` builds clean.
2. All numbers in the paper trace to a CSV in `results/`.
3. `scripts/reproduce_paper.sh` runs end-to-end (allow multi-day runtime; document this).
4. Repo is tagged `v0.1-qce-submission`.

---

## 8. Code Changes — Concrete File-Level List

Group by phase. Files in **bold** are new; others are edits.

### Phase 1

- **`src/features/motion_features.py`** — position, velocity, acceleration, bone vectors, combinations.
- **`src/distances/local_sdtw.py`** — clean Local-SDTW implementation (exact path), used by both classical and quantum variants.
- `src/quantum/overlap_estimation.py` — refactor for vectorized binomial sampling.
- **`src/baselines/mlp.py`**, **`lstm.py`**, **`random_forest.py`**, **`kdtw.py`**, **`gak.py`**.
- **`src/eval/statistics.py`** — Wilcoxon, mean/std helpers.
- `src/eval/result_writer.py` — schema update (add `feature_mode`).
- **`scripts/09_run_classical_baselines.py`**.
- **`scripts/aggregate_results.py`**.
- `scripts/02_run_dtw_baselines.py` — add `--feature-mode` argument.
- `scripts/03_run_subspace_angles.py` — add `--feature-mode` argument.
- `scripts/04_run_quantum_angles_sim.py` — add `--feature-mode`, switch default to full-data, add `--method local_sdtw_swap` mode.
- **`tests/test_motion_features.py`**, **`tests/test_local_sdtw.py`**, **`tests/test_baselines.py`**, **`tests/test_statistics.py`**.
- **`configs/experiment_main_v2.yaml`** — multi-seed, feature-mode list, full-data flag.

### Phase 2

- **`src/eval/few_shot.py`** — episode generator and evaluator.
- **`src/baselines/protonet.py`** — ProtoNet wrapper for any feature extractor.
- **`src/baselines/stgcn.py`** — small ST-GCN backbone (3 layers; use pyskl if possible).
- **`src/baselines/finetune_stgcn.py`** — fine-tuning baseline.
- **`scripts/10_run_few_shot.py`**.
- **`scripts/11_train_stgcn_backbone.py`** — pre-train ST-GCN backbone.
- **`tests/test_few_shot.py`**.
- **`configs/experiment_few_shot.yaml`**.

### Phase 3

- **`src/quantum/hadamard_test.py`** — signed overlap estimation.
- `src/distances/local_sdtw.py` — extend to support Hadamard-test backend.
- **`src/data/utkinect_loader.py`**, **`src/data/utdmhad_loader.py`**.
- **`scripts/12_run_shot_noise_ablation.py`** — exact + injected noise comparisons.
- **`scripts/13_run_early_recognition.py`**.
- **`scripts/14_run_cross_dataset.py`**.
- **`tests/test_hadamard_test.py`**.
- **`configs/utkinect.yaml`**, **`configs/utdmhad.yaml`**.

### Phase 4

- **`paper/main.tex`**, **`paper/sections/*.tex`**, **`paper/refs.bib`**.
- **`scripts/15_generate_paper_figures.py`** — central figure generator for reproducibility.
- **`scripts/reproduce_paper.sh`**.

---

## 9. Risk Register & Fallbacks

### Risk 1 — Q-SDTW does not beat classical baselines at 1-shot

**Probability:** Medium. **Impact:** Plan-killing.

**Mitigation during Phase 1:** Phase 1 acceptance includes a quick few-shot probe (K=1 only, 1 seed, 50 episodes) on the best method + feature combination before declaring Phase 1 done. If the probe shows no advantage, pause and investigate before committing to Phase 2.

**Fallback if confirmed in Phase 2:** Pivot to cross-dataset transfer + early recognition as the primary contributions. Frame the paper as "quantum-compatible motion-subspace geometry as a transferable action descriptor" rather than "few-shot quantum advantage." Still publishable at IEEE QCE or Quantum Machine Intelligence, lower target than the original.

### Risk 2 — Shot-noise-as-regularizer effect does not appear

**Probability:** Medium. **Impact:** Loses one contribution; paper still viable.

**Mitigation:** Run a small pilot at the start of Phase 3 before committing to all conditions. If exact ≥ quantum at all K, drop this contribution and lean harder on cross-dataset transfer.

### Risk 3 — Deep learning few-shot baselines are too strong

**Probability:** Low for K=1 on MSR Action3D (published evidence is they collapse); higher for K=5.

**Mitigation:** ProtoNet with a strong pre-trained backbone (NTU-RGB+D-pretrained ST-GCN) is the worst-case competitor. If it beats Q-SDTW even at K=1, the paper claim must shift to "competitive with deep models without pre-training, while being interpretable and quantum-compatible."

### Risk 4 — UTKinect / UTD-MHAD data access or loader bugs

**Probability:** Medium. **Impact:** Loses generalization story.

**Mitigation:** Start dataset loaders early in Phase 3 (or even late Phase 1). Both datasets are well-documented and widely used.

### Risk 5 — Computational cost of quantum simulation at full-data

**Probability:** Already realized in prior phases. **Impact:** Slows iteration.

**Mitigation:** Phase 1.2.5 includes a profiling and optimization task. Use vectorized binomial sampling, cache subspaces, parallelize across seeds. Consider running the most expensive (rank=3, shots=2048) sweeps overnight on a multi-GPU box.

### Risk 6 — Reviewer asks "why quantum at all if you don't claim speedup?"

**Probability:** Certain.

**Mitigation:** Plan the response in the paper itself. Position: (a) shot-noise-as-regularizer is a real and novel mechanism with structured variance; (b) the formulation is a stepping stone toward provable quantum-kernel methods (cite Huang et al. 2021, Liu et al. 2021); (c) the methodology generalizes to real NISQ hardware as a future-work bridge.

---

## 10. Target Venues & Timeline

### 10.1 Venues

- **Primary**: IEEE QCE (Quantum Week) 2026, typically May–June deadline. Strong fit for hybrid quantum-classical applications papers; reviewers are friendly to this kind of work.
- **Secondary**: Quantum Machine Intelligence (Springer), rolling submission, peer-reviewed.
- **Workshop fallback**: NeurIPS Quantum ML workshop, ICML quantum ML workshop, AAAI quantum ML workshop. Lower bar, gets the work out.
- **Avoid for first submission**: CVPR/ICCV/NeurIPS main track. Bar for quantum ML application papers is unrealistic at first attempt; better to publish at QCE, then aim higher with a follow-up.

### 10.2 Timeline (writing-backward from a May 2026 submission)

| Date | Milestone |
| --- | --- |
| Week 0 | Plan signed off; agent starts Phase 1 |
| Week 2 | Phase 1 acceptance |
| Week 5 | Phase 2 acceptance |
| Week 8 | Phase 3 acceptance |
| Week 10 | Paper draft complete |
| Week 11 | Internal review with advisor |
| Week 12 | Submit |

This is aggressive but achievable. Slip is acceptable on Phase 3 stretch items (Hadamard test, early recognition) but not on Phase 1 or Phase 2 acceptance.

---

## 11. What "Done" Looks Like at the End

The end state of this plan is a repository where:

1. A reviewer can run `scripts/reproduce_paper.sh` and reproduce every paper number.
2. Every paper claim has a CSV trace and a paired statistical test backing it.
3. The Q-SDTW method is implemented cleanly enough to be reused on any new skeleton dataset by writing only a loader.
4. The few-shot evaluation protocol is documented and reusable for future skeleton-action-recognition work.
5. The paper, when read by a quantum-ML reviewer, makes one specific claim ("Q-SDTW outperforms classical and deep baselines in the few-shot regime") backed by three datasets, mechanism analysis, and transfer evidence — no overclaim, no hiding.

That is the deliverable. Everything in this document is in service of that end state. If any task starts feeling like it does not serve that end state, stop and check.

---

## 12. Appendix — Decision Log Template

Every non-trivial choice gets logged here as the work proceeds.

| Date | Decision | Rationale | Made by |
| --- | --- | --- | --- |
| 2026-05-27 | Pivot to few-shot framing | Full-data MSR Action3D is unwinnable vs deep models; few-shot is defensible | Plan author |
| 2026-05-27 | Drop NTU-RGB+D and noise robustness from scope | Time-box scope; both are stretch-only | Plan author |
| 2026-05-27 | Bump seeds 5 → 10 | Tighter confidence intervals for small effect sizes | Plan author |
| (next) | ... | ... | ... |
