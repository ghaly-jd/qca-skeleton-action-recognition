# Research Process: Quantum-Estimated Canonical Angles for HAR

## Purpose

This document turns `RESEARCH_PLAN_quantum_canonical_angles_HAR.md` into a practical execution roadmap. The goal is to make the research finishable in clear phases, with each phase producing concrete code, data artifacts, result files, figures, or paper text.

The main strategic rule is:

> Build the clean classical pipeline first, then add quantum overlap estimation.

The minimum publishable version should prove the core idea on MSR Action3D:

1. Clean dataset processing.
2. Raw DTW baseline.
3. Exact canonical-angle classification.
4. Quantum-estimated subspace affinity on a subset or full split.
5. Shot ablation.
6. Reproducible scripts and honest limitations.

The stronger version adds UTKinect, PCA+DTW, optional VQD/QPCA+DTW, noise robustness, encoding ablations, exact-vs-quantum distance error, and multi-seed statistical tests.

---

## Phase 0: Project Setup and Research Hygiene

### Goal

Create a clean repository skeleton where every experiment can be run from scripts, every result has metadata, and notebooks are used only for exploration.

### Why This Matters

This project can easily become messy because it touches datasets, classical baselines, quantum circuits, ablations, and paper figures. A clean structure keeps the research reproducible and makes it easier to show partial results early.

### Steps

1. Create the repository structure.
   - Add `configs/`, `data/`, `src/`, `scripts/`, `results/`, `tests/`, `notebooks/`, and `paper/`.
   - Keep raw data under `data/raw/`.
   - Keep generated arrays under `data/processed/`.
   - Keep experiment outputs under `results/raw/`, aggregated tables under `results/tables/`, and figures under `results/figures/`.

2. Add environment files.
   - Create `requirements.txt`.
   - Create `environment.yml` if using conda.
   - Create `pyproject.toml` if the project will be run as an installable Python package.

3. Add basic project utilities.
   - `src/utils/seed.py` for deterministic seeds.
   - `src/utils/io.py` for JSON, CSV, NPZ, and path helpers.
   - `src/utils/logging.py` for consistent script output.
   - `src/utils/paths.py` for project-relative paths.

4. Add global experiment config files.
   - `configs/msr_action3d.yaml`
   - `configs/utkinect.yaml`
   - `configs/experiment_main.yaml`
   - `configs/experiment_quantum.yaml`
   - `configs/paper_figures.yaml`

5. Define reproducibility metadata.
   - Every result row should include `dataset`, `method`, `seed`, `parameters`, `accuracy`, `macro_f1`, `runtime_sec`, `git_commit`, and `timestamp`.

### Deliverables

- Repository folders exist.
- Environment files exist.
- Config files exist.
- Utility modules exist.
- Result-writing convention is defined.

### Done When

The repository can import `src` modules and run a placeholder script without path errors.

---

## Phase 1: MSR Action3D Dataset Pipeline

### Goal

Load MSR Action3D, convert each skeleton sequence into a clean `T x D` matrix, save processed files, and validate labels, subjects, frame counts, and feature dimensions.

### Why This Matters

All later claims depend on data correctness. If the loader is wrong, the canonical-angle and quantum results will be meaningless even if the math is implemented correctly.

### Steps

1. Put raw MSR Action3D data in `data/raw/MSRAction3D/`.
   - Document the data source and any manual download steps in `README.md`.
   - Do not commit large raw data files unless explicitly intended.

2. Implement `src/data/msr_loader.py`.
   - Parse raw skeleton files.
   - Extract sequence id, subject id, action id, and repetition if available.
   - Return each sequence as frames of 3D joint coordinates.

3. Implement preprocessing in `src/data/preprocessing.py`.
   - Remove invalid frames.
   - Flatten joints into joint-major order: `[j1_x, j1_y, j1_z, ..., j20_x, j20_y, j20_z]`.
   - Root-center each frame using the selected hip or spine joint.
   - Add optional scale normalization.
   - Add optional feature z-score normalization.
   - Save enough metadata to reproduce the exact choices.

4. Save processed data.
   - Write `data/processed/msr_action3d/sequences.npz`.
   - Write `data/processed/msr_action3d/labels.npy`.
   - Write `data/processed/msr_action3d/subjects.npy`.
   - Write `data/processed/msr_action3d/metadata.json`.

5. Create cross-subject splits in `src/data/splits.py`.
   - Fixed split: train subjects `1, 3, 5, 7, 9`; test subjects `2, 4, 6, 8, 10`.
   - Seeded variants for seeds `0, 1, 2, 3, 4`.
   - Save splits under `data/splits/`.

6. Implement validation in `src/data/validation.py`.
   - Check sequence count.
   - Check number of classes.
   - Check number of subjects.
   - Check frame count statistics.
   - Check feature dimension.
   - Check train/test subject separation.
   - Check labels are not missing or malformed.

7. Add scripts.
   - `scripts/00_prepare_dataset.py`
   - `scripts/01_validate_dataset.py`

### Commands

```bash
python scripts/00_prepare_dataset.py --dataset msr_action3d
python scripts/01_validate_dataset.py --dataset msr_action3d
```

### Deliverables

- `data/processed/msr_action3d/sequences.npz`
- `data/processed/msr_action3d/labels.npy`
- `data/processed/msr_action3d/subjects.npy`
- `data/processed/msr_action3d/metadata.json`
- `data/splits/msr_cross_subject.json`
- `results/tables/dataset_summary.csv`

### Done When

`scripts/01_validate_dataset.py` prints a clean dataset summary and writes `results/tables/dataset_summary.csv`.

---

## Phase 2: Exact Sequence-as-Subspace Pipeline

### Goal

Represent each sequence as a low-dimensional motion subspace and classify actions using canonical-angle distances.

### Why This Matters

This is the core non-quantum contribution. The quantum part only becomes meaningful after the exact classical subspace method works.

### Steps

1. Implement normalization helpers in `src/features/normalization.py`.
   - Sequence centering.
   - Feature standardization.
   - Optional frame L2 normalization.
   - Clear train-only fitting for any global statistics.

2. Implement per-sequence SVD in `src/features/sequence_subspace.py`.
   - Input: `X` with shape `T x D`.
   - Center sequence: `X_centered = X - mean(X)`.
   - Compute SVD.
   - Return top `r` right singular vectors as `U_X` with shape `D x r`.
   - Handle short sequences with a minimum frame rule.
   - Validate orthonormality of basis vectors.

3. Implement canonical angles in `src/distances/canonical_angles.py`.
   - Compute `M = U_X.T @ U_Y`.
   - Compute singular values of `M`.
   - Clip singular values to `[-1, 1]`.
   - Return angles `theta = arccos(sigma)`.

4. Implement subspace distances in `src/distances/subspace_distances.py`.
   - Main: chordal distance.
   - Ablations: projection distance, mean angle, max angle, min angle.

5. Implement 1-nearest neighbor classification in `src/eval/knn.py`.
   - Precompute train subspaces.
   - For each test sequence, find nearest training sequence.
   - Return predicted label and nearest neighbor distance.

6. Implement metrics in `src/eval/metrics.py`.
   - Accuracy.
   - Macro-F1.
   - Confusion matrix if useful for diagnostics.

7. Add tests.
   - `tests/test_subspace.py`
   - `tests/test_canonical_angles.py`
   - Check identical subspaces have distance near zero.
   - Check orthogonal or nearly orthogonal subspaces have larger distance.
   - Check basis sign flips do not change distances.

8. Add experiment script.
   - `scripts/03_run_subspace_angles.py`
   - Run `r` values `[1, 2, 3, 4, 5, 6, 8, 10, 12]`.
   - Run distances `chordal`, `projection`, `mean_angle`, and `max_angle`.
   - Run seeds `0, 1, 2, 3, 4`.

### Commands

```bash
python scripts/03_run_subspace_angles.py \
  --dataset msr_action3d \
  --r-values 1 2 3 4 5 6 8 10 12 \
  --distance chordal projection mean_angle max_angle \
  --seeds 0 1 2 3 4
```

### Deliverables

- `results/raw/canonical_angles_exact.csv`
- Initial rank sweep plot: accuracy vs subspace rank `r`
- Initial distance ablation table

### Done When

`results/raw/canonical_angles_exact.csv` exists, has one row per seed/rank/distance setting, and the best chordal result can be summarized in a table.

---

## Phase 3: Classical Baselines

### Goal

Implement temporal-alignment baselines so the canonical-angle method can be evaluated against standard sequence similarity methods.

### Why This Matters

The paper must show whether subspace geometry is competitive with temporal alignment. DTW is the main reference point.

### Steps

1. Implement raw DTW in `src/distances/dtw.py`.
   - Use Euclidean frame distance.
   - Support optional path-length normalization.
   - Support window ratios: no window, `0.1`, and `0.2`.

2. Add `scripts/02_run_dtw_baselines.py`.
   - Run raw DTW on full `T x 60` sequences.
   - Use the same splits and seeds as canonical angles.

3. Implement PCA projection in `src/features/pca_projection.py`.
   - Fit PCA only on training frames.
   - Transform train and test sequences frame by frame.
   - Test `k` values `[4, 8, 12, 16, 24, 32]`.

4. Add PCA+DTW run mode.
   - Either extend `scripts/02_run_dtw_baselines.py` or create a clear script option.
   - Save raw and PCA+DTW rows in a consistent format.

5. Optional: add VQD/QPCA+DTW.
   - Treat this as optional until the core paper is alive.
   - Do not block the main comparison on this baseline.

6. Aggregate main classical results.
   - Add `scripts/08_generate_tables.py`.
   - Produce mean and standard deviation across seeds.

### Commands

```bash
python scripts/02_run_dtw_baselines.py \
  --dataset msr_action3d \
  --seeds 0 1 2 3 4 \
  --window-ratios none 0.1 0.2

python scripts/08_generate_tables.py --table main_results
```

### Deliverables

- `results/raw/dtw_baselines.csv`
- PCA+DTW rows in `results/raw/dtw_baselines.csv` or a separate raw CSV
- `results/tables/main_results.csv`

### Done When

`results/tables/main_results.csv` includes at least Raw DTW, PCA+DTW, and Exact Canonical Angles.

---

## Phase 4: Quantum-Estimated Subspace Affinity

### Goal

Estimate basis-vector overlaps with quantum circuits, construct a quantum-estimated subspace affinity, and evaluate it as a sequence similarity measure.

### Why This Matters

This is the quantum contribution. The first publishable quantum version does not need to estimate signed canonical angles. It should honestly estimate squared overlaps and call the result quantum-estimated subspace affinity.

### Steps

1. Implement state preparation in `src/quantum/state_preparation.py`.
   - Use amplitude encoding.
   - Convert real basis vectors into valid quantum states.
   - Since MSR features are 60D, pad vectors to the next power of two if needed.
   - Normalize each encoded vector.
   - Keep a deterministic path for simulator reproducibility.

2. Implement SWAP test in `src/quantum/swap_test.py`.
   - Estimate `|<u_i | v_j>|^2`.
   - Return both estimated overlap and shot metadata.
   - Validate on identical, orthogonal, and random normalized vectors.

3. Implement overlap estimation wrapper in `src/quantum/overlap_estimation.py`.
   - Provide a common interface for `swap_test`.
   - Leave room for `hadamard_test` later.
   - Cache repeated overlap estimates if possible.

4. Implement quantum-estimated subspace affinity in `src/distances/quantum_estimated_angles.py`.
   - For subspaces `U_X` and `U_Y`, estimate all `r^2` squared overlaps.
   - Build affinity matrix `A_ij = |<u_i | v_j>|^2`.
   - Compute similarity `S = mean(A)`.
   - Compute distance `d_q_aff = 1 - S`.

5. Start with a small subset.
   - Use `train_per_class = 3`.
   - Use `test_per_class = 2`.
   - Use `r` values `[2, 3, 4]`.
   - Use shots `[128, 256, 512, 1024]`.

6. Add `scripts/04_run_quantum_angles_sim.py`.
   - Save one row per dataset, seed, rank, shots, method.
   - Include runtime because quantum simulation can be expensive.

7. Compare against exact subspace similarity.
   - Confirm the quantum-affinity nearest neighbors are not random.
   - Confirm accuracy generally stabilizes as shots increase.

### Commands

```bash
python scripts/04_run_quantum_angles_sim.py \
  --dataset msr_action3d \
  --r-values 2 3 4 \
  --shots 128 256 512 1024 \
  --subset true \
  --seeds 0 1 2 3 4
```

### Deliverables

- `results/raw/quantum_subspace_affinity.csv`
- Unit tests for quantum overlap estimation
- Small-subset quantum result summary

### Done When

Quantum-estimated subspace affinity runs on the subset and produces stable CSV output for all requested seeds, ranks, and shot counts.

---

## Phase 5: Shot Ablation and Exact-vs-Quantum Error

### Goal

Measure how quantum estimation quality changes with shot count and how close the estimated distance is to the exact classical subspace distance.

### Why This Matters

This is where the quantum result becomes scientifically interpretable. Even if accuracy is lower than the exact method, the paper can show convergence behavior and shot-noise limits.

### Steps

1. Implement shot ablation script.
   - `scripts/05_run_shot_ablation.py`
   - Use shots `[64, 128, 256, 512, 1024, 2048]`.
   - Use ranks `[2, 4]`.
   - Run seeds `0, 1, 2, 3, 4`.

2. Implement distance-error mode.
   - Sample sequence pairs.
   - Compute exact distance.
   - Compute quantum-estimated distance.
   - Save absolute error.
   - Use `num_pairs = 500`, ranks `[2, 4, 6]`, shots `[128, 512, 1024, 2048]`.

3. Generate figures.
   - Accuracy vs shots.
   - Distance error vs shots.
   - Optional: runtime vs shots or runtime vs rank.

4. Interpret results carefully.
   - If accuracy improves with shots, emphasize convergence.
   - If it does not, inspect normalization, rank, and affinity definition.
   - Do not claim exact canonical-angle estimation unless signed inner products are implemented.

### Commands

```bash
python scripts/05_run_shot_ablation.py \
  --dataset msr_action3d \
  --r-values 2 4 \
  --shots 64 128 256 512 1024 2048 \
  --seeds 0 1 2 3 4

python scripts/05_run_shot_ablation.py \
  --dataset msr_action3d \
  --mode distance_error \
  --num-pairs 500 \
  --r-values 2 4 6 \
  --shots 128 512 1024 2048
```

### Deliverables

- `results/raw/shot_ablation.csv`
- `results/raw/distance_error.csv`
- Shot ablation figure
- Exact-vs-quantum error figure

### Done When

The paper can include a clear statement about how shot count affects quantum-estimated subspace similarity.

---

## Phase 6: Normalization, Encoding, and Distance Ablations

### Goal

Understand which preprocessing and distance choices make subspace geometry useful for skeleton action recognition.

### Why This Matters

Subspace methods are sensitive to representation. This phase turns that sensitivity into a controlled analysis instead of a hidden weakness.

### Steps

1. Implement normalization ablation.
   - Feature standardization: `none`, `zscore`.
   - Scale normalization: `true`, `false`.
   - Frame L2 normalization: `true`, `false`.
   - Sequence centering: `true`, `false`.
   - Root centering remains `true`.

2. Run for ranks `[2, 4, 6]`.

3. Implement distance metric ablation.
   - Chordal.
   - Projection.
   - Mean angle.
   - Max angle.
   - Optional min angle.

4. Generate heatmaps and tables.
   - Encoding/normalization heatmap.
   - Distance metric ablation table.

5. Choose the main paper setting.
   - Select the best or most stable normalization.
   - Keep the selection rule honest and reproducible.

### Commands

```bash
python scripts/07_run_encoding_ablation.py --dataset msr_action3d
python scripts/03_run_subspace_angles.py \
  --dataset msr_action3d \
  --r-values 2 4 6 \
  --distance chordal projection mean_angle max_angle \
  --seeds 0 1 2 3 4
```

### Deliverables

- `results/raw/encoding_ablation.csv`
- Updated `results/raw/canonical_angles_exact.csv` with distance ablation rows
- Encoding heatmap
- Distance ablation table

### Done When

The paper can justify the chosen normalization and distance metric.

---

## Phase 7: Noise Robustness

### Goal

Test whether subspace-angle and quantum-affinity methods degrade gracefully under common skeleton corruptions.

### Why This Matters

Skeleton data is noisy. Showing robustness makes the method more credible and may reveal that subspace geometry captures stable global motion patterns.

### Steps

1. Implement noise transforms.
   - Gaussian joint noise.
   - Random frame dropping.
   - Temporal jitter.
   - Missing joints.

2. Define noise levels.
   - Gaussian noise std: `[0.00, 0.01, 0.03, 0.05, 0.10]`.
   - Frame drop ratio: `[0.00, 0.10, 0.20, 0.30]`.
   - Temporal jitter ratio: `[0.00, 0.05, 0.10]`.
   - Missing joint ratio: `[0.00, 0.05, 0.10, 0.20]`.

3. Compare methods.
   - Raw DTW.
   - Exact canonical angles.
   - Quantum-estimated subspace affinity.

4. Generate accuracy degradation plots.

### Commands

```bash
python scripts/06_run_noise_ablation.py --dataset msr_action3d
```

### Deliverables

- `results/raw/noise_robustness.csv`
- Noise robustness figure

### Done When

The paper has evidence about method behavior under skeleton noise.

---

## Phase 8: UTKinect Generalization

### Goal

Repeat the core experiments on UTKinect to show the method is not specific to MSR Action3D.

### Why This Matters

A second dataset makes the paper stronger. It is not required for the earliest demo, but it is important for the strong version.

### Steps

1. Add UTKinect raw data under `data/raw/UTKinect/`.

2. Implement `src/data/utkinect_loader.py`.
   - Parse sequences, labels, and subjects.
   - Validate joint count and feature dimension.
   - Keep the preprocessing interface consistent with MSR.

3. Process and validate UTKinect.
   - Save under `data/processed/utkinect/`.
   - Add split definitions under `data/splits/`.

4. Run core experiments.
   - Dataset validation.
   - Raw DTW.
   - Exact canonical angles.
   - Best quantum-estimated setting from MSR.
   - Shot ablation if time allows.

5. Update main results table.
   - Include both datasets.
   - Report mean and standard deviation across seeds where applicable.

### Commands

```bash
python scripts/00_prepare_dataset.py --dataset utkinect
python scripts/01_validate_dataset.py --dataset utkinect
python scripts/02_run_dtw_baselines.py --dataset utkinect --seeds 0 1 2 3 4
python scripts/03_run_subspace_angles.py --dataset utkinect --seeds 0 1 2 3 4
python scripts/04_run_quantum_angles_sim.py --dataset utkinect --subset true --seeds 0 1 2 3 4
```

### Deliverables

- `data/processed/utkinect/`
- `results/raw/utkinect_results.csv` or dataset-specific rows in shared result files
- Updated `results/tables/main_results.csv`

### Done When

The main results table includes MSR Action3D and UTKinect.

---

## Phase 9: Tables, Figures, and Statistical Reporting

### Goal

Turn raw experiment CSV files into paper-ready tables, figures, and statistical summaries.

### Why This Matters

The paper should be generated from scripts, not manual copy-paste. This also protects against reporting mistakes.

### Steps

1. Implement result aggregation in `src/eval/result_writer.py`.
   - Consistent CSV schema.
   - Git commit capture.
   - Timestamp capture.
   - Parameter serialization.

2. Implement statistics in `src/eval/statistics.py`.
   - Mean and standard deviation.
   - Paired Wilcoxon signed-rank test for per-seed comparisons.
   - Do not report p-values with fewer than 5 seeds.

3. Implement table generation.
   - `scripts/08_generate_tables.py`
   - Dataset summary table.
   - Main results table.
   - Ablation table.

4. Implement figure generation.
   - `scripts/09_generate_figures.py`
   - Method overview figure.
   - Accuracy vs rank.
   - DTW vs canonical angles.
   - Accuracy vs shots.
   - Distance error vs shots.
   - Encoding heatmap.
   - Noise robustness.

5. Validate consistency.
   - Check that every paper number appears in a generated CSV.
   - Check that every figure can be regenerated from raw results.

### Commands

```bash
python scripts/08_generate_tables.py
python scripts/09_generate_figures.py
```

### Deliverables

- `results/tables/dataset_summary.csv`
- `results/tables/main_results.csv`
- `results/tables/ablation_results.csv`
- `results/figures/*.pdf` or `results/figures/*.png`
- `results/paper_results.csv`

### Done When

The paper's tables and figures are fully reproducible from scripts.

---

## Phase 10: Paper Writing

### Goal

Write the paper around the strongest honest result: sequence subspaces and canonical angles provide a meaningful alternative to temporal alignment, and quantum overlap estimation can approximate useful subspace similarity.

### Why This Matters

The paper should not depend on claiming quantum advantage. The scientific contribution is the formulation, the comparison, and the quantum-estimation analysis.

### Writing Order

1. Method.
   - Sequence-as-subspace representation.
   - Canonical angles.
   - Chordal distance.
   - 1-NN classification.
   - Quantum-estimated subspace affinity.

2. Experiments.
   - Datasets.
   - Splits.
   - Baselines.
   - Hyperparameters.
   - Metrics.
   - Reproducibility details.

3. Results.
   - Main comparison.
   - Rank analysis.
   - Shot ablation.
   - Exact-vs-quantum distance error.
   - Normalization/noise ablations if complete.

4. Related Work.
   - Skeleton action recognition.
   - DTW and sequence similarity.
   - Subspace methods and canonical angles.
   - Quantum overlap estimation.

5. Introduction.
   - Motivation.
   - Gap.
   - Contributions.

6. Abstract.

7. Limitations.
   - No quantum advantage claim.
   - SWAP test estimates squared overlaps, not signed overlaps.
   - Quantum simulation cost.
   - Dataset scale.

### Deliverables

- `paper/main.tex`
- `paper/sections/`
- `paper/figures/`
- `paper/refs.bib`

### Done When

There is a complete draft with generated tables and figures inserted.

---

## Phase 11: Reproduction Package and Submission Prep

### Goal

Make the project easy to rerun and ready to share with a reviewer, advisor, or workshop audience.

### Steps

1. Create `scripts/reproduce_paper.sh`.
   - Prepare data if already downloaded.
   - Validate data.
   - Run main experiments.
   - Generate tables.
   - Generate figures.

2. Freeze environment.
   - Update `requirements.txt`.
   - Update `environment.yml`.
   - Record Python version.

3. Final result audit.
   - Confirm every result row has seed, parameters, commit, timestamp, runtime.
   - Confirm no paper number is manually typed without a generated source.

4. Repository release.
   - Push GitHub repository.
   - Tag release as `v0.1-qce-submission`.
   - Archive on Zenodo if possible.

5. Advisor/demo package.
   - Clean repo structure.
   - Processed MSR summary.
   - Exact canonical-angle accuracy table.
   - Raw DTW baseline.
   - Accuracy vs rank plot.
   - One paragraph explaining the quantum extension.

### Commands

```bash
bash scripts/reproduce_paper.sh
git tag v0.1-qce-submission
git push origin v0.1-qce-submission
```

### Done When

The project can be rerun from documented commands and shared with a clear commit hash.

---

## Critical Path

If time is limited, do the work in this order:

1. Repository structure.
2. MSR Action3D loader and validation.
3. Per-sequence SVD.
4. Exact canonical angles.
5. 1-NN classification.
6. Raw DTW baseline.
7. Accuracy vs rank figure.
8. Quantum SWAP-test affinity on a subset.
9. Shot ablation.
10. Main results table.
11. Short paper or poster draft.

This is the shortest path from plan to a research result that can be shown.

