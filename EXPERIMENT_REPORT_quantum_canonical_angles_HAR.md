# Experiment Report: Quantum-Estimated Canonical Angles for HAR

Last updated: 2026-05-24

This file documents the experiments run so far, how they were run, what files they produced, and what the current results mean.

The current project studies skeleton-based human action recognition by:

1. Converting each skeleton sequence into a cleaned matrix `X` with shape `T x 60`.
2. Representing each sequence as a low-rank motion subspace using per-sequence SVD.
3. Comparing subspaces with classical canonical-angle distances.
4. Estimating a quantum subspace-affinity distance with SWAP-test squared-overlap estimates.
5. Evaluating all methods with 1-nearest-neighbor classification.

## Environment And Data Assumptions

All commands below are run from the repository root:

```bash
cd /home/ghali/qca-skeleton-action-recognition
```

The MSR Action3D raw data is expected under:

```text
data/raw/MSRAction3D/
```

The current implementation supports MSR Action3D for the completed experiments. UTKinect is still a future/generalization phase.

## Common Evaluation Setup

### Dataset

The processed dataset is MSR Action3D.

Current validated summary:

| Dataset | Sequences | Classes | Subjects | Min Frames | Max Frames | Mean Frames | Feature Dim |
|---|---:|---:|---:|---:|---:|---:|---:|
| msr_action3d | 566 | 20 | 10 | 9 | 196 | 39.514 | 60 |

One all-zero recording, `a13_s09_e02`, is skipped during preprocessing.

### Feature Representation

Each sequence is represented as:

```text
X_i in R^(T_i x 60)
```

The 60 features come from:

```text
20 joints x 3 coordinates = 60
```

Preprocessing:

- Remove invalid/all-zero frames.
- Flatten joints in joint-major order.
- Root-center skeletons.
- Apply scale normalization from `configs/msr_action3d.yaml`.
- Save processed arrays to `data/processed/msr_action3d/`.

### Splits

Experiments use cross-subject split files under:

```text
data/splits/
```

The fixed MSR split is:

```text
Train subjects: 1, 3, 5, 7, 9
Test subjects:  2, 4, 6, 8, 10
```

Seeded split files also exist:

```text
data/splits/msr_cross_subject_seed0.json
data/splits/msr_cross_subject_seed1.json
data/splits/msr_cross_subject_seed2.json
data/splits/msr_cross_subject_seed3.json
data/splits/msr_cross_subject_seed4.json
```

### Classifier And Metrics

All current similarity experiments use 1-nearest-neighbor classification:

```text
prediction = label of closest training sequence
```

Metrics:

- Accuracy
- Macro-F1
- Runtime in seconds

## Experiment 1: Dataset Preparation And Validation

### Purpose

Verify that MSR Action3D is loaded and converted into clean `T x 60` matrices.

### Command

```bash
python scripts/00_prepare_dataset.py --dataset msr_action3d
python scripts/01_validate_dataset.py --dataset msr_action3d
```

### Produced Files

```text
data/processed/msr_action3d/sequences.npz
data/processed/msr_action3d/labels.npy
data/processed/msr_action3d/subjects.npy
data/processed/msr_action3d/repetition_ids.npy
data/processed/msr_action3d/sequence_ids.npy
data/processed/msr_action3d/metadata.json
data/splits/msr_cross_subject*.json
results/tables/dataset_summary.csv
```

### Result

The dataset validation passed and wrote:

```text
results/tables/dataset_summary.csv
```

Summary:

| Dataset | Sequences | Classes | Subjects | Mean Frames | Feature Dim |
|---|---:|---:|---:|---:|---:|
| msr_action3d | 566 | 20 | 10 | 39.514 | 60 |

## Experiment 2: Exact Canonical-Angle Subspace Classification

### Purpose

Test whether skeleton action sequences can be classified by comparing low-rank motion subspaces, without explicit temporal alignment.

### Method

For each sequence `X`:

1. Center the sequence.
2. Compute SVD.
3. Keep the top `r` right singular vectors as a basis:

```text
U_X in R^(60 x r)
```

For two sequences `X` and `Y`, canonical angles are computed from:

```text
M = U_X.T @ U_Y
singular_values(M) = cos(theta_i)
theta_i = arccos(singular_values_i)
```

Distances tested:

- `chordal`
- `projection`
- `mean_angle`
- `max_angle`

Ranks tested:

```text
1, 2, 3, 4, 5, 6, 8, 10, 12
```

### Command

```bash
python scripts/03_run_subspace_angles.py \
  --dataset msr_action3d \
  --r-values 1 2 3 4 5 6 8 10 12 \
  --distance chordal projection mean_angle max_angle \
  --seeds 0 1 2 3 4
```

### Produced Files

```text
results/raw/canonical_angles_exact.csv
results/tables/canonical_angles_exact_summary.csv
results/figures/accuracy_vs_rank.png
```

### Key Results

Best current exact canonical-angle setting:

| Method | Configuration | Accuracy | Macro-F1 | Runtime |
|---|---|---:|---:|---:|
| Exact canonical angles | mean_angle, r=2 | 0.737226 | 0.717567 | 0.276141 |

Best chordal/projection setting:

| Distance | Rank | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| chordal | 2 | 0.726277 | 0.708104 |
| projection | 2 | 0.726277 | 0.708104 |
| mean_angle | 2 | 0.737226 | 0.717567 |

Interpretation:

- Rank 2 is currently the strongest rank for exact subspace classification.
- Performance decreases for higher ranks, suggesting that the leading motion directions carry the useful class signal while higher-rank components add noise or subject-specific variation.
- `mean_angle` is the current best exact distance by accuracy, while `chordal` remains the main geometrically standard distance from the original plan.

## Experiment 3: Raw DTW Baseline

### Purpose

Compare subspace geometry against a standard temporal-alignment baseline.

### Method

Raw DTW compares full skeleton sequences:

```text
X in R^(T x 60)
```

Frame distance:

```text
d(x_t, y_s) = ||x_t - y_s||_2
```

DTW settings:

- Euclidean frame distance.
- Path-length normalization enabled.
- Windows: `none`, `0.1`, `0.2`.
- Backend used in current result table: `torch_cuda`, device `cuda:0`, dtype `float32`.

### Command

```bash
python scripts/02_run_dtw_baselines.py \
  --dataset msr_action3d \
  --methods raw_dtw \
  --seeds 0 1 2 3 4 \
  --window-ratios none 0.1 0.2 \
  --backend torch_cuda \
  --device cuda:0 \
  --torch-dtype float32
```

### Produced Files

```text
results/raw/dtw_baselines.csv
results/tables/dtw_baselines_summary.csv
```

### Key Results

| Method | Window | Accuracy | Macro-F1 | Runtime |
|---|---|---:|---:|---:|
| Raw DTW | none | 0.843636 | 0.825326 | 36.486400 |
| Raw DTW | 0.1 | 0.829091 | 0.810060 | 34.529554 |
| Raw DTW | 0.2 | 0.836364 | 0.817806 | 35.469746 |

Interpretation:

- Raw DTW is currently stronger than exact canonical-angle classification on full MSR Action3D.
- The unconstrained DTW window is best in this result set.
- DTW is much slower than exact subspace distances in the current implementation.

## Experiment 4: PCA+DTW Baseline

### Purpose

Test whether global dimensionality reduction before DTW improves or stabilizes the temporal-alignment baseline.

### Method

For each split:

1. Fit PCA only on training frames.
2. Project train and test frames into `k` dimensions.
3. Run DTW on projected sequences.

PCA dimensions:

```text
4, 8, 12, 16, 24, 32
```

DTW windows:

```text
none, 0.1, 0.2
```

### Commands Used

The PCA+DTW sweep was split across GPU jobs:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/02_run_dtw_baselines.py \
  --dataset msr_action3d \
  --methods pca_dtw \
  --seeds 0 1 2 3 4 \
  --pca-k-values 4 8 \
  --window-ratios none 0.1 0.2 \
  --backend torch_cuda \
  --device cuda:0 \
  --torch-dtype float32 \
  --output results/raw/pca_dtw_gpu_k_04_08.csv \
  --summary-output results/tables/pca_dtw_gpu_k_04_08_summary.csv

CUDA_VISIBLE_DEVICES=1 python scripts/02_run_dtw_baselines.py \
  --dataset msr_action3d \
  --methods pca_dtw \
  --seeds 0 1 2 3 4 \
  --pca-k-values 12 16 \
  --window-ratios none 0.1 0.2 \
  --backend torch_cuda \
  --device cuda:0 \
  --torch-dtype float32 \
  --output results/raw/pca_dtw_gpu_k_12_16.csv \
  --summary-output results/tables/pca_dtw_gpu_k_12_16_summary.csv

CUDA_VISIBLE_DEVICES=2 python scripts/02_run_dtw_baselines.py \
  --dataset msr_action3d \
  --methods pca_dtw \
  --seeds 0 1 2 3 4 \
  --pca-k-values 24 32 \
  --window-ratios none 0.1 0.2 \
  --backend torch_cuda \
  --device cuda:0 \
  --torch-dtype float32 \
  --output results/raw/pca_dtw_gpu_k_24_32.csv \
  --summary-output results/tables/pca_dtw_gpu_k_24_32_summary.csv
```

After the shards were created, aggregate tables were generated with:

```bash
python scripts/08_generate_tables.py --table all
```

### Produced Files

```text
results/raw/pca_dtw_gpu_k_04_08.csv
results/raw/pca_dtw_gpu_k_12_16.csv
results/raw/pca_dtw_gpu_k_24_32.csv
results/raw/pca_dtw_baselines.csv
results/tables/pca_dtw_baselines_summary.csv
results/tables/main_results.csv
```

### Key Results

Best PCA+DTW setting:

| Method | Configuration | Accuracy | Macro-F1 | Runtime |
|---|---|---:|---:|---:|
| PCA+DTW | k=32, window=none | 0.847273 | 0.829881 | 34.982589 |

Other strong PCA+DTW settings:

| k | Window | Accuracy | Macro-F1 |
|---:|---|---:|---:|
| 24 | none | 0.847273 | 0.829593 |
| 32 | none | 0.847273 | 0.829881 |
| 8 | none | 0.843636 | 0.827765 |
| 8 | 0.1 | 0.843636 | 0.829272 |

Interpretation:

- PCA+DTW slightly improves over raw DTW in the current table.
- Higher PCA dimensions, especially `k=24` and `k=32`, perform best.
- PCA+DTW remains much slower than exact subspace distances.

## Experiment 5: Main Classical Comparison Table

### Purpose

Collect the best current classical baselines and exact subspace method into one table.

### Command

```bash
python scripts/08_generate_tables.py --table all
```

### Produced File

```text
results/tables/main_results.csv
```

### Result

| Method | Configuration | Accuracy | Macro-F1 | Runtime |
|---|---|---:|---:|---:|
| Raw DTW | window=none | 0.843636 | 0.825326 | 36.486400 |
| PCA+DTW | k=32, window=none | 0.847273 | 0.829881 | 34.982589 |
| Exact canonical angles | distance=mean_angle, r=2 | 0.737226 | 0.717567 | 0.276141 |

Interpretation:

- The exact subspace method is currently below DTW and PCA+DTW in accuracy.
- The exact subspace method is much faster.
- This motivates the quantum section as a proof of overlap-estimated subspace similarity, not as a claim of state-of-the-art HAR accuracy.

## Experiment 6: SWAP-Test Quantum Subspace Affinity

### Purpose

Implement and evaluate the first quantum version of the subspace method.

This phase does not estimate signed canonical-angle inner products. Instead, it estimates squared overlaps:

```text
|<u_i | v_j>|^2
```

These squared overlaps define a quantum-estimated subspace affinity.

### Method

For two rank-`r` subspace bases:

```text
U_X = [u_1, ..., u_r]
U_Y = [v_1, ..., v_r]
```

The method estimates all `r^2` squared overlaps:

```text
A_ij = |<u_i | v_j>|^2
```

The current default affinity normalization is:

```text
S(X, Y) = sum_ij A_ij / r
d_q(X, Y) = 1 - S(X, Y)
```

This is the projection-Frobenius style normalization. It gives distance 0 for identical subspaces.

### SWAP-Test Estimation

The SWAP test gives:

```text
P(ancilla = 0) = (1 + |<u | v>|^2) / 2
```

Therefore:

```text
|<u | v>|^2 = 2 * P(ancilla = 0) - 1
```

Current backends:

- `sampling`: fast ideal shot-noise simulation using binomial sampling from the ideal SWAP-test probability.
- `qiskit_aer`: optional circuit simulation backend.
- `exact`: deterministic exact overlap mode for tests and reference comparisons.

The Phase 4 subset sweep used the `sampling` backend.

### Subset Protocol

Quantum simulation is expensive because every test/train sequence pair needs `r^2` overlap estimates.

Subset settings:

```text
train_per_class = 3
test_per_class = 2
train_size = 60
test_size = 40
```

Ranks:

```text
2, 3, 4
```

Shots:

```text
128, 256, 512, 1024
```

Seeds:

```text
0, 1, 2, 3, 4
```

Number of overlap estimates per row:

| Rank | Formula | Overlap Estimates |
|---:|---|---:|
| 2 | 40 test x 60 train x 2^2 | 9600 |
| 3 | 40 test x 60 train x 3^2 | 21600 |
| 4 | 40 test x 60 train x 4^2 | 38400 |

### Command

```bash
python scripts/04_run_quantum_angles_sim.py \
  --dataset msr_action3d \
  --r-values 2 3 4 \
  --shots 128 256 512 1024 \
  --subset true \
  --seeds 0 1 2 3 4
```

### Produced Files

```text
results/raw/quantum_subspace_affinity.csv
results/tables/quantum_subspace_affinity_summary.csv
```

### Results

| Rank | Shots | Accuracy | Accuracy Std | Macro-F1 | Macro-F1 Std | Runtime |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 128 | 0.560000 | 0.046368 | 0.534349 | 0.040504 | 1.034413 |
| 2 | 256 | 0.645000 | 0.050990 | 0.616127 | 0.061991 | 1.028900 |
| 2 | 512 | 0.595000 | 0.055678 | 0.572603 | 0.052977 | 1.032277 |
| 2 | 1024 | 0.655000 | 0.065955 | 0.628460 | 0.068528 | 1.026689 |
| 3 | 128 | 0.525000 | 0.070711 | 0.502786 | 0.073173 | 2.207473 |
| 3 | 256 | 0.530000 | 0.078102 | 0.490325 | 0.081221 | 2.200285 |
| 3 | 512 | 0.600000 | 0.050000 | 0.552984 | 0.048699 | 2.202215 |
| 3 | 1024 | 0.650000 | 0.052440 | 0.607635 | 0.055704 | 2.196741 |
| 4 | 128 | 0.445000 | 0.085732 | 0.427080 | 0.091440 | 3.851769 |
| 4 | 256 | 0.480000 | 0.033166 | 0.456809 | 0.038962 | 3.843904 |
| 4 | 512 | 0.540000 | 0.105594 | 0.514660 | 0.122326 | 3.857643 |
| 4 | 1024 | 0.535000 | 0.071764 | 0.499635 | 0.083947 | 3.839882 |

Best setting:

| Rank | Shots | Accuracy | Macro-F1 |
|---:|---:|---:|---:|
| 2 | 1024 | 0.655000 | 0.628460 |

Interpretation:

- Increasing shots generally improves or stabilizes quantum subspace-affinity performance.
- Rank 2 is currently the best quantum subset rank by mean accuracy.
- Rank 3 at 1024 shots is close to rank 2.
- Rank 4 is noisier and weaker on this subset.
- Runtime increases with rank because the number of overlap estimates grows as `r^2`.

## Experiment 7: Exact-vs-Quantum Subset Comparison

### Purpose

Compare the SWAP-test sampled distance against its exact classical counterpart on the same subset.

This is the most important Phase 4.9 check because it answers:

```text
Does quantum-estimated affinity reproduce exact affinity behavior as shots increase?
```

### Exact Reference

The exact reference uses the same squared-overlap affinity formula but computes overlaps classically:

```text
A_ij = (u_i.T @ v_j)^2
S_exact(X, Y) = sum_ij A_ij / r
d_exact(X, Y) = 1 - S_exact(X, Y)
```

The quantum method estimates the same `A_ij` values with SWAP-test sampling.

### Comparison Metrics

For each setting, the script records:

- Exact affinity accuracy.
- Quantum affinity accuracy.
- Accuracy gap: `quantum - exact`.
- Exact macro-F1.
- Quantum macro-F1.
- Mean absolute distance error.
- RMSE distance error.
- Max absolute distance error.
- Nearest-neighbor agreement.
- Prediction agreement.

### Command

```bash
python scripts/04_compare_quantum_exact_subset.py \
  --dataset msr_action3d \
  --r-values 2 3 4 \
  --shots 128 256 512 1024 \
  --subset true \
  --seeds 0 1 2 3 4
```

### Produced Files

```text
results/raw/quantum_exact_subset_comparison.csv
results/tables/quantum_exact_subset_comparison_summary.csv
```

### Results

| Rank | Shots | Exact Acc. | Quantum Acc. | Acc. Gap | Exact F1 | Quantum F1 | MAE | RMSE | NN Agree | Pred. Agree |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 128 | 0.655000 | 0.560000 | -0.095000 | 0.632667 | 0.534349 | 0.059639 | 0.076230 | 0.475000 | 0.655000 |
| 2 | 256 | 0.655000 | 0.645000 | -0.010000 | 0.632667 | 0.616127 | 0.043031 | 0.054822 | 0.575000 | 0.740000 |
| 2 | 512 | 0.655000 | 0.595000 | -0.060000 | 0.632667 | 0.572603 | 0.030226 | 0.038419 | 0.650000 | 0.760000 |
| 2 | 1024 | 0.655000 | 0.655000 | 0.000000 | 0.632667 | 0.628460 | 0.021615 | 0.027388 | 0.750000 | 0.890000 |
| 3 | 128 | 0.665000 | 0.525000 | -0.140000 | 0.618262 | 0.502786 | 0.068393 | 0.085358 | 0.395000 | 0.580000 |
| 3 | 256 | 0.665000 | 0.530000 | -0.135000 | 0.618262 | 0.490325 | 0.047705 | 0.059471 | 0.520000 | 0.665000 |
| 3 | 512 | 0.665000 | 0.600000 | -0.065000 | 0.618262 | 0.552984 | 0.033046 | 0.041537 | 0.570000 | 0.705000 |
| 3 | 1024 | 0.665000 | 0.650000 | -0.015000 | 0.618262 | 0.607635 | 0.023374 | 0.029325 | 0.680000 | 0.770000 |
| 4 | 128 | 0.515000 | 0.445000 | -0.070000 | 0.484236 | 0.427080 | 0.081090 | 0.098473 | 0.390000 | 0.525000 |
| 4 | 256 | 0.515000 | 0.480000 | -0.035000 | 0.484236 | 0.456809 | 0.055324 | 0.067938 | 0.465000 | 0.580000 |
| 4 | 512 | 0.515000 | 0.540000 | 0.025000 | 0.484236 | 0.514660 | 0.037238 | 0.046216 | 0.555000 | 0.670000 |
| 4 | 1024 | 0.515000 | 0.535000 | 0.020000 | 0.484236 | 0.499635 | 0.025570 | 0.031660 | 0.730000 | 0.775000 |

Best comparison setting by quantum accuracy:

| Rank | Shots | Exact Accuracy | Quantum Accuracy | Distance MAE | Prediction Agreement |
|---:|---:|---:|---:|---:|---:|
| 2 | 1024 | 0.655000 | 0.655000 | 0.021615 | 0.890000 |

Convergence pattern:

| Rank | MAE at 128 Shots | MAE at 1024 Shots | Change |
|---:|---:|---:|---:|
| 2 | 0.059639 | 0.021615 | improves by 0.038024 |
| 3 | 0.068393 | 0.023374 | improves by 0.045019 |
| 4 | 0.081090 | 0.025570 | improves by 0.055520 |

Interpretation:

- The exact-vs-quantum comparison supports the intended Phase 4 claim.
- As shots increase, the quantum-estimated distance gets closer to exact affinity distance.
- At `r=2`, `shots=1024`, the quantum method matches exact affinity accuracy on this subset.
- Prediction agreement at `r=2`, `shots=1024` is high: `0.890000`.
- Nearest-neighbor agreement is lower than prediction agreement because a different nearest training sequence can still have the same action label.
- This result should be described as quantum-estimated subspace affinity, not exact canonical-angle estimation.

## Experiment 8: Verification Tests

### Purpose

Check that the implementation is internally consistent.

### Commands

Focused quantum tests:

```bash
pytest tests/test_quantum_overlap.py
```

Full test suite:

```bash
pytest
```

### Result

Latest full verification:

```text
32 passed, 2 skipped
```

The skipped tests are optional Qiskit/Aer checks when that backend is unavailable in the specific `pytest` environment. The `python` environment used for the Aer smoke run did successfully execute a small Qiskit Aer path.

## Result Artifacts Summary

| File | Meaning |
|---|---|
| `results/tables/dataset_summary.csv` | Validated processed dataset summary. |
| `results/raw/canonical_angles_exact.csv` | Per-run exact canonical-angle classification rows. |
| `results/tables/canonical_angles_exact_summary.csv` | Exact canonical-angle mean/std summary. |
| `results/figures/accuracy_vs_rank.png` | Accuracy vs subspace rank figure. |
| `results/raw/dtw_baselines.csv` | Raw DTW and merged DTW baseline rows. |
| `results/tables/dtw_baselines_summary.csv` | DTW baseline summary. |
| `results/raw/pca_dtw_baselines.csv` | PCA+DTW rows. |
| `results/tables/pca_dtw_baselines_summary.csv` | PCA+DTW summary. |
| `results/tables/main_results.csv` | Main classical comparison table. |
| `results/raw/quantum_subspace_affinity.csv` | SWAP-test quantum affinity subset run. |
| `results/tables/quantum_subspace_affinity_summary.csv` | Quantum affinity subset summary. |
| `results/raw/quantum_exact_subset_comparison.csv` | Exact-vs-quantum subset comparison rows. |
| `results/tables/quantum_exact_subset_comparison_summary.csv` | Exact-vs-quantum subset comparison summary. |

## Current Takeaways

1. The MSR Action3D data pipeline is valid and reproducible.
2. Exact sequence-as-subspace classification works and is fast.
3. DTW/PCA+DTW currently outperform exact subspace classification on full MSR Action3D.
4. The SWAP-test quantum subspace-affinity path is implemented and tested.
5. On the subset, quantum affinity approaches exact affinity as shots increase.
6. The strongest quantum subset setting so far is `r=2`, `shots=1024`.
7. The quantum result should be framed as a shot-noise/convergence analysis of subspace affinity, not as a quantum advantage claim.

## Recommended Next Experiments

The next logical phase is Phase 5:

1. Turn the current exact-vs-quantum comparison into formal shot-ablation outputs.
2. Generate an accuracy-vs-shots figure.
3. Generate a distance-error-vs-shots figure.
4. Add a concise method paragraph explaining the SWAP-test subspace-affinity protocol.
5. Decide whether to run a larger subset or full split for the best setting `r=2`, `shots=1024`.

