# Research Process: Q-SDTW Upgrade

Last updated: 2026-05-24

Source upgrade plan: `Q_SDTW_Massive_Research_Upgrade_Plan.md`

Existing project context:

- `RESEARCH_PLAN_quantum_canonical_angles_HAR.md`
- `RESEARCH_PROCESS_quantum_canonical_angles_HAR.md`
- `RESEARCH_STATUS_quantum_canonical_angles_HAR.md`
- `EXPERIMENT_REPORT_quantum_canonical_angles_HAR.md`

## Purpose

This file turns the Q-SDTW upgrade plan into a detailed implementation process for the current repository.

The upgrade is:

```text
Global sequence subspace
-> local motion subspace sequence
-> DTW over local subspace distances
-> SWAP-estimated local subspace costs
-> shot/noise/generalization analysis
```

The main method should be framed as:

```text
Q-SDTW: Quantum-Estimable Subspace Dynamic Time Warping
```

The central research claim is not that the SWAP test is new, nor that the method gives quantum advantage. The central claim is that skeleton actions can be compared through a temporal alignment of local low-rank motion subspaces, and that the projection-affinity costs used by that alignment are compatible with quantum overlap estimation.

## Current Starting Point

The repository is not starting from zero. The following pieces already exist and should be reused:

| Area | Current state |
|---|---|
| MSR Action3D pipeline | Implemented and validated. 566 usable sequences, 20 classes, 10 subjects, `T x 60` features. |
| Splits | Fixed and seeded MSR cross-subject splits exist under `data/splits/`. |
| Raw DTW | Implemented in `src/distances/dtw.py` and run through `scripts/02_run_dtw_baselines.py`. |
| PCA+DTW | Implemented through the DTW baseline script. |
| Global sequence subspaces | Implemented in `src/features/sequence_subspace.py`. |
| Canonical angles | Implemented in `src/distances/canonical_angles.py` and `src/distances/subspace_distances.py`. |
| Exact subspace sweeps | Implemented in `scripts/03_run_subspace_angles.py`. |
| SWAP-test overlap estimation | Implemented in `src/quantum/state_preparation.py`, `src/quantum/swap_test.py`, and `src/quantum/overlap_estimation.py`. |
| Quantum global affinity | Implemented in `src/distances/quantum_estimated_angles.py` and `scripts/04_run_quantum_angles_sim.py`. |
| Exact-vs-quantum global subset comparison | Implemented in `scripts/04_compare_quantum_exact_subset.py`. |
| Tests | Current suite passes with optional Aer skips. |

Current strongest numbers from the experiment report:

| Method | Best setting | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Raw DTW | window none | 0.843636 | 0.825326 |
| PCA+DTW | `k=32`, window none | 0.847273 | 0.829881 |
| Exact canonical angles | `mean_angle`, `r=2` | 0.737226 | 0.717567 |
| SWAP global affinity subset | `r=2`, 1024 shots | 0.655000 | 0.628460 |

The Q-SDTW process should preserve those results as baselines and build the new local subspace DTW method on top.

## Implementation Rules

1. Keep the existing scripts and result files reproducible.
   - Do not rename the existing `scripts/04_run_quantum_angles_sim.py` or `scripts/04_compare_quantum_exact_subset.py`.
   - Add new Q-SDTW scripts with clear names, even if the upgrade plan's draft numbering overlaps.

2. Exact classical first, quantum second.
   - Implement exact local subspace DTW before SWAP-estimated local subspace DTW.
   - Every quantum result needs an exact counterpart on the same split or subset.

3. Reuse the current project contracts.
   - A skeleton sequence is a `T x D` NumPy array.
   - A subspace basis is a `D x r` NumPy array.
   - A pairwise distance matrix is `N_test x N_train`.
   - 1-NN prediction should continue to use `src/eval/knn.py`.

4. Every experiment must write raw rows and a summary table.
   - Raw rows go under `results/raw/`.
   - Aggregated tables go under `results/tables/`.
   - Figures go under `results/figures/`.

5. Main claims must stay honest.
   - Use "projection affinity", "canonical-angle-inspired", and "quantum-estimated subspace distance".
   - Do not claim exact canonical-angle estimation from SWAP tests alone.
   - Do not claim quantum advantage.

## Script Naming Convention

Because this repo already has two `04_*.py` scripts, use these names for the upgrade:

| New task | Script |
|---|---|
| Formal global projection-affinity baseline | `scripts/05_run_global_subspace_affinity.py` |
| Exact Local Subspace-DTW | `scripts/06_run_local_sdtw.py` |
| SWAP global convergence wrapper | `scripts/07_run_swap_global_convergence.py` |
| Table generation | Keep existing `scripts/08_generate_tables.py` |
| SWAP Local Subspace-DTW | `scripts/09_run_swap_local_sdtw.py` |
| Noise simulation | `scripts/10_run_noise_simulation.py` |
| QAE efficiency experiment | `scripts/11_run_qae_efficiency.py` |
| Figures | `scripts/12_generate_figures.py` |

Keep `scripts/08_generate_tables.py` for table generation unless it becomes too crowded. If it does, split table generation later with a descriptive name, but do not do that before the main Q-SDTW results exist.

## Data Contracts

Use these names and shapes consistently:

```text
Raw sequence:
  X: np.ndarray with shape T x D

Global subspace:
  U_X: np.ndarray with shape D x r

Local window:
  W_i: np.ndarray with shape L_i x D

Local subspace:
  U_i: np.ndarray with shape D x r

Local subspace sequence:
  [U_1, U_2, ..., U_m]

Local cost matrix:
  C: np.ndarray with shape m x n

Pairwise classifier distance matrix:
  distances: np.ndarray with shape N_test x N_train
```

Important rank rule:

If local windows are centered before SVD, a window with `L` frames can support at most `L - 1` nonzero temporal rank. Therefore:

```text
valid local rank r <= min(window_length - 1, feature_dim)
```

The main experiments should skip invalid `(window_length, rank)` settings instead of silently reducing the rank. A separate `--allow-effective-rank` option can be added later for diagnostics, but it should not be used in the main paper tables.

## Phase 1: Formalize Exact Global Projection Affinity

### Goal

Add a named exact global projection-affinity baseline that matches the quantity estimated by the current SWAP global method.

The current exact canonical-angle script reports `chordal`, `projection`, `mean_angle`, and `max_angle`, while the quantum method estimates:

```text
S(U,V) = (1/r) * sum_ij (u_i.T @ v_j)^2
d(U,V) = 1 - S(U,V)
```

This exact projection-affinity baseline should become a first-class classical method.

### Files To Add Or Update

| File | Change |
|---|---|
| `src/distances/quantum_estimated_angles.py` | Keep existing exact affinity functions for now. |
| `src/distances/subspace_affinity.py` | Optional later refactor if the affinity functions need a neutral classical home. |
| `scripts/05_run_global_subspace_affinity.py` | New script for exact global affinity sweeps. |
| `tests/test_quantum_overlap.py` | Already tests exact affinity. Add a smaller neutral test if refactored. |
| `scripts/08_generate_tables.py` | Include global projection affinity in main and Q-SDTW comparison tables. |

### Implementation Steps

1. Reuse `compute_subspaces()` and `stack_bases()` from `src/features/sequence_subspace.py`.
2. Reuse `pairwise_exact_subspace_affinity_distances()` from `src/distances/quantum_estimated_angles.py`.
3. Run 1-NN with `predict_1nn_from_distances()`.
4. Write rows with:

```text
dataset
seed
r
distance = projection_affinity
accuracy
macro_f1
runtime_sec
train_size
test_size
skipped_sequences
git_commit
timestamp
```

5. Write summary rows grouped by:

```text
dataset
r
distance
```

### Command

```bash
python scripts/05_run_global_subspace_affinity.py \
  --dataset msr_action3d \
  --r-values 1 2 3 4 5 6 8 10 12 \
  --seeds 0 1 2 3 4
```

### Outputs

```text
results/raw/global_subspace_affinity.csv
results/tables/global_subspace_affinity_summary.csv
```

### Done When

- The script reproduces the exact affinity counterpart used by `scripts/04_compare_quantum_exact_subset.py`.
- The result table can answer whether projection affinity behaves like, better than, or worse than the existing canonical-angle distances.
- `pytest` still passes.

### Completion Status

Completed on MSR Action3D:

```text
results/raw/global_subspace_affinity.csv
results/tables/global_subspace_affinity_summary.csv
```

The full sweep wrote 45 raw rows. The best global projection-affinity setting is:

```text
rank: 2
affinity_normalization: projection_frobenius
accuracy: 0.726277
macro_f1: 0.708104
runtime_mean: 0.224775
```

## Phase 2: Implement Local Motion Subspace Extraction

### Goal

Convert each full skeleton sequence into a sequence of local low-rank motion subspaces.

This is the main representation upgrade.

### Files To Add

| File | Purpose |
|---|---|
| `src/features/local_subspace.py` | Sliding windows and local subspace sequence extraction. |
| `tests/test_local_subspace.py` | Unit tests for windowing, rank handling, short sequences, and basis validity. |

### Proposed Data Classes

```python
@dataclass(frozen=True)
class LocalSubspaceWindow:
    sequence_id: str
    window_index: int
    start_frame: int
    end_frame: int
    basis: np.ndarray
    singular_values: np.ndarray
    num_frames: int
    feature_dim: int
    rank: int


@dataclass(frozen=True)
class LocalSubspaceSequence:
    sequence_id: str
    windows: list[LocalSubspaceWindow]
    window_length: int
    stride: int
    rank: int
    feature_mode: str
```

### Required Functions

```python
def make_sliding_windows(
    sequence: np.ndarray,
    *,
    window_length: int,
    stride: int,
    short_sequence_mode: str = "single",
) -> list[tuple[int, int, np.ndarray]]:
    ...


def compute_local_subspace_sequence(
    sequence: np.ndarray,
    *,
    sequence_id: str,
    rank: int,
    window_length: int,
    stride: int,
    center_sequence: bool = True,
    feature_standardization: str = "none",
    frame_l2_normalization: bool = False,
    short_sequence_mode: str = "single",
) -> LocalSubspaceSequence:
    ...


def compute_local_subspace_sequences(
    sequences: Sequence[np.ndarray],
    sequence_ids: Sequence[str],
    *,
    rank: int,
    window_length: int,
    stride: int,
    ...
) -> list[LocalSubspaceSequence]:
    ...
```

### Windowing Rules

Use deterministic sliding windows:

```text
start = 0, stride, 2 * stride, ...
end = start + window_length
```

Include only full windows by default. If the sequence has leftover frames at the end, add one final boundary-aligned window when it would otherwise miss more than half a stride of motion.

For short sequences:

```text
if T < window_length:
    use the full sequence as one window
```

Then validate that:

```text
T_window > rank
```

If the short sequence cannot support the requested rank, skip that sequence for the setting and report it in `skipped_sequences`.

### Feature Modes

Start with position features only, because the current repo already uses `T x 60` position sequences.

Then add a small helper later:

| Feature mode | Shape |
|---|---|
| `position` | `T x D` |
| `velocity` | `(T - 1) x D` |
| `position_velocity` | `(T - 1) x 2D` |

Recommended file for this helper:

```text
src/features/motion_features.py
```

Do not block Local Subspace-DTW on velocity features. Implement position-only first.

### Tests

Add tests for:

1. Window count for a normal sequence.
2. Short sequence creates one window.
3. Invalid rank raises a clear error.
4. Each local basis has shape `D x r`.
5. Each local basis is orthonormal.
6. Repeated calls with the same input are deterministic.

### Done When

`compute_local_subspace_sequence()` can convert several MSR sequences into local subspace sequences for settings such as:

```text
window_length = 10
stride = 5
rank = 2
```

and all local subspace tests pass.

## Phase 3: Implement Exact Local Subspace-DTW

### Goal

Implement DTW where each local cost compares two local motion subspaces instead of two raw skeleton frames.

### Files To Add Or Update

| File | Change |
|---|---|
| `src/distances/local_sdtw.py` | Exact Local Subspace-DTW distance functions. |
| `src/distances/dtw.py` | Optional: add `dtw_distance_from_cost_matrix()` to reuse DTW logic. |
| `tests/test_local_sdtw.py` | Unit tests for local costs, DTW behavior, and pairwise distances. |

### Local Cost Metrics

Implement projection affinity first:

```text
overlaps = U.T @ V
affinity = sum(overlaps ** 2) / r
cost = 1 - affinity
```

Then support canonical-angle distances by reusing `subspace_distance()`:

```text
projection_affinity
chordal
mean_angle
max_angle
```

For `projection_affinity`, distance is in `[0, 1]` when the bases are valid and have equal rank.

### Required Functions

```python
def local_subspace_cost(
    basis_x: np.ndarray,
    basis_y: np.ndarray,
    *,
    metric: str = "projection_affinity",
    affinity_normalization: str = "projection_frobenius",
) -> float:
    ...


def local_subspace_cost_matrix(
    sequence_x: LocalSubspaceSequence,
    sequence_y: LocalSubspaceSequence,
    *,
    metric: str = "projection_affinity",
) -> np.ndarray:
    ...


def local_sdtw_distance(
    sequence_x: LocalSubspaceSequence,
    sequence_y: LocalSubspaceSequence,
    *,
    metric: str = "projection_affinity",
    normalize_by_path_length: bool = True,
    window_ratio: WindowRatio = None,
) -> float:
    ...


def pairwise_local_sdtw_distances(
    test_sequences: Sequence[LocalSubspaceSequence],
    train_sequences: Sequence[LocalSubspaceSequence],
    *,
    metric: str = "projection_affinity",
    normalize_by_path_length: bool = True,
    window_ratio: WindowRatio = None,
) -> np.ndarray:
    ...
```

### DTW Over A Precomputed Cost Matrix

The current `dtw_distance()` computes Euclidean frame costs inside the DTW loop. Local-SDTW needs DTW over a precomputed `m x n` cost matrix.

Add either:

```python
def dtw_distance_from_cost_matrix(
    cost_matrix: np.ndarray,
    *,
    normalize_by_path_length: bool = True,
    window_ratio: WindowRatio = None,
) -> float:
    ...
```

or keep it private inside `src/distances/local_sdtw.py`.

Prefer adding it to `src/distances/dtw.py` only if raw DTW and Local-SDTW can share the same window/path normalization tests cleanly.

### Tests

Add tests for:

1. Identical local subspace sequences have near-zero projection-affinity Local-SDTW distance.
2. Distance matrix shape is `N_test x N_train`.
3. Path-length normalization reduces accumulated cost as expected.
4. `window_ratio="none"` and `window_ratio=None` behave the same.
5. The local cost for identical bases is zero.
6. The local cost for orthogonal rank-one bases is one.
7. Invalid metric names raise clear errors.

### Done When

The following small in-memory test works without loading the dataset:

```text
sequence -> local subspaces -> local cost matrix -> DTW distance -> 1-NN prediction
```

and all tests pass.

## Phase 4: Run Exact Local Subspace-DTW On MSR

### Goal

Find the best exact Local Subspace-DTW settings on MSR Action3D.

This is the most important classical upgrade. It should answer:

```text
Does local subspace alignment improve over global subspace comparison?
Does it approach or beat raw DTW/PCA+DTW?
```

### Files To Add

| File | Purpose |
|---|---|
| `scripts/06_run_local_sdtw.py` | Exact Local-SDTW sweep. |
| `configs/experiment_local_sdtw.yaml` | Window, stride, rank, feature, and DTW settings. |

### Main Sweep

Start with a controlled MSR sweep:

```text
window_length: 5, 10, 15, 20
stride: 2, 5, 10
rank: 1, 2, 3, 4, 6
DTW window: none, 0.1, 0.2
distance: projection_affinity
feature_mode: position
seeds: 0, 1, 2, 3, 4
```

Skip invalid settings such as `window_length=5, rank=6`.

After the projection-affinity sweep, run a smaller metric ablation:

```text
distance: projection_affinity, chordal, mean_angle
window_length: top 2 values from projection-affinity sweep
stride: top 1 or 2 values
rank: top 2 values
```

### Output Schema

Raw rows:

```text
dataset
seed
method = local_sdtw_exact
backend
device
dtype
feature_mode
window_length
stride
r
local_distance
dtw_window
normalize_by_path_length
accuracy
macro_f1
runtime_sec
feature_runtime_sec
distance_runtime_sec
train_size
test_size
skipped_sequences
num_train_windows_mean
num_test_windows_mean
git_commit
timestamp
```

Summary rows should group by:

```text
dataset
backend
device
dtype
feature_mode
window_length
stride
r
local_distance
dtw_window
```

### Command

```bash
python scripts/06_run_local_sdtw.py \
  --dataset msr_action3d \
  --window-lengths 5 10 15 20 \
  --strides 2 5 10 \
  --r-values 1 2 3 4 6 \
  --dtw-windows none 0.1 0.2 \
  --distance projection_affinity \
  --feature-mode position \
  --seeds 0 1 2 3 4 \
  --backend torch_cuda \
  --device cuda:0
```

### Outputs

```text
results/raw/local_subspace_dtw_msr_action3d.csv
results/tables/local_subspace_dtw_summary_msr_action3d.csv
results/tables/local_subspace_dtw_best_msr_action3d.csv
```

### Runtime Strategy

Local-SDTW will be slower than global subspace distances but should be much cheaper than quantum Local-SDTW.

Use these optimizations:

1. Compute local subspace sequences once per `(seed, rank, window_length, stride, feature_mode)`.
2. Cache train and test local subspace sequences in memory inside the script.
3. Reuse train/test labels and indices across DTW window settings.
4. Add `--limit-train` and `--limit-test` for smoke tests.
5. Use `--backend torch_cuda` for the main projection-affinity sweep when PyTorch can see CUDA.

The CUDA backend currently targets the projection-affinity Local-SDTW path, which is
the main Phase 4 sweep metric. Canonical-angle local-distance ablations can remain
on the NumPy backend unless they become a runtime bottleneck.

### Smoke Command

```bash
python scripts/06_run_local_sdtw.py \
  --dataset msr_action3d \
  --window-lengths 10 \
  --strides 5 \
  --r-values 2 \
  --dtw-windows none \
  --distance projection_affinity \
  --seeds 0 \
  --limit-train 20 \
  --limit-test 10 \
  --output results/raw/local_subspace_dtw_smoke.csv \
  --summary-output results/tables/local_subspace_dtw_smoke_summary.csv
```

### Done When

- A smoke run completes quickly.
- The full MSR exact Local-SDTW sweep completes.
- The summary table identifies top exact settings.
- The best Local-SDTW result is compared against Raw DTW, PCA+DTW, global projection affinity, and exact canonical angles.

### Completion Status

Completed on MSR Action3D with CUDA projection-affinity Local-SDTW:

```text
results/raw/local_subspace_dtw_msr_action3d.csv
results/tables/local_subspace_dtw_summary_msr_action3d.csv
```

The full sweep wrote 855 raw rows. The current best exact Local-SDTW setting is:

```text
feature_mode: position
window_length: 5
stride: 2
rank: 1
local_distance: projection_affinity
dtw_window: 0.1
backend: torch_cuda
device: cuda:0
dtype: float32
accuracy: 0.847273
macro_f1: 0.833853
```

## Phase 5: Generate Local-SDTW Tables And Figures

### Goal

Turn the exact Local-SDTW sweep into paper-ready analysis.

### Files To Add Or Update

| File | Change |
|---|---|
| `scripts/08_generate_tables.py` | Add Q-SDTW table modes. |
| `scripts/12_generate_figures.py` | Add local heatmaps and rank plots. |
| `results/tables/q_sdtw_main_results.csv` | New main Q-SDTW table. |

### Required Tables

```text
results/tables/q_sdtw_main_results.csv
results/tables/local_subspace_dtw_best_msr_action3d.csv
results/tables/global_vs_local_subspace_msr_action3d.csv
```

The main Q-SDTW table should include:

```text
Raw DTW
PCA+DTW
Exact canonical angles
Global projection affinity
Local Subspace-DTW
SWAP global affinity
SWAP Local-SDTW, once available
```

### Required Figures

```text
results/figures/local_sdtw_accuracy_heatmap_msr_action3d.png
results/figures/local_sdtw_runtime_heatmap_msr_action3d.png
results/figures/global_vs_local_accuracy_vs_rank_msr_action3d.png
```

### Figure Details

Heatmap 1:

```text
x-axis: window_length
y-axis: rank
cell: mean accuracy
facet or separate file: stride
```

Heatmap 2:

```text
x-axis: window_length
y-axis: stride
cell: mean runtime
facet or separate file: rank
```

Rank plot:

```text
x-axis: rank
y-axis: accuracy
lines: global projection affinity, exact canonical angles, Local-SDTW
```

### Done When

The best local settings can be selected from generated tables and figures, not by manual inspection only.

### Completion Status

Completed for the first MSR Action3D exact Local-SDTW sweep:

```text
results/tables/local_subspace_dtw_best_msr_action3d.csv
results/tables/global_vs_local_subspace_msr_action3d.csv
results/tables/q_sdtw_main_results.csv
results/figures/local_sdtw_accuracy_heatmap_msr_action3d.png
results/figures/local_sdtw_runtime_heatmap_msr_action3d.png
```

The Q-SDTW main table currently reports:

```text
Raw DTW:                 accuracy 0.843636, macro-F1 0.825326
PCA+DTW:                 accuracy 0.847273, macro-F1 0.829881
Exact canonical angles:  accuracy 0.737226, macro-F1 0.717567
Global projection aff.:  accuracy 0.726277, macro-F1 0.708104
Local Subspace-DTW:      accuracy 0.847273, macro-F1 0.833853
SWAP global affinity:    accuracy 0.655000, macro-F1 0.628460
```

## Phase 6: Formalize SWAP Global Convergence

### Goal

Turn the current global SWAP subset results into a formal convergence experiment with more shots and larger subsets.

This phase strengthens the existing quantum result before adding Local-SDTW quantum costs.

### Files To Add Or Update

| File | Change |
|---|---|
| `scripts/07_run_swap_global_convergence.py` | Wrapper or refactor around current global quantum scripts. |
| `scripts/04_compare_quantum_exact_subset.py` | Reuse or extend for larger subsets and output names. |
| `configs/experiment_quantum.yaml` | Add full shot list and larger subset presets. |
| `scripts/12_generate_figures.py` | Add shot convergence figures. |

### Parameters

```text
r: 2, 3, 4
shots: 128, 256, 512, 1024, 2048, 4096
subset small: train_per_class=3, test_per_class=2
subset medium: train_per_class=5, test_per_class=3
seeds: 0, 1, 2, 3, 4
simulator: sampling
```

### Command

```bash
python scripts/07_run_swap_global_convergence.py \
  --dataset msr_action3d \
  --r-values 2 3 4 \
  --shots 128 256 512 1024 2048 4096 \
  --subset true \
  --train-per-class 5 \
  --test-per-class 3 \
  --seeds 0 1 2 3 4
```

### Outputs

```text
results/raw/swap_global_exact_comparison_msr_action3d.csv
results/tables/swap_global_exact_comparison_summary_msr_action3d.csv
results/figures/swap_global_mae_vs_shots_msr_action3d.png
results/figures/swap_global_accuracy_vs_shots_msr_action3d.png
results/figures/swap_global_prediction_agreement_msr_action3d.png
```

### Done When

- Distance MAE decreases as shots increase, or any failure to decrease is investigated and documented.
- Accuracy, prediction agreement, and nearest-neighbor agreement are summarized.
- The current global SWAP result is reproducible through the new formal output names.

### Implementation Status

Phase 6 tooling is implemented:

```text
scripts/07_run_swap_global_convergence.py
scripts/12_generate_figures.py --figure swap_global_convergence
```

A tiny smoke run completed on 2026-05-26 and wrote temporary `/tmp` CSV/PNG
artifacts. The formal small-subset sweep (`train_per_class=3`,
`test_per_class=2`) also completed through 4096 shots and wrote:

```text
results/raw/swap_global_exact_comparison_msr_action3d.csv
results/tables/swap_global_exact_comparison_summary_msr_action3d.csv
results/figures/swap_global_mae_vs_shots_msr_action3d.png
results/figures/swap_global_accuracy_vs_shots_msr_action3d.png
results/figures/swap_global_prediction_agreement_msr_action3d.png
```

The medium-subset sweep remains optional/pending.

## Phase 7: Implement SWAP-Estimated Local Subspace-DTW

### Goal

Estimate each local subspace cost using SWAP-test squared-overlap estimates, then run DTW over the quantum-estimated local cost matrix.

This is the main quantum upgrade.

### Files To Add

| File | Purpose |
|---|---|
| `src/distances/quantum_local_sdtw.py` | Quantum local cost matrix and Q-SDTW distance functions. |
| `scripts/09_run_swap_local_sdtw.py` | Exact-vs-quantum Local-SDTW subset experiment. |
| `tests/test_quantum_local_sdtw.py` | Unit tests for quantum local costs and convergence checks. |

### Required Functions

```python
def quantum_local_subspace_cost(
    basis_x: np.ndarray,
    basis_y: np.ndarray,
    *,
    estimator: SwapTestOverlapEstimator,
    affinity_normalization: str = "projection_frobenius",
) -> float:
    ...


def quantum_local_subspace_cost_matrix(
    sequence_x: LocalSubspaceSequence,
    sequence_y: LocalSubspaceSequence,
    *,
    estimator: SwapTestOverlapEstimator,
    affinity_normalization: str = "projection_frobenius",
) -> np.ndarray:
    ...


def quantum_local_sdtw_distance(
    sequence_x: LocalSubspaceSequence,
    sequence_y: LocalSubspaceSequence,
    *,
    estimator: SwapTestOverlapEstimator,
    normalize_by_path_length: bool = True,
    window_ratio: WindowRatio = None,
) -> float:
    ...
```

### Exact-vs-Quantum Comparison

For each test/train sequence pair:

1. Build exact local cost matrix.
2. Build quantum-estimated local cost matrix.
3. Compute exact Local-SDTW distance.
4. Compute quantum Local-SDTW distance.
5. Store:

```text
distance_abs_error
cost_matrix_mae
cost_matrix_rmse
dtw_distance_exact
dtw_distance_quantum
```

At the classifier level, store:

```text
exact_accuracy
quantum_accuracy
accuracy_gap_quantum_minus_exact
exact_macro_f1
quantum_macro_f1
macro_f1_gap_quantum_minus_exact
nearest_neighbor_agreement
prediction_agreement
```

### Quantum Runtime Control

Do not run the full Local-SDTW quantum grid first.

Use this staged order:

1. Tiny smoke: 2 to 5 classes, `train_per_class=1`, `test_per_class=1`.
2. Small subset: all classes, `train_per_class=3`, `test_per_class=2`.
3. Medium subset: all classes, `train_per_class=5`, `test_per_class=3`.
4. Only then consider full split, and only for one best setting.

### First Main Settings

Use the best exact Local-SDTW settings from Phase 4:

```text
window_length: 5
stride: 2
rank: 1
shots: 512, 1024, 2048
DTW window: 0.1
feature_mode: position
subset: true
train_per_class: 3
test_per_class: 2
seeds: 0, 1, 2
```

### Command

```bash
python scripts/09_run_swap_local_sdtw.py \
  --dataset msr_action3d \
  --window-length 5 \
  --stride 2 \
  --rank 1 \
  --shots 512 1024 2048 \
  --dtw-window 0.1 \
  --subset true \
  --train-per-class 3 \
  --test-per-class 2 \
  --seeds 0 1 2
```

### Outputs

```text
results/raw/swap_local_sdtw_comparison_msr_action3d.csv
results/tables/swap_local_sdtw_comparison_summary_msr_action3d.csv
results/figures/swap_local_sdtw_mae_vs_shots_msr_action3d.png
results/figures/swap_local_sdtw_accuracy_vs_shots_msr_action3d.png
results/figures/swap_local_sdtw_prediction_agreement_msr_action3d.png
```

### Diagnostic Figures

For a small number of representative sequence pairs, save:

```text
results/figures/local_cost_matrix_exact_msr_action3d.png
results/figures/local_cost_matrix_quantum_msr_action3d.png
results/figures/local_cost_matrix_error_msr_action3d.png
results/figures/local_dtw_path_exact_vs_quantum_msr_action3d.png
```

### Caching

The current `SwapTestOverlapEstimator` already caches vector-pair estimates in memory. For Local-SDTW, add experiment-level cache tracking:

```text
num_overlap_estimates
num_cache_hits
num_local_costs
num_dtw_pairs
```

Do not add disk caching until the in-memory implementation is correct. If disk caching becomes necessary, use a deterministic key based on:

```text
dataset
sequence_id_x
window_index_x
sequence_id_y
window_index_y
rank
shots
simulator
seed
affinity_normalization
```

### Tests

Add tests for:

1. Quantum local cost with `simulator="exact"` matches exact projection-affinity cost.
2. Identical local subspace sequences have near-zero Q-SDTW distance in exact simulator mode.
3. Sampling estimates are bounded in `[0, 1]` for costs.
4. Increasing shots reduces error on a fixed toy example, allowing a loose tolerance.
5. Pairwise quantum Local-SDTW distance matrix has correct shape.

### Done When

- Tiny smoke test works.
- Small subset exact-vs-quantum Local-SDTW comparison writes raw and summary CSV files.
- At least one shot-convergence figure exists for Local-SDTW.
- The paper can compare exact Local-SDTW and SWAP-estimated Local-SDTW on the same subset.

### Implementation Status

Phase 7 core tooling is implemented:

```text
src/distances/quantum_local_sdtw.py
scripts/09_run_swap_local_sdtw.py
tests/test_quantum_local_sdtw.py
scripts/12_generate_figures.py --figure swap_local_sdtw_convergence
```

A three-class smoke run completed on 2026-05-26 and wrote:

```text
results/raw/swap_local_sdtw_smoke.csv
results/tables/swap_local_sdtw_smoke_summary.csv
results/raw/swap_local_sdtw_pair_diagnostics_smoke.csv
```

The first all-class subset run completed on 2026-05-26 for `train_per_class=3`,
`test_per_class=2`, seeds `0 1 2`, and shots `512 1024 2048`. It wrote:

```text
results/raw/swap_local_sdtw_comparison_msr_action3d.csv
results/tables/swap_local_sdtw_comparison_summary_msr_action3d.csv
results/raw/swap_local_sdtw_pair_diagnostics_msr_action3d.csv
results/figures/swap_local_sdtw_mae_vs_shots_msr_action3d.png
results/figures/swap_local_sdtw_accuracy_vs_shots_msr_action3d.png
results/figures/swap_local_sdtw_prediction_agreement_msr_action3d.png
```

The best current SWAP Local-SDTW subset row is:

```text
shots: 2048
accuracy: 0.750000
macro_f1: 0.725556
mean_abs_distance_error: 0.004893
mean_cost_matrix_mae: 0.014821
prediction_agreement: 0.958333
```

The full MSR split SWAP Local-SDTW run completed on 2026-05-27 for seeds
`0 1 2` and shots `512 1024 2048`. It wrote:

```text
results/raw/swap_local_sdtw_full_msr_action3d.csv
results/tables/swap_local_sdtw_full_summary_msr_action3d.csv
```

The best current full-split SWAP Local-SDTW classification row is:

```text
shots: 512
accuracy: 0.852121
macro_f1: 0.838612
```

The cleanest current full-split convergence row is:

```text
shots: 2048
accuracy: 0.850909
macro_f1: 0.837930
mean_abs_distance_error: 0.004975
mean_cost_matrix_mae: 0.014800
prediction_agreement: 0.974546
```

## Phase 8: Add UTKinect Support

### Goal

Make Q-SDTW a multi-dataset project rather than an MSR-only project.

UTKinect is the mandatory second dataset because it is small and suitable for quantum subset experiments.

### Files To Add Or Update

| File | Change |
|---|---|
| `src/data/utkinect_loader.py` | Load UTKinect skeleton files. |
| `configs/utkinect.yaml` | Fill in raw path, processed path, skeleton format, split protocol. |
| `scripts/00_prepare_dataset.py` | Add `--dataset utkinect` support. |
| `scripts/01_validate_dataset.py` | Add dataset config routing if needed. |
| `src/data/splits.py` | Add UTKinect cross-subject split helper. |
| `tests/test_utkinect_dataset_pipeline.py` | Loader and split tests using small fixtures. |

### Split

Use the same cross-subject protocol as MSR:

```text
Train subjects: 1, 3, 5, 7, 9
Test subjects:  2, 4, 6, 8, 10
```

Optionally add leave-one-subject-out later.

### Commands

```bash
python scripts/00_prepare_dataset.py --dataset utkinect
python scripts/01_validate_dataset.py --dataset utkinect
```

Then run:

```bash
python scripts/02_run_dtw_baselines.py --dataset utkinect --seeds 0 1 2 3 4
python scripts/05_run_global_subspace_affinity.py --dataset utkinect --seeds 0 1 2 3 4
python scripts/06_run_local_sdtw.py --dataset utkinect --seeds 0 1 2 3 4
python scripts/07_run_swap_global_convergence.py --dataset utkinect --subset true --seeds 0 1 2 3 4
```

### Outputs

```text
data/processed/utkinect/
data/splits/utkinect_cross_subject_seed*.json
results/raw/*_utkinect.csv
results/tables/*_utkinect.csv
```

### Done When

- UTKinect appears in `results/tables/dataset_summary.csv`.
- Raw DTW, PCA+DTW, global affinity, and exact Local-SDTW have UTKinect rows.
- At least one SWAP global subset comparison is run on UTKinect.

## Phase 9: Add UTD-MHAD Support

### Goal

Add a stronger third benchmark after MSR and UTKinect are stable.

### Files To Add Or Update

| File | Change |
|---|---|
| `src/data/utd_mhad_loader.py` | Load UTD-MHAD skeleton files. |
| `configs/utd_mhad.yaml` | Dataset paths and preprocessing settings. |
| `scripts/00_prepare_dataset.py` | Add `--dataset utd_mhad`. |
| `tests/test_utd_mhad_dataset_pipeline.py` | Loader fixture tests. |

### Split

Use one clear cross-subject split and document it:

```text
Train subjects: 1, 3, 5, 7
Test subjects:  2, 4, 6, 8
```

or:

```text
Train subjects: 1, 2, 3, 4
Test subjects:  5, 6, 7, 8
```

Choose one before running final experiments. Do not report both as if they are the same protocol.

### Required Experiments

```text
Raw DTW
PCA+DTW
Global projection affinity
Exact Local-SDTW
SWAP global subset
Optional SWAP Local-SDTW subset
```

### Done When

`results/tables/cross_dataset_main_results.csv` includes MSR, UTKinect, and UTD-MHAD.

## Phase 10: Noise And Aer Validation

### Goal

Make the quantum analysis hardware-aware without claiming current hardware advantage.

### Files To Add Or Update

| File | Change |
|---|---|
| `scripts/10_run_noise_simulation.py` | Noise sweeps for global and local affinity. |
| `src/quantum/swap_test.py` | Reuse Aer backend and add clean noise-model entry points if needed. |
| `configs/experiment_quantum.yaml` | Add noise levels and Aer validation presets. |
| `tests/test_quantum_overlap.py` | Keep Aer tests optional. |

### Noise Types

Start with:

```text
shot noise
readout noise
depolarizing noise
```

Add amplitude damping only after the first noise table exists.

### Tiny Aer Validation

Use:

```text
classes: 2 to 5
train_per_class: 1
test_per_class: 1
rank: 2
shots: 1024
```

Compare:

```text
exact
sampling
qiskit_aer_statevector
qiskit_aer_qasm
noisy_qasm
```

### Outputs

```text
results/raw/aer_validation.csv
results/tables/aer_validation_summary.csv
results/raw/noise_simulation_msr_action3d.csv
results/tables/noise_simulation_summary_msr_action3d.csv
results/figures/noise_vs_distance_error_msr_action3d.png
results/figures/noise_vs_accuracy_msr_action3d.png
```

### Done When

- Ideal Aer agrees with the fast sampling backend on a tiny setup.
- Noisy results produce distance-error and accuracy curves.
- The paper can discuss noise sensitivity as characterization, not advantage.

## Phase 11: Optional QAE Measurement-Efficiency Experiment

### Goal

Add a stronger quantum-algorithmic angle by comparing ordinary SWAP sampling with QAE-style overlap estimation.

Do this only after exact Local-SDTW and SWAP Local-SDTW subset results exist.

### Scope

Run QAE on pairwise overlaps first, not full classification.

Sample vector pairs from:

```text
global subspace bases
local subspace bases
```

Compare error against:

```text
number of quantum queries
```

### Files To Add

| File | Purpose |
|---|---|
| `src/quantum/qae_overlap.py` | QAE-style overlap estimator. |
| `scripts/11_run_qae_efficiency.py` | Query-efficiency experiment. |
| `tests/test_qae_overlap.py` | Small estimator tests or mocked checks. |

### Outputs

```text
results/raw/qae_overlap_efficiency.csv
results/tables/qae_overlap_efficiency_summary.csv
results/figures/qae_vs_sampling_error.png
```

### Done When

The paper can include a small query-efficiency plot. Full QAE classification is not required for the main paper.

## Phase 12: Statistical Reporting And Reproducibility

### Goal

Make every reported number traceable to a generated result file.

### Files To Add Or Update

| File | Change |
|---|---|
| `src/eval/statistics.py` | Mean, std, paired tests, confidence intervals if needed. |
| `scripts/08_generate_tables.py` | Add cross-dataset and Q-SDTW result tables. |
| `scripts/12_generate_figures.py` | Generate all paper figures. |
| `scripts/reproduce_q_sdtw.sh` | Reproduce core Q-SDTW pipeline. |
| `README.md` | Add Q-SDTW command section after results stabilize. |

### Required Final Tables

```text
results/tables/dataset_summary.csv
results/tables/q_sdtw_main_results.csv
results/tables/cross_dataset_main_results.csv
results/tables/global_subspace_rank_summary_msr_action3d.csv
results/tables/local_subspace_dtw_best_msr_action3d.csv
results/tables/swap_global_exact_comparison_summary_msr_action3d.csv
results/tables/swap_local_sdtw_comparison_summary_msr_action3d.csv
```

### Required Final Figures

```text
results/figures/q_sdtw_framework.png
results/figures/global_vs_local_subspace.png
results/figures/global_vs_local_accuracy_vs_rank_msr_action3d.png
results/figures/local_sdtw_accuracy_heatmap_msr_action3d.png
results/figures/swap_global_mae_vs_shots_msr_action3d.png
results/figures/swap_local_sdtw_mae_vs_shots_msr_action3d.png
results/figures/local_cost_matrix_exact_msr_action3d.png
results/figures/local_cost_matrix_quantum_msr_action3d.png
results/figures/local_dtw_path_exact_vs_quantum_msr_action3d.png
results/figures/cross_dataset_results.png
```

### Reproduction Script

Create:

```text
scripts/reproduce_q_sdtw.sh
```

Minimum content:

```bash
#!/usr/bin/env bash
set -euo pipefail

python scripts/01_validate_dataset.py --dataset msr_action3d
python scripts/02_run_dtw_baselines.py --dataset msr_action3d --methods raw_dtw pca_dtw
python scripts/05_run_global_subspace_affinity.py --dataset msr_action3d
python scripts/06_run_local_sdtw.py --dataset msr_action3d
python scripts/07_run_swap_global_convergence.py --dataset msr_action3d --subset true
python scripts/09_run_swap_local_sdtw.py --dataset msr_action3d --subset true
python scripts/08_generate_tables.py --table all
python scripts/12_generate_figures.py --all
```

Keep this script conservative. It should reproduce the core paper results, not every stretch experiment.

### Done When

- Every paper table and figure is generated by a script.
- Every result row contains seed, parameters, runtime, git commit, and timestamp where practical.
- A clean checkout can rerun the core pipeline after datasets are placed in `data/raw/`.

## Priority Order

Follow this order unless a blocker appears:

1. Exact global projection-affinity baseline.
2. Local subspace extraction module and tests.
3. Exact Local-SDTW distance module and tests.
4. MSR exact Local-SDTW smoke run.
5. MSR exact Local-SDTW full sweep.
6. Local-SDTW heatmaps and main Q-SDTW table.
7. Formal SWAP global convergence with larger shots/subsets.
8. SWAP Local-SDTW tiny smoke.
9. SWAP Local-SDTW MSR subset.
10. UTKinect loader and classical comparison.
11. UTKinect SWAP global subset.
12. UTD-MHAD support.
13. Noise and Aer validation.
14. QAE efficiency.
15. Paper/reproduction package.

## Minimum Publishable Version

A strong minimum version should include:

```text
MSR Action3D
UTKinect
Raw DTW
PCA+DTW
Exact canonical angles
Global projection affinity
Exact Local Subspace-DTW
SWAP global affinity convergence
SWAP Local-SDTW subset
Shot convergence figures
Generated result tables
Clear limitations
```

## Strong Full-Paper Version

The stronger version should include:

```text
MSR Action3D
UTKinect
UTD-MHAD
Raw DTW
PCA+DTW
Global projection affinity
Exact canonical angles
Exact Local Subspace-DTW
SWAP global affinity
SWAP Local-SDTW
Noise simulation
Aer validation
QAE overlap-efficiency experiment
Cross-dataset result table
Full reproduction script
```

## Risk Management

### Risk 1: Local-SDTW does not beat raw DTW

This is still usable if Local-SDTW improves over global subspaces and is much more compact than raw frame DTW.

Report:

```text
Raw DTW is the strongest temporal baseline.
Global subspaces are fastest but lose temporal order.
Local-SDTW recovers temporal structure while preserving quantum-estimable subspace costs.
```

### Risk 2: Local-SDTW is too slow

Reduce the sweep:

```text
window_length: 10, 15
stride: 5
rank: 2, 3
DTW window: none
```

Add caching and only broaden the grid after the smoke and focused sweep work.

### Risk 3: Quantum Local-SDTW is too expensive

Use subsets and selected settings only.

The paper does not need full quantum Local-SDTW on every dataset. It needs:

```text
exact Local-SDTW full classical evidence
SWAP Local-SDTW subset convergence evidence
```

### Risk 4: SWAP noise disrupts nearest-neighbor decisions

Focus on convergence and error characterization:

```text
distance MAE
cost matrix MAE
prediction agreement
nearest-neighbor agreement
```

Do not oversell noisy accuracy.

### Risk 5: Dataset loaders take too long

Prioritize UTKinect after MSR. Add UTD-MHAD only when UTKinect is processed and validated.

## Immediate Next Implementation Checklist

Use this checklist for the next coding session:

1. Add `src/features/local_subspace.py`.
2. Add `tests/test_local_subspace.py`.
3. Add `src/distances/local_sdtw.py`.
4. Add `tests/test_local_sdtw.py`.
5. Add `scripts/05_run_global_subspace_affinity.py`.
6. Add `scripts/06_run_local_sdtw.py`.
7. Run a Local-SDTW smoke test on MSR.
8. Extend `scripts/08_generate_tables.py` with Local-SDTW summary rows.
9. Add `scripts/12_generate_figures.py` with the first heatmap.
10. Only then implement `scripts/09_run_swap_local_sdtw.py`.

## Definition Of Done For The Upgrade

The Q-SDTW upgrade is ready to show when:

1. `results/tables/q_sdtw_main_results.csv` compares Raw DTW, PCA+DTW, global subspace methods, exact Local-SDTW, SWAP global, and SWAP Local-SDTW subset.
2. `results/figures/local_sdtw_accuracy_heatmap_msr_action3d.png` shows the local parameter sweep.
3. `results/figures/swap_local_sdtw_mae_vs_shots_msr_action3d.png` shows quantum convergence for local costs.
4. UTKinect has at least dataset validation, Raw DTW, PCA+DTW, global projection affinity, and exact Local-SDTW rows.
5. All new tests pass.
6. The paper framing is "quantum-compatible local motion-subspace alignment", not "quantum advantage".
