# Q-SDTW for Skeleton-Based Action Recognition

This repository supports the original canonical-angle research plan in
`RESEARCH_PLAN_quantum_canonical_angles_HAR.md` and the Q-SDTW upgrade plan in
`Q_SDTW_RESEARCH_PROCESS.md`.

The project investigates skeleton action recognition by comparing low-rank
motion subspaces. The current upgrade method is Q-SDTW:

```text
local skeleton windows -> local motion subspaces -> DTW over projection-affinity costs
```

The quantum-facing claim is that the local projection-affinity costs used by
Local Subspace-DTW are compatible with SWAP-test overlap estimation. The project
does not claim quantum advantage.

## Research Snapshot

As of 2026-05-27, the main MSR Action3D Q-SDTW pipeline is complete. The
classical method, the quantum-estimated global baseline, and the full-split
SWAP-estimated Local-SDTW experiment have all been implemented and run.

The strongest result is the full-split SWAP Local-SDTW experiment:

```text
SWAP Local-SDTW, full MSR split
feature=position, window_length=5, stride=2, rank=1, dtw_window=0.1
train=291, test=275, seeds=0 1 2

best classification row: shots=512
accuracy=0.852121, macro-F1=0.838612

best convergence row: shots=2048
accuracy=0.850909, macro-F1=0.837930
distance MAE=0.004975, prediction agreement=0.974546
```

This is stronger than the original global subspace methods and competitive with,
or slightly better than, the strongest Raw/PCA DTW baselines on MSR Action3D.

## Method

Q-SDTW compares skeleton actions through temporal alignment of local low-rank
motion subspaces.

The exact classical method is:

```text
1. Convert each skeleton sequence to a T x 60 matrix.
2. Split each sequence into overlapping local temporal windows.
3. Compute a rank-r SVD subspace basis for each local window.
4. Compare two local subspaces with projection-affinity distance.
5. Align two sequences of local subspaces with DTW.
6. Classify by 1-nearest neighbor over Local-SDTW distances.
```

For two local subspace bases `U` and `V` with rank `r`, the main local cost is:

```text
affinity(U, V) = ||U^T V||_F^2 / r
cost(U, V) = 1 - affinity(U, V)
```

The SWAP-estimated version replaces each exact squared overlap inside
`||U^T V||_F^2` with a SWAP-test squared-overlap estimate. DTW and 1-NN
classification are then run on the resulting quantum-estimated local cost
matrices.

This gives the main research claim:

```text
Local projection-affinity costs for skeleton action alignment can be estimated
with SWAP-test overlap estimation, and the induced Local-SDTW distances converge
toward the exact Local-SDTW distances as shots increase.
```

The implementation intentionally makes no quantum-advantage claim. The quantum
experiments are shot-noise simulations of a quantum-estimable distance.

## Completed Experiments

The completed MSR Action3D experiment set includes:

| Experiment | Status | Main output |
|---|---|---|
| Dataset preparation and validation | Done | `results/tables/dataset_summary.csv` |
| Raw DTW baseline | Done | `results/tables/dtw_baselines_summary.csv` |
| PCA+DTW baseline | Done | `results/tables/pca_dtw_baselines_summary.csv` |
| Exact canonical-angle subspace baselines | Done | `results/tables/canonical_angles_exact_summary.csv` |
| Exact global projection affinity | Done | `results/tables/global_subspace_affinity_summary.csv` |
| Exact Local Subspace-DTW full sweep | Done | `results/tables/local_subspace_dtw_summary_msr_action3d.csv` |
| SWAP global affinity convergence | Done on subset | `results/tables/swap_global_exact_comparison_summary_msr_action3d.csv` |
| SWAP Local-SDTW smoke and subset checks | Done | `results/tables/swap_local_sdtw_comparison_summary_msr_action3d.csv` |
| SWAP Local-SDTW full split | Done | `results/tables/swap_local_sdtw_full_summary_msr_action3d.csv` |
| Main Q-SDTW comparison table | Done | `results/tables/q_sdtw_main_results.csv` |
| Local-SDTW and SWAP convergence figures | Done | `results/figures/` |

## Main Results

The main comparison table is:

```text
results/tables/q_sdtw_main_results.csv
```

Current headline rows:

| Method | Setting | Accuracy | Macro-F1 | Notes |
|---|---|---:|---:|---|
| Raw DTW | window none | 0.843636 | 0.825326 | Full MSR split |
| PCA+DTW | k=32, window none | 0.847273 | 0.829881 | Full MSR split |
| Exact canonical angles | mean_angle, r=2 | 0.737226 | 0.717567 | Full MSR split |
| Global projection affinity | r=2 | 0.726277 | 0.708104 | Full MSR split |
| Exact Local-SDTW | L=5, stride=2, r=1, window=0.1 | 0.847273 | 0.833853 | Full MSR split |
| SWAP global affinity | r=2, shots=2048 | 0.675000 | 0.645524 | Subset convergence experiment |
| SWAP Local-SDTW | L=5, stride=2, r=1, shots=512 | 0.852121 | 0.838612 | Full MSR split |

The full-split SWAP Local-SDTW shot trend is:

| Shots | Accuracy | Macro-F1 | Distance MAE | Cost MAE | Prediction Agreement |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.852121 | 0.838612 | 0.011314 | 0.028022 | 0.969697 |
| 1024 | 0.846061 | 0.831292 | 0.007448 | 0.020406 | 0.976969 |
| 2048 | 0.850909 | 0.837930 | 0.004975 | 0.014800 | 0.974546 |

The exact-vs-SWAP trend is the important quantum-estimation result: as shots
increase, Local-SDTW distance error and local cost-matrix error decrease. The
classifier remains close to the exact Local-SDTW classifier across all tested
shot counts.

## Conclusion

The MSR Action3D results support the Q-SDTW upgrade.

First, global subspace comparison alone is not strong enough for this dataset:
the best exact global projection-affinity result reaches `0.726277` accuracy,
and exact canonical-angle baselines peak at `0.737226`. This suggests that a
single global low-rank subspace loses important temporal structure.

Second, local subspace alignment fixes that weakness. Exact Local-SDTW reaches
`0.847273` accuracy and `0.833853` macro-F1, matching the best PCA+DTW accuracy
and improving macro-F1. This supports the central classical claim that skeleton
actions benefit from comparing aligned sequences of local motion subspaces.

Third, SWAP-estimated Local-SDTW works on the full MSR split. The full-split
SWAP Local-SDTW run reaches `0.852121` accuracy at 512 shots and retains
`0.850909` accuracy at 2048 shots, while distance MAE drops to `0.004975`.
This supports the quantum-facing claim that the Local-SDTW projection-affinity
costs are compatible with SWAP-test overlap estimation.

The honest interpretation is:

```text
Q-SDTW is a competitive skeleton action recognition method on MSR Action3D, and
its key local subspace costs are quantum-estimable. The current evidence shows
shot convergence and full-split viability, but does not claim quantum speedup.
```

Remaining optional work for a publication-grade package:

- Add a second dataset such as UTKinect.
- Add noise/Aer validation for circuit-level simulation.
- Add statistical reporting and a one-command reproduction script.
- Expand SWAP Local-SDTW to seeds `0 1 2 3 4` if more full-split statistical
  confidence is needed.

## Current Stage

The MSR Action3D pipeline is complete for the original baselines, exact
Q-SDTW, global SWAP convergence, and full-split SWAP Local-SDTW. Dataset
preparation, exact canonical-angle experiments, Raw DTW, PCA+DTW, exact global
projection affinity, exact Local-SDTW, SWAP global subset experiments, and
SWAP-estimated Local-SDTW full-split experiments are complete for MSR Action3D.

Current completed artifacts include:

- MSR skeleton loading from extracted text files or bundled RAR archives.
- Per-sequence preprocessing into `T x 60` matrices.
- Processed arrays and metadata under `data/processed/msr_action3d/`.
- Fixed and seeded cross-subject split files under `data/splits/`.
- Dataset validation and `results/tables/dataset_summary.csv`.
- Exact canonical-angle sweep under `results/raw/canonical_angles_exact.csv`.
- Exact global projection-affinity sweep under `results/raw/global_subspace_affinity.csv`.
- Raw DTW and PCA+DTW baselines under `results/raw/dtw_baselines.csv`.
- First main comparison table under `results/tables/main_results.csv`.
- Local motion subspace extraction in `src/features/local_subspace.py`.
- Exact Local Subspace-DTW in `src/distances/local_sdtw.py`.
- Full MSR Local-SDTW CUDA sweep under `results/raw/local_subspace_dtw_msr_action3d.csv`.
- Local-SDTW best settings and heatmaps under `results/tables/` and `results/figures/`.
- Q-SDTW comparison table under `results/tables/q_sdtw_main_results.csv`.
- SWAP-estimated Local-SDTW implementation, smoke outputs, subset results, and full-split MSR results.

The quantum-estimated subspace-affinity extension is now implemented for the
first SWAP-test version. The formal small-subset global convergence sweep over
ranks `2, 3, 4` and shots `128, 256, 512, 1024, 2048, 4096` is complete.

## Quick Setup Check

From the repository root:

```bash
python scripts/check_setup.py
```

This verifies that project imports work and writes a tiny metadata-rich CSV row to `results/raw/setup_check.csv`.

## Environment

The classical canonical-angle pipeline runs on CPU with NumPy. The Raw/PCA DTW
baselines and projection-affinity Local-SDTW sweep can run on CUDA through
PyTorch.

Basic Python environment:

```bash
pip install -r requirements.txt
```

Recommended GPU environment for the DTW scripts:

```bash
conda create -n qca-gpu python=3.10 -y
conda activate qca-gpu
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

Check CUDA visibility:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda'); print(torch.cuda.get_arch_list() if torch.cuda.is_available() else [])"
```

## MSR Action3D Dataset Setup

The supplied dataset archive is ignored by git. Put the raw MSR files under
`data/raw/MSRAction3D/`:

```bash
unzip -o OneDrive_1_5-20-2026.zip -d data/raw/MSRAction3D
```

The preparation script prefers `MSRAction3DSkeletonReal3D.rar` and falls back to
`MSRAction3DSkeleton(20joints).rar` if needed. It uses the first three columns of
each skeleton row and keeps confidence values only for metadata/validity checks.

```bash
python scripts/00_prepare_dataset.py --dataset msr_action3d
python scripts/01_validate_dataset.py --dataset msr_action3d
```

Current validated summary:

```text
dataset,num_sequences,num_classes,num_subjects,min_frames,max_frames,mean_frames,feature_dim
msr_action3d,566,20,10,9,196,39.514,60
```

One all-zero recording, `a13_s09_e02`, is skipped and recorded in
`data/processed/msr_action3d/metadata.json`.

## Reproducing Current Results

Run these commands from the repository root after preparing the MSR Action3D data.

```bash
python scripts/00_prepare_dataset.py --dataset msr_action3d
python scripts/01_validate_dataset.py --dataset msr_action3d
```

Exact canonical-angle sweep:

```bash
python scripts/03_run_subspace_angles.py \
  --dataset msr_action3d \
  --r-values 1 2 3 4 5 6 8 10 12 \
  --distance chordal projection mean_angle max_angle \
  --seeds 0 1 2 3 4
```

Exact global projection-affinity sweep:

```bash
python scripts/05_run_global_subspace_affinity.py \
  --dataset msr_action3d \
  --r-values 1 2 3 4 5 6 8 10 12 \
  --seeds 0 1 2 3 4
```

Outputs:

- `results/raw/global_subspace_affinity.csv`
- `results/tables/global_subspace_affinity_summary.csv`

Raw DTW baseline on one CUDA device:

```bash
python scripts/02_run_dtw_baselines.py \
  --dataset msr_action3d \
  --methods raw_dtw \
  --seeds 0 1 2 3 4 \
  --window-ratios none 0.1 0.2 \
  --backend torch_cuda \
  --device cuda:0 \
  --torch-dtype float32 \
  --output results/raw/dtw_baselines_raw_only.csv \
  --summary-output results/tables/dtw_baselines_raw_only_summary.csv
```

PCA+DTW baseline can be split across three GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/02_run_dtw_baselines.py --dataset msr_action3d --methods pca_dtw --seeds 0 1 2 3 4 --pca-k-values 4 8 --window-ratios none 0.1 0.2 --backend torch_cuda --device cuda:0 --torch-dtype float32 --output results/raw/pca_dtw_gpu_k_04_08.csv --summary-output results/tables/pca_dtw_gpu_k_04_08_summary.csv
CUDA_VISIBLE_DEVICES=1 python scripts/02_run_dtw_baselines.py --dataset msr_action3d --methods pca_dtw --seeds 0 1 2 3 4 --pca-k-values 12 16 --window-ratios none 0.1 0.2 --backend torch_cuda --device cuda:0 --torch-dtype float32 --output results/raw/pca_dtw_gpu_k_12_16.csv --summary-output results/tables/pca_dtw_gpu_k_12_16_summary.csv
CUDA_VISIBLE_DEVICES=2 python scripts/02_run_dtw_baselines.py --dataset msr_action3d --methods pca_dtw --seeds 0 1 2 3 4 --pca-k-values 24 32 --window-ratios none 0.1 0.2 --backend torch_cuda --device cuda:0 --torch-dtype float32 --output results/raw/pca_dtw_gpu_k_24_32.csv --summary-output results/tables/pca_dtw_gpu_k_24_32_summary.csv
```

Generate aggregate tables:

```bash
python scripts/08_generate_tables.py --table all
```

This writes:

- `results/raw/dtw_baselines.csv`
- `results/raw/pca_dtw_baselines.csv`
- `results/tables/dtw_baselines_summary.csv`
- `results/tables/pca_dtw_baselines_summary.csv`
- `results/tables/main_results.csv`

Exact Local Subspace-DTW CUDA sweep:

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
  --device cuda:0 \
  --torch-dtype float32
```

Outputs:

- `results/raw/local_subspace_dtw_msr_action3d.csv`
- `results/tables/local_subspace_dtw_summary_msr_action3d.csv`

Generate Q-SDTW tables and Local-SDTW heatmaps:

```bash
python scripts/08_generate_tables.py --table local_sdtw
python scripts/08_generate_tables.py \
  --table q_sdtw_main \
  --quantum-global-summary results/tables/swap_global_exact_comparison_summary_msr_action3d.csv \
  --swap-local-summary results/tables/swap_local_sdtw_full_summary_msr_action3d.csv
python scripts/12_generate_figures.py --figure local_sdtw_heatmaps
```

Outputs:

- `results/tables/local_subspace_dtw_best_msr_action3d.csv`
- `results/tables/global_vs_local_subspace_msr_action3d.csv`
- `results/tables/q_sdtw_main_results.csv`
- `results/figures/local_sdtw_accuracy_heatmap_msr_action3d.png`
- `results/figures/local_sdtw_runtime_heatmap_msr_action3d.png`

Quantum-estimated subspace affinity smoke run:

```bash
python scripts/04_run_quantum_angles_sim.py \
  --dataset msr_action3d \
  --r-values 2 \
  --shots 32 \
  --seeds 0 \
  --subset true \
  --train-per-class 1 \
  --test-per-class 1 \
  --limit-train 4 \
  --limit-test 3 \
  --output results/raw/quantum_subspace_affinity_smoke.csv \
  --summary-output results/tables/quantum_subspace_affinity_smoke_summary.csv
```

Phase 4 subset sweep:

```bash
python scripts/04_run_quantum_angles_sim.py \
  --dataset msr_action3d \
  --r-values 2 3 4 \
  --shots 128 256 512 1024 \
  --subset true \
  --seeds 0 1 2 3 4
```

Outputs:

- `results/raw/quantum_subspace_affinity.csv`
- `results/tables/quantum_subspace_affinity_summary.csv`

Exact-vs-quantum subset comparison:

```bash
python scripts/04_compare_quantum_exact_subset.py \
  --dataset msr_action3d \
  --r-values 2 3 4 \
  --shots 128 256 512 1024 \
  --subset true \
  --seeds 0 1 2 3 4
```

Outputs:

- `results/raw/quantum_exact_subset_comparison.csv`
- `results/tables/quantum_exact_subset_comparison_summary.csv`

Formal SWAP global convergence wrapper:

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

Outputs:

- `results/raw/swap_global_exact_comparison_msr_action3d.csv`
- `results/tables/swap_global_exact_comparison_summary_msr_action3d.csv`

Generate SWAP global convergence figures:

```bash
python scripts/12_generate_figures.py --figure swap_global_convergence
```

Outputs:

- `results/figures/swap_global_mae_vs_shots_msr_action3d.png`
- `results/figures/swap_global_accuracy_vs_shots_msr_action3d.png`
- `results/figures/swap_global_prediction_agreement_msr_action3d.png`

SWAP-estimated Local-SDTW smoke run:

```bash
python scripts/09_run_swap_local_sdtw.py \
  --dataset msr_action3d \
  --window-length 5 \
  --stride 2 \
  --rank 1 \
  --shots 128 256 512 \
  --dtw-window 0.1 \
  --subset true \
  --train-per-class 1 \
  --test-per-class 1 \
  --limit-classes 3 \
  --seeds 0 \
  --output results/raw/swap_local_sdtw_smoke.csv \
  --summary-output results/tables/swap_local_sdtw_smoke_summary.csv \
  --pair-output results/raw/swap_local_sdtw_pair_diagnostics_smoke.csv
```

First all-class subset command:

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

Outputs:

- `results/raw/swap_local_sdtw_comparison_msr_action3d.csv`
- `results/tables/swap_local_sdtw_comparison_summary_msr_action3d.csv`
- `results/raw/swap_local_sdtw_pair_diagnostics_msr_action3d.csv`

Full-split SWAP-estimated Local-SDTW command:

```bash
python scripts/09_run_swap_local_sdtw.py \
  --dataset msr_action3d \
  --window-length 5 \
  --stride 2 \
  --rank 1 \
  --shots 512 1024 2048 \
  --dtw-window 0.1 \
  --subset false \
  --seeds 0 1 2 \
  --no-pair-output \
  --output results/raw/swap_local_sdtw_full_msr_action3d.csv \
  --summary-output results/tables/swap_local_sdtw_full_summary_msr_action3d.csv
```

Outputs:

- `results/raw/swap_local_sdtw_full_msr_action3d.csv`
- `results/tables/swap_local_sdtw_full_summary_msr_action3d.csv`

## Script Reference

### `scripts/check_setup.py`

Smoke-checks imports, seeding, paths, logging, and result writing.

```bash
python scripts/check_setup.py
```

Output:

- `results/raw/setup_check.csv`

Arguments: none.

### `scripts/00_prepare_dataset.py`

Loads raw MSR Action3D skeleton files, preprocesses each sequence into a `T x 60`
matrix, saves processed arrays, and creates cross-subject split files.

```bash
python scripts/00_prepare_dataset.py --dataset msr_action3d
```

Important arguments:

| Argument | Default | What it does |
|---|---:|---|
| `--dataset` | required | Dataset name. Currently only `msr_action3d`. |
| `--config` | `configs/msr_action3d.yaml` | Dataset and preprocessing config. |
| `--raw-dir` | config value | Override raw dataset directory. |
| `--processed-dir` | config value | Override processed output directory. |
| `--use-depth-coordinates` | off | Use depth/pixel skeleton archive instead of Real3D coordinates. |

Outputs:

- `data/processed/msr_action3d/sequences.npz`
- `data/processed/msr_action3d/labels.npy`
- `data/processed/msr_action3d/subjects.npy`
- `data/processed/msr_action3d/repetition_ids.npy`
- `data/processed/msr_action3d/sequence_ids.npy`
- `data/processed/msr_action3d/metadata.json`
- `data/splits/msr_cross_subject*.json`

### `scripts/01_validate_dataset.py`

Validates processed arrays and split integrity, then writes the dataset summary table.

```bash
python scripts/01_validate_dataset.py --dataset msr_action3d
```

Important arguments:

| Argument | Default | What it does |
|---|---:|---|
| `--dataset` | required | Dataset name. Currently only `msr_action3d`. |
| `--config` | `configs/msr_action3d.yaml` | Dataset config used for expected feature dimension. |
| `--processed-dir` | config value | Override processed dataset directory. |
| `--split-path` | `data/splits/msr_cross_subject.json` | Split file to validate. |
| `--output` | `results/tables/dataset_summary.csv` | Output summary CSV path. |

Output:

- `results/tables/dataset_summary.csv`

### `scripts/03_run_subspace_angles.py`

Runs exact sequence-as-subspace classification. Each sequence is converted to a
rank-`r` SVD motion subspace, test/train subspaces are compared with canonical-angle
distances, and labels are predicted with 1-nearest neighbor.

```bash
python scripts/03_run_subspace_angles.py \
  --dataset msr_action3d \
  --r-values 1 2 3 4 5 6 8 10 12 \
  --distance chordal projection mean_angle max_angle \
  --seeds 0 1 2 3 4
```

Important arguments:

| Argument | Default | What it does |
|---|---:|---|
| `--dataset` | required | Dataset name. Currently only `msr_action3d`. |
| `--experiment-config` | `configs/experiment_main.yaml` | Rank, distance, seed, and normalization defaults. |
| `--dataset-config` | `configs/msr_action3d.yaml` | Processed dataset location. |
| `--r-values` | config value | Subspace ranks to evaluate. |
| `--distance` | config value | Distances to evaluate: `chordal`, `projection`, `mean_angle`, `max_angle`, `min_angle`. |
| `--seeds` | config value | Split seeds to evaluate. |
| `--feature-standardization` | `none` | Optional per-sequence feature standardization. Use `zscore` for z-scoring. |
| `--l2-normalize-frames` | off | L2-normalize each frame before SVD. |
| `--output` | `results/raw/canonical_angles_exact.csv` | Per-run result CSV. |
| `--summary-output` | `results/tables/canonical_angles_exact_summary.csv` | Mean/std summary CSV. |
| `--plot-output` | `results/figures/accuracy_vs_rank.png` | Accuracy-vs-rank figure. |
| `--no-plot` | off | Skip plot generation. |

Outputs:

- `results/raw/canonical_angles_exact.csv`
- `results/tables/canonical_angles_exact_summary.csv`
- `results/figures/accuracy_vs_rank.png`

### `scripts/02_run_dtw_baselines.py`

Runs DTW 1-nearest-neighbor baselines. It supports raw sequence DTW and PCA+DTW.
PCA is fit only on training frames for each split.

Raw DTW example:

```bash
python scripts/02_run_dtw_baselines.py \
  --dataset msr_action3d \
  --methods raw_dtw \
  --seeds 0 1 2 3 4 \
  --window-ratios none 0.1 0.2 \
  --backend torch_cuda \
  --device cuda:0
```

PCA+DTW example:

```bash
python scripts/02_run_dtw_baselines.py \
  --dataset msr_action3d \
  --methods pca_dtw \
  --seeds 0 1 2 3 4 \
  --pca-k-values 4 8 12 16 24 32 \
  --window-ratios none 0.1 0.2 \
  --backend torch_cuda \
  --device cuda:0
```

Important arguments:

| Argument | Default | What it does |
|---|---:|---|
| `--dataset` | required | Dataset name. Currently only `msr_action3d`. |
| `--experiment-config` | `configs/experiment_main.yaml` | Seeds, DTW windows, and PCA component defaults. |
| `--dataset-config` | `configs/msr_action3d.yaml` | Processed dataset location. |
| `--seeds` | config value | Split seeds to evaluate. |
| `--methods` | `raw_dtw` | One or both of `raw_dtw`, `pca_dtw`. |
| `--window-ratios` | config value | DTW Sakoe-Chiba windows. Use `none` for unconstrained DTW. |
| `--pca-k-values` | config value | PCA dimensions for `pca_dtw`. Ignored for `raw_dtw`. |
| `--no-path-length-normalization` | off | Disable DTW path-length normalization. |
| `--backend` | `auto` | DTW backend: `auto`, `numpy`, `numba`, `torch`, `torch_cpu`, `torch_cuda`. |
| `--device` | auto | Torch device such as `cuda:0`, `cuda:1`, or `cpu`. |
| `--torch-dtype` | `float32` | Torch backend dtype: `float32` or `float64`. |
| `--limit-train` | none | Limit train sequences for smoke tests. |
| `--limit-test` | none | Limit test sequences for smoke tests. |
| `--output` | `results/raw/dtw_baselines.csv` | Per-run result CSV. |
| `--summary-output` | `results/tables/dtw_baselines_summary.csv` | Mean/std summary CSV. |

Backend notes:

- `auto` uses `numba` if available, otherwise NumPy.
- `torch_cuda` uses GPU through PyTorch and is fastest on the current workstation.
- If `torch.cuda.is_available()` is true but PyTorch lacks the GPU architecture, install a newer CUDA-enabled PyTorch build.

Outputs:

- Per-run CSV at `--output`
- Summary CSV at `--summary-output`

### `scripts/06_run_local_sdtw.py`

Runs exact Local Subspace-DTW over local motion subspace sequences. The main
supported CUDA path is projection-affinity Local-SDTW.

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

Important arguments:

| Argument | Default | What it does |
|---|---:|---|
| `--dataset` | required | Dataset name. Currently only `msr_action3d`. |
| `--window-lengths` | config value | Local window lengths. |
| `--strides` | config value | Local window strides. |
| `--r-values` | config value | Local subspace ranks. Invalid `rank >= window_length` settings are skipped. |
| `--distance` | config value | Local subspace distance. CUDA currently targets `projection_affinity`. |
| `--dtw-windows` | config value | Local-SDTW Sakoe-Chiba windows. Use `none` for unconstrained DTW. |
| `--feature-mode` | config value | One of `position`, `velocity`, `position_velocity`. |
| `--backend` | `numpy` | One of `numpy`, `torch`, `torch_cpu`, `torch_cuda`. |
| `--device` | auto | Torch device such as `cuda:0`, `cuda:1`, or `cpu`. |
| `--torch-dtype` | `float32` | Torch backend dtype: `float32` or `float64`. |
| `--limit-train` | none | Limit train sequences for smoke tests. |
| `--limit-test` | none | Limit test sequences for smoke tests. |
| `--output` | `results/raw/local_subspace_dtw_msr_action3d.csv` | Per-run result CSV. |
| `--summary-output` | `results/tables/local_subspace_dtw_summary_msr_action3d.csv` | Mean/std summary CSV. |

Outputs:

- `results/raw/local_subspace_dtw_msr_action3d.csv`
- `results/tables/local_subspace_dtw_summary_msr_action3d.csv`

### `scripts/07_run_swap_global_convergence.py`

Runs the formal exact-vs-SWAP global subspace-affinity convergence experiment
with output names reserved for Phase 6.

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

Outputs:

- `results/raw/swap_global_exact_comparison_msr_action3d.csv`
- `results/tables/swap_global_exact_comparison_summary_msr_action3d.csv`

### `scripts/09_run_swap_local_sdtw.py`

Runs exact-vs-SWAP Local Subspace-DTW using the current best exact Local-SDTW
setting by default: `L=5`, stride `2`, rank `1`, DTW window `0.1`. It supports
both controlled subsets and the full MSR split.

```bash
python scripts/09_run_swap_local_sdtw.py \
  --dataset msr_action3d \
  --shots 512 1024 2048 \
  --subset false \
  --seeds 0 1 2 \
  --no-pair-output \
  --output results/raw/swap_local_sdtw_full_msr_action3d.csv \
  --summary-output results/tables/swap_local_sdtw_full_summary_msr_action3d.csv
```

Outputs:

- `results/raw/swap_local_sdtw_full_msr_action3d.csv`
- `results/tables/swap_local_sdtw_full_summary_msr_action3d.csv`

### `scripts/08_generate_tables.py`

Merges DTW result shards, regenerates DTW summaries, extracts PCA-only tables,
and writes main/Q-SDTW comparison tables.

```bash
python scripts/08_generate_tables.py --table all
```

Important arguments:

| Argument | Default | What it does |
|---|---:|---|
| `--table` | `all` | One of `all`, `dtw_baselines`, `main_results`, `local_sdtw`, `q_sdtw_main`. |

Outputs:

- `results/raw/dtw_baselines.csv`
- `results/raw/pca_dtw_baselines.csv`
- `results/tables/dtw_baselines_summary.csv`
- `results/tables/pca_dtw_baselines_summary.csv`
- `results/tables/main_results.csv`
- `results/tables/local_subspace_dtw_best_msr_action3d.csv`
- `results/tables/global_vs_local_subspace_msr_action3d.csv`
- `results/tables/q_sdtw_main_results.csv`

### `scripts/12_generate_figures.py`

Generates Q-SDTW figures. Implemented modes write Local-SDTW heatmaps and SWAP
global shot-convergence figures.

```bash
python scripts/12_generate_figures.py --figure local_sdtw_heatmaps
python scripts/12_generate_figures.py --figure swap_global_convergence
python scripts/12_generate_figures.py \
  --figure swap_local_sdtw_convergence \
  --swap-local-summary-input results/tables/swap_local_sdtw_full_summary_msr_action3d.csv
```

Outputs:

- `results/figures/local_sdtw_accuracy_heatmap_msr_action3d.png`
- `results/figures/local_sdtw_runtime_heatmap_msr_action3d.png`
- `results/figures/swap_global_mae_vs_shots_msr_action3d.png`
- `results/figures/swap_global_accuracy_vs_shots_msr_action3d.png`
- `results/figures/swap_global_prediction_agreement_msr_action3d.png`
- `results/figures/swap_local_sdtw_mae_vs_shots_msr_action3d.png`
- `results/figures/swap_local_sdtw_accuracy_vs_shots_msr_action3d.png`
- `results/figures/swap_local_sdtw_prediction_agreement_msr_action3d.png`

## Current Headline Results

From `results/tables/q_sdtw_main_results.csv`:

```text
Raw DTW:                 accuracy 0.843636, macro-F1 0.825326
PCA+DTW, k=32:           accuracy 0.847273, macro-F1 0.829881
Exact canonical angles:  accuracy 0.737226, macro-F1 0.717567
Global projection aff.:  accuracy 0.726277, macro-F1 0.708104
Local Subspace-DTW CUDA: accuracy 0.847273, macro-F1 0.833853
Quantum subspace affinity subset, r=2, shots=2048:
                          accuracy 0.675000, macro-F1 0.645524
SWAP Local-SDTW full split, shots=512:
                          accuracy 0.852121, macro-F1 0.838612
SWAP Local-SDTW full split, shots=2048:
                          accuracy 0.850909, macro-F1 0.837930,
                          distance MAE 0.004975, prediction agreement 0.974546
Exact-vs-quantum subset comparison, r=2, shots=2048:
                          exact accuracy 0.655000, quantum accuracy 0.675000,
                          distance MAE 0.015368, prediction agreement 0.915000
Exact-vs-SWAP Local-SDTW full split, shots=2048:
                          exact accuracy 0.847273, quantum accuracy 0.850909,
                          distance MAE 0.004975, prediction agreement 0.974546
```

Best exact Local-SDTW setting:

```text
feature=position, window_length=5, stride=2, rank=1, dtw_window=0.1
backend=torch_cuda, device=cuda:0, dtype=float32
accuracy=0.847273, macro-F1=0.833853
```

## Key Documents

- `RESEARCH_PLAN_quantum_canonical_angles_HAR.md`: original research plan.
- `RESEARCH_PROCESS_quantum_canonical_angles_HAR.md`: step-by-step execution roadmap.
- `RESEARCH_STATUS_quantum_canonical_angles_HAR.md`: living task/status tracker.
- `Q_SDTW_Massive_Research_Upgrade_Plan.md`: Q-SDTW upgrade plan.
- `Q_SDTW_RESEARCH_PROCESS.md`: detailed Q-SDTW implementation process.
- `Q_SDTW_RESEARCH_STATUS.md`: living Q-SDTW task/status tracker.
