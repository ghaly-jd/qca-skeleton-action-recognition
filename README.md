# Quantum-Estimated Canonical Angles for Skeleton-Based Action Recognition

This repository supports the research plan in `RESEARCH_PLAN_quantum_canonical_angles_HAR.md`.

The project investigates skeleton action recognition by representing each action sequence as a low-dimensional motion subspace, comparing sequences with canonical angles, and estimating subspace affinity with quantum overlap-estimation circuits.

## Current Stage

The MSR Action3D classical pipeline is now active through Phase 3. Phase 0 setup,
Phase 1 dataset preparation, Phase 2 exact canonical-angle experiments, and the core
Phase 3 DTW baselines are complete for MSR Action3D.

Current completed artifacts include:

- MSR skeleton loading from extracted text files or bundled RAR archives.
- Per-sequence preprocessing into `T x 60` matrices.
- Processed arrays and metadata under `data/processed/msr_action3d/`.
- Fixed and seeded cross-subject split files under `data/splits/`.
- Dataset validation and `results/tables/dataset_summary.csv`.
- Exact canonical-angle sweep under `results/raw/canonical_angles_exact.csv`.
- Raw DTW and PCA+DTW baselines under `results/raw/dtw_baselines.csv`.
- First main comparison table under `results/tables/main_results.csv`.

The next research phase is the quantum-estimated subspace-affinity extension.

## Quick Setup Check

From the repository root:

```bash
python scripts/check_setup.py
```

This verifies that project imports work and writes a tiny metadata-rich CSV row to `results/raw/setup_check.csv`.

## Environment

The classical canonical-angle pipeline runs on CPU with NumPy. The DTW baseline can
run on CPU or on CUDA through PyTorch.

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

### `scripts/08_generate_tables.py`

Merges DTW result shards, regenerates DTW summaries, extracts PCA-only tables, and
writes the main comparison table.

```bash
python scripts/08_generate_tables.py --table all
```

Important arguments:

| Argument | Default | What it does |
|---|---:|---|
| `--table` | `all` | One of `all`, `dtw_baselines`, `main_results`. |

Outputs:

- `results/raw/dtw_baselines.csv`
- `results/raw/pca_dtw_baselines.csv`
- `results/tables/dtw_baselines_summary.csv`
- `results/tables/pca_dtw_baselines_summary.csv`
- `results/tables/main_results.csv`

## Current Headline Results

From `results/tables/main_results.csv`:

```text
Raw DTW:                accuracy 0.843636, macro-F1 0.825326
PCA+DTW, k=32:          accuracy 0.847273, macro-F1 0.829881
Exact canonical angles: accuracy 0.737226, macro-F1 0.717567
```

## Key Documents

- `RESEARCH_PLAN_quantum_canonical_angles_HAR.md`: original research plan.
- `RESEARCH_PROCESS_quantum_canonical_angles_HAR.md`: step-by-step execution roadmap.
- `RESEARCH_STATUS_quantum_canonical_angles_HAR.md`: living task/status tracker.
