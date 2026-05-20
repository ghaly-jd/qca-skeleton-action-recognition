# Quantum-Estimated Canonical Angles for Skeleton-Based Action Recognition

## Research Plan and Experiment Roadmap

**Project name:** `quantum-canonical-angles-har`  
**Target venue:** IEEE Quantum Week / QCE Workshop or Poster  
**Main idea:** Treat each skeleton action sequence as a low-dimensional motion subspace, compare sequences using canonical angles, and implement a hybrid quantum version where subspace overlaps are estimated with quantum circuits.

---

## 1. Core Research Direction

The main research question is:

> Can skeleton action sequences be compared as low-dimensional motion subspaces, and can quantum overlap estimation reproduce canonical-angle sequence similarity well enough for action recognition?

Traditional DTW compares sequences by temporal alignment:

\[
X, Y \rightarrow d_{\text{DTW}}(X,Y)
\]

This project instead proposes:

\[
\text{sequence} \rightarrow \text{motion subspace} \rightarrow \text{canonical angles} \rightarrow \text{distance/classification}
\]

Then the quantum version becomes:

\[
\text{basis-vector overlaps} \rightarrow \text{quantum overlap estimation} \rightarrow \text{approximate canonical-angle / subspace-affinity distance}
\]

---

## 2. Main Contributions

The paper should claim the following contributions:

1. **Sequence-as-subspace representation**  
   Each skeleton action sequence is represented by a low-dimensional motion subspace extracted from the sequence itself.

2. **Canonical-angle sequence similarity**  
   Action sequences are compared using canonical angles between their motion subspaces instead of relying only on temporal alignment.

3. **Quantum-estimated subspace similarity**  
   A hybrid quantum-classical protocol estimates pairwise basis-vector overlaps using quantum circuits and uses them to approximate sequence-level subspace similarity.

4. **Comparison against DTW-based baselines**  
   The method is evaluated against raw DTW, PCA+DTW, and optionally VQD/QPCA+DTW.

5. **Shot-noise and rank analysis**  
   The study analyzes how shots, subspace rank, normalization, and quantum estimation noise affect classification accuracy and distance quality.

---

## 3. Clean Repository Structure

Create a new repo:

```bash
mkdir quantum-canonical-angles-har
cd quantum-canonical-angles-har
git init
```

Recommended structure:

```text
quantum-canonical-angles-har/
│
├── README.md
├── RESEARCH_PLAN.md
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── msr_action3d.yaml
│   ├── utkinect.yaml
│   ├── experiment_main.yaml
│   ├── experiment_quantum.yaml
│   └── paper_figures.yaml
│
├── data/
│   ├── raw/
│   │   ├── MSRAction3D/
│   │   └── UTKinect/
│   ├── processed/
│   │   ├── msr_action3d/
│   │   └── utkinect/
│   └── splits/
│       ├── msr_cross_subject.json
│       ├── msr_cross_subject_seed0.json
│       ├── msr_cross_subject_seed1.json
│       ├── msr_cross_subject_seed2.json
│       ├── msr_cross_subject_seed3.json
│       └── msr_cross_subject_seed4.json
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── msr_loader.py
│   │   ├── utkinect_loader.py
│   │   ├── preprocessing.py
│   │   ├── splits.py
│   │   └── validation.py
│   │
│   ├── features/
│   │   ├── sequence_subspace.py
│   │   ├── normalization.py
│   │   ├── pca_projection.py
│   │   └── encoding.py
│   │
│   ├── distances/
│   │   ├── dtw.py
│   │   ├── canonical_angles.py
│   │   ├── subspace_distances.py
│   │   └── quantum_estimated_angles.py
│   │
│   ├── quantum/
│   │   ├── state_preparation.py
│   │   ├── swap_test.py
│   │   ├── hadamard_test.py
│   │   ├── overlap_estimation.py
│   │   ├── noise_models.py
│   │   └── hardware_runner.py
│   │
│   ├── eval/
│   │   ├── knn.py
│   │   ├── metrics.py
│   │   ├── statistics.py
│   │   └── result_writer.py
│   │
│   └── utils/
│       ├── seed.py
│       ├── io.py
│       ├── logging.py
│       └── paths.py
│
├── scripts/
│   ├── 00_prepare_dataset.py
│   ├── 01_validate_dataset.py
│   ├── 02_run_dtw_baselines.py
│   ├── 03_run_subspace_angles.py
│   ├── 04_run_quantum_angles_sim.py
│   ├── 05_run_shot_ablation.py
│   ├── 06_run_noise_ablation.py
│   ├── 07_run_encoding_ablation.py
│   ├── 08_generate_tables.py
│   ├── 09_generate_figures.py
│   └── reproduce_paper.sh
│
├── results/
│   ├── raw/
│   ├── tables/
│   ├── figures/
│   └── paper_results.csv
│
├── notebooks/
│   └── exploration_only.ipynb
│
├── tests/
│   ├── test_loaders.py
│   ├── test_subspace.py
│   ├── test_canonical_angles.py
│   ├── test_dtw.py
│   └── test_quantum_overlap.py
│
└── paper/
    ├── main.tex
    ├── sections/
    ├── figures/
    └── refs.bib
```

Rule:

> Notebooks are only for exploration. Every number in the paper must be generated from scripts.

---

## 4. Datasets

Use two datasets.

### 4.1 Main Dataset: MSR Action3D

Expected processed format:

\[
X_i \in \mathbb{R}^{T_i \times 60}
\]

because:

\[
20 \text{ joints} \times 3 \text{ coordinates} = 60
\]

This should be the main benchmark.

### 4.2 Second Dataset: UTKinect-Action3D

Use UTKinect as a generalization dataset.

The goal is not to beat SOTA. The goal is to show that the subspace-angle behavior is not specific to one dataset.

---

## 5. Data Preprocessing

Every sequence should become:

\[
X \in \mathbb{R}^{T \times D}
\]

where:

\[
D = 60
\]

For every sequence:

1. Load raw skeleton data.
2. Remove invalid frames.
3. Flatten joints into a 60D vector:

\[
[J_1^x, J_1^y, J_1^z, ..., J_{20}^x, J_{20}^y, J_{20}^z]
\]

4. Root-center using hip/spine joint.
5. Optional scale normalization.
6. Optional z-score normalization.
7. Save processed data as `.npz`.

Recommended processed output:

```text
data/processed/msr_action3d/
├── sequences.npz
├── labels.npy
├── subjects.npy
└── metadata.json
```

Each sequence metadata item should include:

```json
{
  "sequence_id": "a01_s01_e01",
  "subject_id": 1,
  "action_id": 1,
  "num_frames": 42,
  "feature_dim": 60
}
```

---

## 6. Evaluation Splits

Use cross-subject evaluation.

For MSR Action3D, use a fixed cross-subject split:

```text
Train subjects: 1, 3, 5, 7, 9
Test subjects: 2, 4, 6, 8, 10
```

Also create five seeded split variants if needed:

```python
seeds = [0, 1, 2, 3, 4]
```

Every result table should report:

\[
\text{mean} \pm \text{std}
\]

Avoid single-seed results in the final paper.

---

## 7. Sequence-as-Subspace Method

Given one skeleton sequence:

\[
X \in \mathbb{R}^{T \times 60}
\]

First center the sequence:

\[
\tilde{X} = X - \mu_X
\]

Then compute SVD:

\[
\tilde{X} = A \Sigma U^\top
\]

Take the top \(r\) right singular vectors:

\[
U_X \in \mathbb{R}^{60 \times r}
\]

This \(U_X\) is the motion subspace of sequence \(X\).

Recommended values:

```python
r_values_full = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
r_values_paper = [2, 4, 6, 8, 10]
```

Use the full list for experiments, but only show the cleanest subset in the paper.

---

## 8. Canonical-Angle Distance

Given two sequence subspaces:

\[
U_X, U_Y \in \mathbb{R}^{60 \times r}
\]

Compute:

\[
M = U_X^\top U_Y
\]

Then:

\[
M = P \Sigma Q^\top
\]

The singular values are:

\[
\sigma_i = \cos(\theta_i)
\]

Therefore:

\[
\theta_i = \arccos(\sigma_i)
\]

Implementation detail:

```python
sigma = np.clip(sigma, -1.0, 1.0)
theta = np.arccos(sigma)
```

### 8.1 Main Distance: Chordal Distance

Use this as the main method:

\[
d_{\text{chordal}}(X,Y) =
\sqrt{\sum_{i=1}^{r} \sin^2(\theta_i)}
\]

### 8.2 Additional Distances for Ablation

Projection distance:

\[
d_{\text{proj}}(X,Y) =
\|U_XU_X^\top - U_YU_Y^\top\|_F
\]

Mean angle:

\[
d_{\text{mean}}(X,Y) =
\frac{1}{r}\sum_{i=1}^{r}\theta_i
\]

Maximum angle:

\[
d_{\text{max}}(X,Y) =
\max_i \theta_i
\]

Minimum angle:

\[
d_{\text{min}}(X,Y) =
\theta_1
\]

Main paper should emphasize chordal distance. The rest can go in ablation or appendix.

---

## 9. Classification Protocol

Use 1-nearest neighbor.

For each test sequence \(X_q\):

\[
\hat{y} = y_j
\]

where:

\[
j = \arg\min_j d(X_q, X_j)
\]

This keeps the method simple and makes the paper about similarity metrics, not model capacity.

---

## 10. Baselines

### 10.1 Raw DTW

Use the full skeleton sequence:

\[
X \in \mathbb{R}^{T \times 60}
\]

Frame distance:

\[
d(x_t, y_s) = \|x_t - y_s\|_2
\]

Parameters:

```yaml
dtw:
  frame_distance: euclidean
  normalize_by_path_length: true
  window_ratios: [null, 0.1, 0.2]
```

### 10.2 PCA + DTW

Fit global PCA on training frames only.

Project each frame:

\[
x_t \in \mathbb{R}^{60} \rightarrow z_t \in \mathbb{R}^{k}
\]

Then run DTW on projected sequences.

Use:

```python
k_values = [4, 8, 12, 16, 24, 32]
```

### 10.3 VQD/QPCA + DTW

This is the older quantum-PCA pipeline.

Use as an optional baseline, not the main method.

Use:

```python
k_values = [4, 8, 12, 16]
```

### 10.4 Exact Classical Canonical Angles

This is the new main non-quantum method.

Use:

```python
r_values = [2, 4, 6, 8, 10]
distance = "chordal"
```

### 10.5 Quantum-Estimated Subspace Similarity

This is the new quantum method.

Use quantum circuits to estimate basis-vector overlaps.

Initial parameters:

```python
r_values = [2, 3, 4]
shots = [128, 256, 512, 1024]
```

Later full run:

```python
r_values = [2, 4, 6, 8]
shots = [128, 256, 512, 1024, 2048]
```

---

## 11. Quantum Canonical-Angle / Subspace-Similarity Protocol

For two bases:

\[
U_X = [u_1, ..., u_r]
\]

\[
U_Y = [v_1, ..., v_r]
\]

Exact canonical angles require:

\[
G_{ij} = \langle u_i | v_j \rangle
\]

Then:

\[
\sigma(G) = \cos(\theta_i)
\]

The quantum part estimates these overlaps.

---

## 12. Quantum Version 1: SWAP-Test Subspace Affinity

A SWAP test estimates:

\[
|\langle u_i | v_j \rangle|^2
\]

Build an affinity matrix:

\[
A_{ij} = |\langle u_i | v_j \rangle|^2
\]

Define sequence similarity:

\[
S(X,Y) = \frac{1}{r}\sum_{i,j} A_{ij}
\]

Then distance:

\[
d_{\text{q-aff}}(X,Y) = 1 - S(X,Y)
\]

Important: this is not exact canonical angles. Call it:

> Quantum-estimated subspace affinity

This is the easiest publishable quantum version.

---

## 13. Quantum Version 2: Signed Inner-Product Estimation

For exact canonical angles, estimate:

\[
\langle u_i | v_j \rangle
\]

using a Hadamard-test-style circuit or another signed inner-product estimation method.

Then build:

\[
\hat{G}_{ij}
\]

and compute approximate canonical angles:

\[
\hat{\theta}_i = \arccos(\sigma_i(\hat{G}))
\]

This is the mathematically cleaner version, but more difficult.

Recommended strategy:

1. Implement Version 1 first.
2. Use Version 2 if time allows.
3. Mention Version 2 as future work if not completed.

---

## 14. Global Configuration

Recommended config:

```yaml
seeds: [0, 1, 2, 3, 4]

datasets:
  - msr_action3d
  - utkinect

preprocessing:
  root_center: true
  scale_normalize: true
  remove_invalid_frames: true
  coordinate_mode: xyz
  flatten_order: joint_major

normalization:
  sequence_centering: true
  feature_standardization:
    - none
    - zscore
  l2_normalize_frames:
    - false
    - true

subspace:
  extraction: per_sequence_svd
  center_sequence: true
  r_values: [1, 2, 3, 4, 5, 6, 8, 10, 12]
  min_frames_required: 10

canonical_angles:
  clip_singular_values: true
  distances:
    - chordal
    - projection
    - mean_angle
    - max_angle

dtw:
  frame_distance: euclidean
  normalize_by_path_length: true
  window_ratios: [null, 0.1, 0.2]

quantum:
  simulator: qiskit_aer
  state_preparation: amplitude_encoding
  shots: [64, 128, 256, 512, 1024, 2048]
  seed_simulator: true
  overlap_method:
    - swap_test
    - hadamard_test
  noise_model: true
  real_hardware: false
```

---

## 15. Experiments

## Experiment 1: Dataset Validation

Purpose:

Show the data pipeline is clean.

Command:

```bash
python scripts/00_prepare_dataset.py --dataset msr_action3d
python scripts/01_validate_dataset.py --dataset msr_action3d
```

Output:

```text
results/tables/dataset_summary.csv
```

Columns:

```text
dataset,num_sequences,num_classes,num_subjects,min_frames,max_frames,mean_frames,feature_dim
```

Paper output:

**Table 1: Dataset summary**

---

## Experiment 2: Raw DTW Baseline

Command:

```bash
python scripts/02_run_dtw_baselines.py \
  --dataset msr_action3d \
  --seeds 0 1 2 3 4 \
  --window-ratios none 0.1 0.2
```

Output:

```text
results/raw/dtw_baselines.csv
```

Columns:

```text
dataset,seed,method,window_ratio,accuracy,macro_f1,runtime_sec
```

Purpose:

Establish classical temporal-alignment baseline.

---

## Experiment 3: Exact Canonical-Angle Classification

Command:

```bash
python scripts/03_run_subspace_angles.py \
  --dataset msr_action3d \
  --r-values 1 2 3 4 5 6 8 10 12 \
  --distance chordal projection mean_angle max_angle \
  --seeds 0 1 2 3 4
```

Output:

```text
results/raw/canonical_angles_exact.csv
```

Columns:

```text
dataset,seed,r,distance,accuracy,macro_f1,runtime_sec
```

Paper figure:

**Accuracy vs subspace rank \(r\)**

Message:

> Sequence-level subspaces can classify actions without explicit temporal alignment.

---

## Experiment 4: DTW vs Canonical-Angle Main Comparison

Compare:

| Method | Description |
|---|---|
| Raw DTW | temporal alignment |
| PCA + DTW | global dimensionality reduction + temporal alignment |
| VQD/QPCA + DTW | quantum preprocessing + temporal alignment |
| Exact canonical angles | sequence-level subspace geometry |
| Quantum-estimated subspace affinity | quantum-estimated subspace similarity |

Output:

```text
results/tables/main_results.csv
```

Columns:

```text
dataset,method,parameter,accuracy_mean,accuracy_std,f1_mean,f1_std,runtime_mean
```

Paper output:

**Main Results Table**

---

## Experiment 5: Quantum-Estimated Subspace Affinity

First run on a small subset.

Use:

```yaml
subset:
  train_per_class: 3
  test_per_class: 2
```

Command:

```bash
python scripts/04_run_quantum_angles_sim.py \
  --dataset msr_action3d \
  --r-values 2 3 4 \
  --shots 128 256 512 1024 \
  --subset true \
  --seeds 0 1 2 3 4
```

Output:

```text
results/raw/quantum_subspace_affinity.csv
```

Columns:

```text
dataset,seed,r,shots,overlap_method,accuracy,macro_f1,runtime_sec
```

Important computational note:

For each pair of sequences, the number of overlap estimates is:

\[
r^2
\]

So for \(r = 4\), each sequence pair needs 16 overlap estimates.

---

## Experiment 6: Shot Ablation

Purpose:

Show how quantum-estimated similarity improves as the number of shots increases.

Use:

```python
shots = [64, 128, 256, 512, 1024, 2048]
r_values = [2, 4]
```

Command:

```bash
python scripts/05_run_shot_ablation.py \
  --dataset msr_action3d \
  --r-values 2 4 \
  --shots 64 128 256 512 1024 2048 \
  --seeds 0 1 2 3 4
```

Output:

```text
results/raw/shot_ablation.csv
```

Paper figure:

**Accuracy vs shots**

Expected message:

> Quantum-estimated subspace similarity approaches exact subspace similarity as shot count increases.

---

## Experiment 7: Exact-vs-Quantum Distance Error

Purpose:

Measure how close quantum-estimated distances are to exact classical distances.

For many sequence pairs:

\[
d_{\text{exact}}(X,Y)
\]

and:

\[
d_{\text{quantum}}(X,Y)
\]

Compute:

\[
|d_{\text{exact}} - d_{\text{quantum}}|
\]

Use:

```python
num_pairs = 500
r_values = [2, 4, 6]
shots = [128, 512, 1024, 2048]
```

Command:

```bash
python scripts/05_run_shot_ablation.py \
  --dataset msr_action3d \
  --mode distance_error \
  --num-pairs 500 \
  --r-values 2 4 6 \
  --shots 128 512 1024 2048
```

Output:

```text
results/raw/distance_error.csv
```

Columns:

```text
dataset,seed,r,shots,pair_id,exact_distance,quantum_distance,absolute_error
```

Paper figure:

**Quantum distance error vs shots**

---

## Experiment 8: Normalization / Encoding Ablation

Test:

| Factor | Values |
|---|---|
| Root centering | true |
| Scale normalization | true / false |
| Feature standardization | none / z-score |
| Frame L2 normalization | false / true |
| Sequence centering | true / false |

Initial grid:

```python
standardization = ["none", "zscore"]
scale_normalize = [True, False]
l2_frame = [True, False]
sequence_center = [True, False]
```

Total:

\[
2 \times 2 \times 2 \times 2 = 16
\]

Run for:

```python
r_values = [2, 4, 6]
```

Output:

```text
results/raw/encoding_ablation.csv
```

Paper figure:

**Encoding / normalization heatmap**

Message:

> Subspace-angle performance depends strongly on how skeleton geometry is normalized.

---

## Experiment 9: Noise Robustness

Purpose:

Test whether subspace-angle methods are robust to skeleton noise.

Noise types:

1. Gaussian joint noise
2. Random frame dropping
3. Temporal jitter
4. Missing joints

Parameters:

```yaml
gaussian_noise_std: [0.00, 0.01, 0.03, 0.05, 0.10]
frame_drop_ratio: [0.00, 0.10, 0.20, 0.30]
temporal_jitter_ratio: [0.00, 0.05, 0.10]
missing_joint_ratio: [0.00, 0.05, 0.10, 0.20]
```

Compare:

- Raw DTW
- Exact canonical angles
- Quantum-estimated subspace affinity

Output:

```text
results/raw/noise_robustness.csv
```

Paper figure:

**Accuracy degradation under noise**

---

## Experiment 10: Second Dataset Generalization

Repeat core experiments on UTKinect:

1. Dataset validation
2. Raw DTW
3. Exact canonical angles
4. Best quantum-estimated setting
5. Shot ablation, if time allows

Output:

```text
results/raw/utkinect_results.csv
```

Purpose:

Show that the method is not MSR-only.

---

## 16. Paper Tables and Figures

### Table 1: Dataset Summary

Columns:

```text
Dataset | #Sequences | #Classes | #Subjects | Mean Frames | Feature Dim
```

### Table 2: Main Results

Columns:

```text
Method | MSR Accuracy | UTKinect Accuracy | Runtime
```

Methods:

- Raw DTW
- PCA + DTW
- VQD/QPCA + DTW
- Exact Canonical Angles
- Quantum-Estimated Subspace Affinity

### Table 3: Ablation Results

Columns:

```text
Distance | r | Accuracy | Macro-F1
```

Distances:

- Chordal
- Projection
- Mean angle
- Max angle

### Figure 1: Method Overview

Pipeline:

```text
Skeleton sequence -> sequence matrix -> SVD subspace -> canonical angles -> 1-NN classification
```

Quantum branch:

```text
Subspace basis vectors -> quantum states -> overlap estimation -> subspace affinity
```

### Figure 2: Accuracy vs Subspace Rank

Message:

> There is an optimal subspace rank for action recognition.

### Figure 3: DTW vs Canonical Angles

Message:

> Subspace geometry is competitive with temporal alignment.

### Figure 4: Shot Ablation

Message:

> Quantum-estimated similarity stabilizes as shots increase.

### Figure 5: Exact-vs-Quantum Distance Error

Message:

> Quantum overlap estimation approximates classical subspace distances within shot-noise limits.

### Figure 6: Encoding / Normalization Heatmap

Message:

> Normalization choices strongly affect subspace geometry.

### Figure 7: Noise Robustness

Message:

> Subspace-angle methods degrade gracefully under skeleton noise.

---

## 17. Statistical Reporting

For every method:

\[
\text{accuracy mean} \pm \text{std}
\]

Use:

```python
seeds = [0, 1, 2, 3, 4]
```

For comparing methods:

- Use paired Wilcoxon signed-rank test.
- Compare per-seed accuracy.
- Report p-values in appendix or footnotes.

Example:

```text
Exact canonical angles outperformed raw DTW on MSR Action3D
by 4.2 percentage points on average across five seeds
(Wilcoxon signed-rank test, p = 0.031).
```

Do not report p-values if there are fewer than 5 seeds.

---

## 18. Reproducibility Requirements

Create:

```text
requirements.txt
environment.yml
scripts/reproduce_paper.sh
results/paper_results.csv
```

Every result must include:

```text
dataset
method
seed
parameters
accuracy
macro_f1
runtime_sec
git_commit
timestamp
```

At submission:

1. Push GitHub repo.
2. Tag release:

```bash
git tag v0.1-qce-submission
git push origin v0.1-qce-submission
```

3. Archive on Zenodo if possible.
4. Put code link and commit hash in the paper.

---

## 19. Implementation Order

Do not start with quantum circuits.

Start with the classical clean pipeline first.

### Phase 1: Repo + Dataset

- [ ] Create repo.
- [ ] Add folder structure.
- [ ] Download MSR Action3D.
- [ ] Write loader.
- [ ] Save processed `.npz`.
- [ ] Validate shapes and labels.
- [ ] Create cross-subject split.

Done when:

```bash
python scripts/01_validate_dataset.py --dataset msr_action3d
```

prints a clean dataset summary.

---

### Phase 2: Exact Canonical Angles

- [ ] Implement per-sequence SVD.
- [ ] Implement canonical angles.
- [ ] Implement chordal distance.
- [ ] Implement 1-NN classifier.
- [ ] Run exact canonical-angle classification.

Done when:

```bash
python scripts/03_run_subspace_angles.py --dataset msr_action3d
```

produces:

```text
results/raw/canonical_angles_exact.csv
```

---

### Phase 3: DTW Baselines

- [ ] Implement DTW.
- [ ] Run raw DTW.
- [ ] Run PCA + DTW.
- [ ] Compare against canonical angles.

Done when:

```text
results/tables/main_results.csv
```

has at least:

- Raw DTW
- PCA + DTW
- Exact canonical angles

---

### Phase 4: Quantum Overlap Simulation

- [ ] Implement amplitude encoding.
- [ ] Implement SWAP test.
- [ ] Estimate squared overlaps.
- [ ] Build quantum subspace affinity.
- [ ] Run on small subset.
- [ ] Run shot ablation.

Done when:

```text
results/raw/quantum_subspace_affinity.csv
results/raw/shot_ablation.csv
```

exist and are clean.

---

### Phase 5: Robustness + Encoding Ablations

- [ ] Run normalization ablation.
- [ ] Run noise robustness.
- [ ] Run distance metric ablation.

Done when:

```text
results/raw/encoding_ablation.csv
results/raw/noise_robustness.csv
```

exist.

---

### Phase 6: UTKinect

- [ ] Add UTKinect loader.
- [ ] Validate dataset.
- [ ] Run exact canonical angles.
- [ ] Run DTW baseline.
- [ ] Run best quantum setting.

Done when the main table has both datasets.

---

### Phase 7: Paper Writing

Write in this order:

1. Method
2. Experiments
3. Results
4. Related Work
5. Introduction
6. Abstract
7. Limitations

Do not write the introduction first.

---

## 20. Minimum Viable Publishable Version

If time is short, the minimum publishable paper needs:

- [ ] MSR Action3D loaded cleanly
- [ ] Raw DTW baseline
- [ ] Exact canonical-angle method
- [ ] Quantum-estimated subspace affinity on subset or full MSR
- [ ] Shot ablation
- [ ] Clear limitations
- [ ] Reproducible scripts

This can be a strong poster or workshop submission.

---

## 21. Strong Version

The strong version needs:

- [ ] MSR Action3D
- [ ] UTKinect
- [ ] Raw DTW
- [ ] PCA + DTW
- [ ] VQD/QPCA + DTW
- [ ] Exact canonical angles
- [ ] Quantum-estimated subspace affinity
- [ ] Shot ablation
- [ ] Exact-vs-quantum distance error
- [ ] Encoding ablation
- [ ] Noise robustness
- [ ] Multi-seed statistics
- [ ] Full reproducibility package

This is the version to aim for.

---

## 22. Risk Management

### Risk 1: Quantum method is slower and less accurate

That is okay.

Frame it as:

> The goal is not immediate quantum advantage. The goal is to evaluate whether quantum overlap estimation can reproduce useful subspace-similarity structure for sequence recognition.

### Risk 2: Canonical angles underperform DTW

Still publishable if the analysis is strong.

Frame it as:

> Canonical-angle methods capture global motion geometry, while DTW captures local temporal alignment. Their failure modes differ.

Also test hybrid distance:

\[
d_{\text{hybrid}} = \alpha d_{\text{DTW}} + (1-\alpha)d_{\text{angle}}
\]

with:

```python
alpha_values = [0.25, 0.5, 0.75]
```

### Risk 3: SWAP test only gives squared overlaps

Be honest.

Use the term:

> quantum-estimated subspace affinity

Do not claim exact canonical angles unless signed inner products are estimated.

### Risk 4: Too many experiments

Prioritize:

1. Exact canonical angles
2. DTW baseline
3. Quantum subspace affinity
4. Shot ablation
5. UTKinect
6. Noise / encoding ablation

---

## 23. Suggested Paper Title Options

### Conservative

**Canonical-Angle Sequence Similarity for Skeleton-Based Action Recognition**

### Quantum-focused

**Quantum-Estimated Subspace Similarity for Skeleton-Based Action Recognition**

### Best overall

**From Temporal Alignment to Quantum Subspace Similarity: Canonical-Angle Methods for Skeleton Action Recognition**

### More technical

**Hybrid Quantum-Classical Estimation of Canonical-Angle Sequence Similarity for Human Action Recognition**

---

## 24. One-Sentence Contribution

Use this version for now:

> We represent skeleton action sequences as low-dimensional motion subspaces and evaluate canonical-angle distances, including a hybrid quantum-estimated subspace-affinity variant, as sequence-similarity measures for human action recognition.

---

## 25. Immediate Next Steps

Do these first:

1. Create the repo.
2. Add the folder structure.
3. Download MSR Action3D.
4. Implement dataset loader.
5. Implement per-sequence SVD.
6. Implement exact canonical angles.
7. Run 1-NN classification with chordal distance.
8. Only after that, implement quantum overlap estimation.

Do not start with IBM hardware.  
Do not start with VQD.  
Do not start with paper writing.

Start with:

```bash
python scripts/00_prepare_dataset.py --dataset msr_action3d
python scripts/01_validate_dataset.py --dataset msr_action3d
python scripts/03_run_subspace_angles.py --dataset msr_action3d
```

Once exact canonical-angle classification works, the research is alive.

---

## 26. Definition of “Ready to Show Sensei”

You are ready to show Fukui sensei when you have:

- Clean repo structure
- MSR Action3D processed
- Exact canonical-angle accuracy table
- Raw DTW baseline
- One plot: accuracy vs subspace rank \(r\)
- One paragraph explaining the quantum extension

Do not wait until everything is perfect.

Show him the first clean result quickly.

---

## 27. Final Strategic Advice

The strongest version of this paper is not:

> Quantum beats classical.

The strongest version is:

> Skeleton sequences can be treated as motion subspaces, canonical angles provide a meaningful alternative to DTW, and quantum overlap estimation gives a principled way to approximate this subspace similarity.

That is honest, original, and publishable.

Ship the clean classical version first.  
Then add quantum estimation.  
Then add ablations.  
Then write the paper.
