# Massive Research Upgrade Plan  
## Q-SDTW: Quantum-Estimable Subspace Dynamic Time Warping for Skeleton-Based Human Action Recognition

**Working title:**  
**Q-SDTW: Quantum-Estimable Local Motion-Subspace Alignment for Skeleton-Based Human Action Recognition**

**Project goal:**  
Upgrade the current SWAP-test subspace-affinity experiment into a large, publishable research project that combines:

1. Classical DTW from the previous master's research.
2. SVD/PCA/QPCA-style dimensionality reduction.
3. Global motion-subspace representation.
4. Local temporal motion subspaces.
5. Canonical-angle-inspired projection affinity.
6. SWAP-test quantum overlap estimation.
7. Optional Quantum Amplitude Estimation for measurement-efficiency analysis.
8. Multi-dataset validation on MSR Action3D, UTKinect, and UTD-MHAD.

---

# 1. Core Thesis of the New Research

The upgraded research should not be framed as:

> "We used the SWAP test for human action recognition."

That is too small.

The upgraded research should be framed as:

> Skeleton actions contain both temporal structure and geometric motion structure. Raw DTW captures temporal alignment but operates in high-dimensional frame space. Global subspace methods are compact and quantum-compatible but lose temporal ordering. We propose a local motion-subspace alignment framework that preserves coarse temporal structure while producing a distance function that can be estimated using quantum overlap primitives such as the SWAP test.

The new contribution is not the SWAP test alone.  
The contribution is the full framework:

> **Local Subspace-DTW + Quantum-Estimable Subspace Affinity + Shot/Noise/Generalization Analysis**

---

# 2. Main Research Questions

## RQ1: Can skeleton actions be represented as sequences of local low-rank motion subspaces?

Instead of representing a full action sequence as one matrix and comparing it globally, split the action into short temporal windows. Each window becomes a local motion subspace.

Given a skeleton sequence:

```text
X in R^(T x d)
```

where:

```text
T = number of frames
d = skeleton feature dimension
```

split it into windows:

```text
X = [W_1, W_2, ..., W_m]
```

For each window:

```text
W_i in R^(L x d)
```

compute SVD:

```text
W_i = A_i Sigma_i V_i^T
```

keep the top `r` right singular vectors:

```text
U_i in R^(d x r)
```

Now each action is represented as:

```text
X -> {U_1, U_2, ..., U_m}
```

This gives a sequence of local motion subspaces.

---

## RQ2: Can DTW align sequences of local motion subspaces better than global subspace comparison?

Raw DTW compares individual frames:

```text
d(x_t, y_s) = ||x_t - y_s||_2
```

Local Subspace-DTW compares local motion subspaces:

```text
d(U_i, V_j) = 1 - (1/r) ||U_i^T V_j||_F^2
```

This distance is related to canonical angles:

```text
||U_i^T V_j||_F^2 = sum_k cos^2(theta_k)
```

So the distance becomes:

```text
d(U_i, V_j) = 1 - (1/r) sum_k cos^2(theta_k)
```

This keeps the temporal-alignment strength of DTW while replacing raw frame distance with a compact geometric motion distance.

---

## RQ3: Can this subspace distance be estimated quantumly?

The SWAP test estimates squared overlap:

```text
|<u | v>|^2
```

For two subspace bases:

```text
U = [u_1, ..., u_r]
V = [v_1, ..., v_r]
```

the projection affinity is:

```text
S(U,V) = (1/r) sum_i sum_j |<u_i | v_j>|^2
```

This is exactly compatible with SWAP-test estimation.

Therefore, the quantum-estimated local subspace distance is:

```text
d_q(U,V) = 1 - (1/r) sum_i sum_j SWAP_ESTIMATE(|<u_i | v_j>|^2)
```

---

## RQ4: Does the quantum-estimated distance converge to the exact classical distance?

This is the current strongest result from the existing report. The upgraded version should make it formal and larger:

- Distance MAE vs shots
- RMSE vs shots
- nearest-neighbor agreement vs shots
- prediction agreement vs shots
- accuracy vs shots
- noise robustness

---

## RQ5: Can the upgraded method generalize beyond MSR Action3D?

Use multiple datasets:

1. MSR Action3D
2. UTKinect Action3D
3. UTD-MHAD
4. Optional NTU RGB+D subset

---

# 3. Datasets

## 3.1 Dataset 1: MSR Action3D

**Role:** Main development and comparison dataset.

Use this first because the current repo already supports it.

Known setup from the current experiment report:

```text
Sequences: 566 processed usable sequences
Classes: 20
Subjects: 10
Feature dimension: 60
Skeleton format: 20 joints x 3D coordinates
Current split:
Train subjects: 1, 3, 5, 7, 9
Test subjects:  2, 4, 6, 8, 10
```

One all-zero sequence is skipped during preprocessing:

```text
a13_s09_e02
```

### Required MSR experiments

1. Raw DTW baseline
2. PCA+DTW baseline
3. SVD/PCA/QPCA-style DTW from previous master's repo
4. Global exact subspace affinity
5. Exact canonical-angle distances
6. Local Subspace-DTW
7. SWAP-estimated global subspace affinity
8. SWAP-estimated Local Subspace-DTW on subsets
9. Noise simulation
10. Shot-ablation analysis

---

## 3.2 Dataset 2: UTKinect Action3D

**Role:** Mandatory generalization dataset.

Why use it:

- Small enough for quantum experiments.
- Skeleton-based.
- Different from MSR Action3D.
- Good for proving the method is not MSR-only.

Known dataset properties:

```text
Actions: 10
Subjects: 10
Repetitions: each subject performs each action twice
Modalities: RGB, depth, skeleton joint locations
Frame rate: approximately 30 fps
```

### Recommended UTKinect split

Use cross-subject split:

```text
Train subjects: 1, 3, 5, 7, 9
Test subjects:  2, 4, 6, 8, 10
```

Also add leave-one-subject-out if time allows.

### Required UTKinect experiments

1. Raw DTW
2. PCA+DTW
3. Global subspace affinity
4. Local Subspace-DTW
5. SWAP-estimated subspace affinity
6. SWAP-estimated Local Subspace-DTW on subset or full, depending runtime

---

## 3.3 Dataset 3: UTD-MHAD

**Role:** Strong second benchmark.

Why use it:

- Larger and more modern than UTKinect.
- More classes than UTKinect.
- Still much smaller than NTU RGB+D.
- Has skeleton data and additional modalities.

Known dataset properties:

```text
Actions: 27
Subjects: 8
Usable sequences: 861 after removing corrupted sequences
Modalities: RGB, depth, skeleton, inertial signals
```

### Recommended UTD-MHAD split

Common split:

```text
Train subjects: 1, 3, 5, 7
Test subjects:  2, 4, 6, 8
```

or:

```text
Train subjects: 1, 2, 3, 4
Test subjects:  5, 6, 7, 8
```

Use one split as the main protocol, but document it clearly.

### Required UTD-MHAD experiments

1. Raw DTW
2. PCA+DTW
3. Global subspace affinity
4. Local Subspace-DTW
5. SWAP-estimated subspace affinity on subset
6. Shot-ablation on a selected subset
7. Optional noisy quantum simulation

---

## 3.4 Dataset 4: NTU RGB+D Subset

**Role:** Optional stretch benchmark.

Do not use full NTU RGB+D at the beginning. It is too large for full pairwise quantum-simulation experiments.

Use only a controlled subset, for example:

```text
10 classes
10 subjects
1 camera/view
selected skeleton sequences only
```

Use NTU only if MSR + UTKinect + UTD-MHAD are already complete.

### Optional NTU subset experiments

1. Raw DTW
2. PCA+DTW
3. Global subspace affinity
4. Local Subspace-DTW
5. No quantum full pairwise simulation unless heavily subsetted

---

# 4. Method Family

The paper should compare a family of methods, not only one method.

---

## 4.1 Method 0: Raw DTW Baseline

Input:

```text
X in R^(T x d)
Y in R^(S x d)
```

Frame distance:

```text
d(x_t, y_s) = ||x_t - y_s||_2
```

DTW distance:

```text
DTW(X,Y)
```

Settings:

```text
window ratios: none, 0.1, 0.2
normalization: path-length normalized
backend: CPU and/or GPU
```

Purpose:

- Strong classical baseline.
- Temporal alignment reference.
- Shows how much accuracy is lost or gained by subspace methods.

---

## 4.2 Method 1: PCA+DTW / SVD+DTW

This connects to the previous master's research.

Procedure:

1. Fit PCA on training frames only.
2. Project train/test frames to `k` dimensions.
3. Run DTW in the lower-dimensional space.

Parameters:

```text
k values: 4, 8, 12, 16, 24, 32
DTW windows: none, 0.1, 0.2
```

Purpose:

- Reuse and merge the master's QDTW direction.
- Establish dimensionality-reduced temporal baseline.
- Compare with local subspace representation.

---

## 4.3 Method 2: Global Exact Subspace Affinity

Represent each full sequence as one subspace.

For sequence:

```text
X in R^(T x d)
```

do:

```text
center X
SVD(X)
keep top r right singular vectors U_X in R^(d x r)
```

For pair:

```text
S(X,Y) = (1/r) ||U_X^T U_Y||_F^2
d(X,Y) = 1 - S(X,Y)
```

Parameters:

```text
r values: 1, 2, 3, 4, 5, 6, 8, 10, 12
```

Purpose:

- Fair exact classical reference for the SWAP-estimated method.
- Must be added because the current quantum method estimates projection affinity, not mean angle directly.

---

## 4.4 Method 3: Exact Canonical-Angle Distances

Compute:

```text
M = U_X^T U_Y
singular_values(M) = cos(theta_i)
theta_i = arccos(sigma_i)
```

Distances:

```text
chordal
projection
mean_angle
max_angle
```

Purpose:

- Canonical-angle interpretation.
- Compare projection affinity with explicit angle distances.
- Helps justify the geometry.

---

## 4.5 Method 4: Local Subspace-DTW

This is the main new classical method.

### Step 1: Window the sequence

Given:

```text
X in R^(T x d)
```

create local windows:

```text
W_1, W_2, ..., W_m
```

with:

```text
window length L
stride s
```

Parameters:

```text
L values: 5, 10, 15, 20
stride values: 2, 5, 10
```

Handle very short sequences:

- If `T < L`, use the full sequence as one window.
- Or pad/repeat boundary frames.
- Document the choice.

### Step 2: Compute local subspace

For each window:

```text
W_i in R^(L x d)
```

center and compute SVD:

```text
W_i = A_i Sigma_i V_i^T
```

keep:

```text
U_i in R^(d x r)
```

Parameters:

```text
local rank r: 1, 2, 3, 4, 6
```

### Step 3: Define local subspace distance

For two local subspaces:

```text
U_i, V_j
```

distance:

```text
d(U_i,V_j) = 1 - (1/r) ||U_i^T V_j||_F^2
```

Also test:

```text
chordal distance
mean angle distance
projection distance
```

### Step 4: Run DTW over subspace sequences

Now each action is:

```text
X -> [U_1, U_2, ..., U_m]
Y -> [V_1, V_2, ..., V_n]
```

Run DTW where the local cost is:

```text
cost(i,j) = d(U_i,V_j)
```

Output:

```text
LocalSubspaceDTW(X,Y)
```

### Parameters

```text
window length L: 5, 10, 15, 20
stride s: 2, 5, 10
rank r: 1, 2, 3, 4, 6
DTW window: none, 0.1, 0.2
distance: projection_affinity, chordal, mean_angle
```

### Purpose

This is the big methodological upgrade.

Expected benefit:

- Keeps temporal alignment.
- Uses compact local geometry.
- Should outperform global subspace affinity.
- May become competitive with raw DTW or PCA+DTW.
- Quantum-estimable because local distance is based on overlaps.

---

## 4.6 Method 5: SWAP-Estimated Global Subspace Affinity

This is the current quantum method.

For each pair of subspaces:

```text
U = [u_1, ..., u_r]
V = [v_1, ..., v_r]
```

estimate:

```text
|<u_i | v_j>|^2
```

using SWAP test.

Then compute:

```text
S_q(U,V) = (1/r) sum_i sum_j SWAP_ESTIMATE(|<u_i|v_j>|^2)
d_q(U,V) = 1 - S_q(U,V)
```

Parameters:

```text
r: 1, 2, 3, 4, 6
shots: 128, 256, 512, 1024, 2048, 4096
backend: exact, sampling, qiskit_aer
```

Purpose:

- Formalize the existing quantum result.
- Show shot convergence.
- Show prediction agreement with exact affinity.

---

## 4.7 Method 6: SWAP-Estimated Local Subspace-DTW

This is the big quantum method.

Same as Local Subspace-DTW, but each local cost is estimated with SWAP tests:

```text
cost_q(i,j) = 1 - (1/r) sum_a sum_b SWAP_ESTIMATE(|<u_a^i | v_b^j>|^2)
```

Then run DTW over the estimated cost matrix:

```text
Q-LocalSubspaceDTW(X,Y)
```

Parameters:

```text
window length L: 10, 15
stride s: 5
rank r: 2, 3
shots: 512, 1024, 2048
DTW window: none or 0.1
```

Do not sweep everything at first because runtime will explode.

Use a staged approach:

1. Find best exact Local Subspace-DTW settings.
2. Use only top 2 or 3 settings for quantum estimation.
3. Run quantum on subset first.
4. Expand if results are promising.

---

## 4.8 Method 7: QAE-Enhanced Overlap Estimation

This is optional but important for making the quantum part stronger.

Basic SWAP sampling has measurement complexity:

```text
O(1/epsilon^2)
```

Quantum Amplitude Estimation can theoretically improve this to:

```text
O(1/epsilon)
```

The experiment does not need to be large.

Compare:

```text
regular SWAP sampling
QAE-style overlap estimation
```

Metrics:

```text
overlap error
distance MAE
number of quantum queries
classification stability on tiny subset
```

Purpose:

- Adds an algorithmic quantum contribution beyond ordinary SWAP tests.
- Shows a path toward measurement efficiency.
- Makes the paper stronger for quantum reviewers.

---

# 5. Preprocessing Pipeline

All datasets should be converted into one unified format:

```text
data/processed/{dataset}/sequences.npz
data/processed/{dataset}/labels.npy
data/processed/{dataset}/subjects.npy
data/processed/{dataset}/sequence_ids.npy
data/processed/{dataset}/metadata.json
```

Each sequence should be:

```text
X_i in R^(T_i x d)
```

---

## 5.1 Skeleton Normalization

For every dataset:

1. Remove invalid/all-zero frames.
2. Root-center skeletons.
3. Normalize body scale.
4. Optionally rotate/align skeleton orientation.
5. Flatten joints into feature vectors.
6. Save feature dimension in metadata.

Suggested normalization options:

```text
root joint: hip center / spine base
scale: average bone length or shoulder-hip distance
orientation: optional torso-aligned coordinate frame
```

Run ablation:

```text
root-centered only
root-centered + scale normalized
root-centered + scale + orientation normalized
```

Purpose:

- Check whether subspace methods are sensitive to skeleton normalization.
- Improve cross-subject generalization.

---

## 5.2 Temporal Normalization

For raw DTW, variable length is okay.

For local subspaces:

- Keep variable length.
- Use sliding windows.
- Handle short sequences carefully.

Optional temporal preprocessing:

```text
smoothing: none, moving average, Savitzky-Golay
resampling: none, 32 frames, 64 frames
velocity features: position only, position+velocity
```

Ablation:

```text
position only
velocity only
position + velocity
```

This could be important because subspaces of velocity may capture motion style better than raw positions.

---

# 6. Experimental Plan

---

## Experiment Group A: Dataset Validation

### Goal

Verify every dataset loads correctly and produces clean skeleton matrices.

### Datasets

```text
MSR Action3D
UTKinect Action3D
UTD-MHAD
optional NTU subset
```

### Outputs

```text
results/tables/dataset_summary.csv
```

Columns:

```text
dataset
num_sequences
num_classes
num_subjects
min_frames
max_frames
mean_frames
feature_dim
num_skipped_sequences
```

### Acceptance criteria

- No all-zero sequences remain.
- Labels are correct.
- Subjects are correct.
- Feature dimensions match metadata.
- Splits contain no subject leakage.

---

## Experiment Group B: Classical Baselines

### Goal

Establish strong classical references.

### Methods

1. Raw DTW
2. PCA+DTW
3. SVD/PCA+DTW from master's repo
4. Global subspace affinity
5. Exact canonical-angle distances
6. Local Subspace-DTW

### Datasets

```text
MSR Action3D
UTKinect
UTD-MHAD
```

### Metrics

```text
accuracy
macro-F1
runtime
per-class F1
confusion matrix
```

### Output files

```text
results/raw/classical_baselines_{dataset}.csv
results/tables/classical_baselines_summary_{dataset}.csv
results/figures/classical_accuracy_runtime_{dataset}.png
results/figures/confusion_matrix_{method}_{dataset}.png
```

### Expected outcome

Local Subspace-DTW should beat global subspace affinity and ideally approach or beat raw DTW/PCA+DTW.

---

## Experiment Group C: Global Subspace Rank Ablation

### Goal

Understand how rank affects global subspace classification.

### Method

Global exact subspace affinity and exact canonical-angle distances.

### Parameters

```text
r: 1, 2, 3, 4, 5, 6, 8, 10, 12
distance: projection_affinity, chordal, mean_angle, max_angle
```

### Datasets

```text
MSR Action3D
UTKinect
UTD-MHAD
```

### Outputs

```text
results/raw/global_subspace_rank_{dataset}.csv
results/tables/global_subspace_rank_summary_{dataset}.csv
results/figures/global_accuracy_vs_rank_{dataset}.png
```

### Questions answered

- Is low rank consistently better?
- Does rank 2 remain best across datasets?
- Does higher rank add noise?

---

## Experiment Group D: Local Subspace-DTW Parameter Sweep

### Goal

Find the best local subspace settings.

### Parameters

```text
window length L: 5, 10, 15, 20
stride s: 2, 5, 10
rank r: 1, 2, 3, 4, 6
DTW window: none, 0.1, 0.2
distance: projection_affinity, chordal, mean_angle
features: position only, velocity only, position+velocity
```

### Datasets

Start with:

```text
MSR Action3D
```

Then transfer top settings to:

```text
UTKinect
UTD-MHAD
```

### Outputs

```text
results/raw/local_subspace_dtw_sweep_{dataset}.csv
results/tables/local_subspace_dtw_best_{dataset}.csv
results/figures/local_sdtw_heatmap_{dataset}.png
```

### Heatmaps

Produce heatmaps:

```text
accuracy vs window length and rank
accuracy vs stride and rank
runtime vs window length and stride
```

### Expected result

This is the experiment most likely to make the research bigger and stronger.

---

## Experiment Group E: Exact-vs-SWAP Global Affinity

### Goal

Formalize the current quantum result.

### Method

Compare:

```text
exact global projection affinity
SWAP-estimated global projection affinity
```

### Parameters

```text
r: 1, 2, 3, 4, 6
shots: 128, 256, 512, 1024, 2048, 4096
backend: sampling
seeds: 0, 1, 2, 3, 4
```

### Subset settings

Small subset:

```text
train_per_class: 3
test_per_class: 2
```

Larger subset:

```text
train_per_class: 5
test_per_class: 3
```

If runtime allows:

```text
train_per_class: 7
test_per_class: 4
```

### Metrics

```text
exact accuracy
quantum accuracy
accuracy gap
exact macro-F1
quantum macro-F1
distance MAE
distance RMSE
max distance error
nearest-neighbor agreement
prediction agreement
runtime
```

### Outputs

```text
results/raw/swap_global_exact_comparison_{dataset}.csv
results/tables/swap_global_exact_comparison_summary_{dataset}.csv
results/figures/swap_global_mae_vs_shots_{dataset}.png
results/figures/swap_global_accuracy_vs_shots_{dataset}.png
results/figures/swap_global_prediction_agreement_{dataset}.png
```

### Expected result

Distance error should decrease as shots increase.

---

## Experiment Group F: Exact-vs-SWAP Local Subspace-DTW

### Goal

Test the main quantum method.

### Method

Compare:

```text
Exact Local Subspace-DTW
SWAP-estimated Local Subspace-DTW
```

### Parameters

Only use best settings from Group D.

Example:

```text
window length L: 10 or 15
stride s: 5
rank r: 2 or 3
DTW window: none or 0.1
shots: 512, 1024, 2048
```

### Datasets

Start:

```text
MSR Action3D subset
```

Then:

```text
UTKinect full or subset
UTD-MHAD subset
```

### Metrics

```text
exact Local-SDTW accuracy
quantum Local-SDTW accuracy
accuracy gap
distance MAE
DTW cost matrix MAE
nearest-neighbor agreement
prediction agreement
runtime
```

### Additional analysis

For a few sequence pairs, visualize:

```text
exact local cost matrix
quantum-estimated local cost matrix
absolute error matrix
DTW alignment path exact
DTW alignment path quantum
```

### Outputs

```text
results/raw/swap_local_sdtw_comparison_{dataset}.csv
results/tables/swap_local_sdtw_comparison_summary_{dataset}.csv
results/figures/local_cost_matrix_exact_{dataset}.png
results/figures/local_cost_matrix_quantum_{dataset}.png
results/figures/local_cost_matrix_error_{dataset}.png
results/figures/local_dtw_path_exact_vs_quantum_{dataset}.png
```

### Expected result

The quantum-estimated local DTW path should become closer to the exact path as shots increase.

This is a strong visual and methodological contribution.

---

## Experiment Group G: Noise Simulation

### Goal

Make the quantum section realistic.

### Method

Use Qiskit Aer or custom noisy probability model.

Noise types:

```text
shot noise
readout noise
depolarizing noise
amplitude damping
```

Noise levels:

```text
0.001
0.005
0.01
0.02
```

### Experiments

Run for:

```text
global affinity r=2
local Subspace-DTW best setting
shots: 1024, 2048
```

### Metrics

```text
overlap MAE
subspace distance MAE
classification accuracy
prediction agreement
noise sensitivity curve
```

### Outputs

```text
results/raw/noise_simulation_{dataset}.csv
results/tables/noise_simulation_summary_{dataset}.csv
results/figures/noise_vs_distance_error_{dataset}.png
results/figures/noise_vs_accuracy_{dataset}.png
```

### Expected result

Even if performance drops under noise, the analysis is valuable.

Important framing:

> The goal is not to claim current hardware advantage, but to characterize how noise affects motion-subspace affinity estimation.

---

## Experiment Group H: Qiskit Aer Circuit Validation

### Goal

Show that the SWAP-test implementation is not only binomial sampling from a formula.

### Tiny setup

Use:

```text
dataset: MSR or UTKinect
classes: 2 to 5
train_per_class: 1
test_per_class: 1
rank: 2
shots: 1024
```

Compare backends:

```text
exact
sampling
qiskit_aer_statevector
qiskit_aer_qasm
noisy_qasm
```

### Metrics

```text
overlap estimate error
distance estimate error
prediction agreement
runtime
```

### Outputs

```text
results/raw/aer_validation.csv
results/tables/aer_validation_summary.csv
```

### Expected result

Aer qasm should agree with sampling under ideal conditions.

---

## Experiment Group I: QAE Measurement-Efficiency Experiment

### Goal

Add a stronger quantum-algorithmic angle.

### Setup

Do this on pairwise overlaps first, not full classification.

Sample many vector pairs from local/global subspaces.

Compare:

```text
SWAP sampling
QAE-style estimation
```

### Independent variable

```text
number of quantum queries
```

### Dependent variables

```text
overlap MAE
distance MAE
confidence interval
```

### Outputs

```text
results/raw/qae_overlap_efficiency.csv
results/tables/qae_overlap_efficiency_summary.csv
results/figures/qae_vs_sampling_error.png
```

### Expected result

QAE should show better query efficiency in simulation.

Even if full classification is not run with QAE, this supports the future scalability claim.

---

## Experiment Group J: Dataset Generalization

### Goal

Show the method is not overfitted to MSR Action3D.

### Use best methods

For each dataset:

```text
Raw DTW
PCA+DTW
Global subspace affinity
Local Subspace-DTW
SWAP-estimated global affinity
SWAP-estimated Local Subspace-DTW subset
```

### Output table

```text
results/tables/cross_dataset_main_results.csv
```

Columns:

```text
dataset
method
best_config
accuracy
macro_f1
runtime
quantum_or_classical
```

### Expected conclusion

The method should be consistently valid across datasets, even if accuracy differs.

---

# 7. Metrics

Use the same metrics everywhere.

## Classification metrics

```text
accuracy
macro-F1
weighted-F1
per-class F1
confusion matrix
```

## Quantum estimation metrics

```text
overlap MAE
overlap RMSE
subspace distance MAE
subspace distance RMSE
max absolute error
nearest-neighbor agreement
prediction agreement
accuracy gap
macro-F1 gap
```

## Runtime metrics

```text
preprocessing time
feature extraction time
distance matrix time
classification time
total runtime
number of overlap estimates
number of circuits
number of shots
```

## DTW-specific metrics

```text
cost matrix error
alignment path agreement
normalized DTW distance
```

---

# 8. Required Figures for the Paper

## Figure 1: Overall framework

Pipeline:

```text
Skeleton sequence
-> preprocessing
-> local windows
-> SVD local subspaces
-> local subspace distance
-> DTW alignment
-> 1-NN classification
```

Add quantum branch:

```text
local subspace distance
-> SWAP-test overlap estimation
-> quantum-estimated local cost matrix
-> DTW
```

---

## Figure 2: Global vs Local Subspace

Show:

```text
full sequence -> one global subspace
full sequence -> many local subspaces
```

Explain why local subspaces preserve temporal structure.

---

## Figure 3: Accuracy vs rank

For global subspace and local subspace methods.

---

## Figure 4: Local Subspace-DTW parameter heatmap

Accuracy vs:

```text
window length
rank
```

---

## Figure 5: Shot convergence

Plots:

```text
distance MAE vs shots
accuracy vs shots
prediction agreement vs shots
```

---

## Figure 6: Exact vs quantum local cost matrix

For one pair:

```text
exact local cost matrix
quantum local cost matrix
absolute error
```

---

## Figure 7: DTW path comparison

Show exact DTW path and quantum-estimated DTW path.

---

## Figure 8: Noise robustness

```text
noise level vs distance error
noise level vs accuracy
```

---

## Figure 9: Cross-dataset results

Bar chart:

```text
method accuracy across MSR, UTKinect, UTD-MHAD
```

---

# 9. Paper Contribution Structure

The paper should claim the following contributions.

## Contribution 1

A local motion-subspace representation for skeleton action sequences.

## Contribution 2

A Subspace-DTW distance that aligns sequences of local motion subspaces instead of raw skeleton frames.

## Contribution 3

A quantum-estimable projection affinity based on SWAP-test squared-overlap estimation.

## Contribution 4

A shot-based convergence study showing that quantum-estimated subspace distances approach exact classical distances.

## Contribution 5

A multi-dataset evaluation against raw DTW, PCA+DTW, global subspace, and canonical-angle baselines.

## Contribution 6

A hardware-aware/noise-aware analysis using simulated quantum noise and optional QAE measurement-efficiency experiments.

---

# 10. Paper Framing

## Avoid these claims

Do not claim:

```text
quantum advantage
state-of-the-art HAR
SWAP test is new
full canonical angles are estimated by SWAP test alone
quantum method beats all classical baselines
```

## Use these claims

Use:

```text
quantum-compatible motion similarity
canonical-angle-inspired projection affinity
SWAP-test-estimated subspace distance
shot-noise convergence
hybrid temporal-geometric action recognition
path toward quantum-assisted HAR
```

---

# 11. Implementation Plan

## Phase 0: Repo Merge and Cleanup

Merge useful parts from the master's QDTW repo into the new repo.

Bring over:

```text
DTW implementation
PCA/QPCA experiment structure
result table generation style
plotting utilities
dataset preprocessing lessons
README structure
```

Do not blindly copy old code. Refactor into modules.

Suggested repo structure:

```text
q-sdtw-har/
├── configs/
│   ├── datasets/
│   │   ├── msr_action3d.yaml
│   │   ├── utkinect.yaml
│   │   └── utd_mhad.yaml
│   ├── experiments/
│   │   ├── raw_dtw.yaml
│   │   ├── pca_dtw.yaml
│   │   ├── global_subspace.yaml
│   │   ├── local_sdtw.yaml
│   │   ├── swap_global.yaml
│   │   ├── swap_local_sdtw.yaml
│   │   ├── noise.yaml
│   │   └── qae.yaml
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data/
│   ├── features/
│   ├── dtw/
│   ├── subspaces/
│   ├── quantum/
│   ├── evaluation/
│   └── visualization/
├── scripts/
│   ├── 00_prepare_dataset.py
│   ├── 01_validate_dataset.py
│   ├── 02_run_raw_dtw.py
│   ├── 03_run_pca_dtw.py
│   ├── 04_run_global_subspace.py
│   ├── 05_run_local_sdtw.py
│   ├── 06_run_swap_global.py
│   ├── 07_run_swap_local_sdtw.py
│   ├── 08_run_noise_sim.py
│   ├── 09_run_qae_efficiency.py
│   ├── 10_generate_tables.py
│   └── 11_generate_figures.py
├── results/
│   ├── raw/
│   ├── tables/
│   └── figures/
├── tests/
├── docs/
│   ├── METHOD.md
│   ├── EXPERIMENTS.md
│   └── REPRODUCIBILITY.md
└── README.md
```

---

## Phase 1: Dataset Support

Implement unified loaders:

```text
MSRAction3DLoader
UTKinectLoader
UTDMHADLoader
```

Each loader outputs:

```text
sequence array
label
subject
sequence id
metadata
```

Add validation tests:

```text
test_dataset_shapes.py
test_no_all_zero_sequences.py
test_split_no_leakage.py
```

---

## Phase 2: Classical Baselines

Implement or clean:

```text
raw DTW
PCA+DTW
global projection affinity
canonical-angle distances
1-NN classifier
```

Generate main classical results on MSR first.

---

## Phase 3: Local Subspace-DTW

Implement:

```text
make_windows(X, L, stride)
local_subspace(window, rank)
local_subspace_sequence(X)
subspace_distance(U, V)
local_sdtw_distance(seqA, seqB)
```

Optimization:

- Cache local subspaces.
- Precompute distance matrices.
- Use multiprocessing or GPU where possible.
- Start with exact implementation before quantum.

---

## Phase 4: SWAP Global Affinity

Clean current implementation.

Backends:

```text
exact
sampling
qiskit_aer
```

Add tests:

```text
identical vectors -> overlap approx 1
orthogonal vectors -> overlap approx 0
distance converges with shots
identical subspaces -> distance approx 0
```

---

## Phase 5: SWAP Local Subspace-DTW

Implement quantum-estimated local cost matrices.

Important caching:

For each pair of local subspaces:

```text
r^2 overlaps
```

This can explode.

Cache overlap estimates:

```text
cache key = dataset, seq_i, window_i, seq_j, window_j, rank, shots, seed
```

Start with subset experiments.

---

## Phase 6: Noise and Aer

Add:

```text
ideal qasm simulation
noisy qasm simulation
readout noise model
depolarizing noise model
```

Run tiny validation first.

---

## Phase 7: QAE

Implement query-efficiency experiment.

Do not start here. This is after the main method works.

---

## Phase 8: Paper Tables and Figures

Automate:

```text
main_results.csv
dataset_summary.csv
classical_results.csv
quantum_convergence.csv
noise_summary.csv
cross_dataset_results.csv
```

Every figure should be generated by script.

---

# 12. Priority Order

If time is limited, do this order:

## Must-have

1. Full exact projection-affinity baseline
2. Local Subspace-DTW exact method
3. MSR Action3D complete classical comparison
4. SWAP global shot-ablation
5. UTKinect dataset support
6. UTKinect classical comparison
7. SWAP local Subspace-DTW subset
8. Paper-ready figures

## Strongly recommended

9. UTD-MHAD support
10. Noise simulation
11. Larger quantum subset
12. Qiskit Aer tiny validation

## Stretch

13. QAE measurement-efficiency
14. NTU RGB+D subset
15. Real IBM hardware tiny demo

---

# 13. Minimum Publishable Version

A minimum strong workshop/poster version should include:

```text
MSR Action3D
UTKinect
Raw DTW
PCA+DTW
Global subspace affinity
Local Subspace-DTW
SWAP global affinity
SWAP local affinity subset
shot convergence plots
```

This is publishable as a serious early-stage quantum application paper.

---

# 14. Strong Full-Paper Version

A strong full-paper version should include:

```text
MSR Action3D
UTKinect
UTD-MHAD
Raw DTW
PCA+DTW
Global subspace
Exact canonical angles
Local Subspace-DTW
SWAP global affinity
SWAP Local Subspace-DTW
noise simulation
Aer validation
QAE measurement-efficiency
cross-dataset result table
```

This is the target if the goal is a big paper.

---

# 15. Expected Main Results Table

The final paper should have a table like:

| Dataset | Method | Accuracy | Macro-F1 | Runtime | Quantum-estimable |
|---|---:|---:|---:|---:|---|
| MSR | Raw DTW | ... | ... | ... | No |
| MSR | PCA+DTW | ... | ... | ... | Partially |
| MSR | Global Subspace | ... | ... | ... | Yes |
| MSR | Local Subspace-DTW | ... | ... | ... | Yes |
| MSR | SWAP Global | ... | ... | ... | Yes |
| MSR | SWAP Local-SDTW | ... | ... | ... | Yes |
| UTK | Raw DTW | ... | ... | ... | No |
| UTK | Local Subspace-DTW | ... | ... | ... | Yes |
| UTD | Raw DTW | ... | ... | ... | No |
| UTD | Local Subspace-DTW | ... | ... | ... | Yes |

---

# 16. Expected Quantum Convergence Table

| Dataset | Method | Rank | Shots | Exact Acc | Quantum Acc | MAE | NN Agree | Pred Agree |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MSR | Global | 2 | 128 | ... | ... | ... | ... | ... |
| MSR | Global | 2 | 1024 | ... | ... | ... | ... | ... |
| MSR | Local-SDTW | 2 | 512 | ... | ... | ... | ... | ... |
| MSR | Local-SDTW | 2 | 2048 | ... | ... | ... | ... | ... |
| UTK | Global | 2 | 1024 | ... | ... | ... | ... | ... |

---

# 17. Writing Strategy

## Abstract structure

1. Problem: Skeleton HAR needs temporal alignment and compact motion representation.
2. Gap: DTW is accurate but expensive; global subspace methods are compact but lose temporal order; quantum overlap estimation has not been integrated into local motion alignment.
3. Method: Local Subspace-DTW and SWAP-estimated projection affinity.
4. Experiments: MSR Action3D, UTKinect, UTD-MHAD.
5. Results: Local Subspace-DTW improves global subspace, quantum estimates converge with shots, noise analysis characterizes hardware sensitivity.
6. Claim: A quantum-compatible motion similarity framework, not a quantum advantage claim.

---

# 18. Final Recommendation

Yes, merge the master's QDTW research with this new quantum subspace-affinity research.

But do not present it as a messy combination.

Present it as a clean evolution:

```text
Frame-level DTW
-> PCA/SVD-reduced DTW
-> Global motion subspaces
-> Local motion-subspace DTW
-> SWAP-estimated quantum local subspace affinity
-> noise/QAE scalability analysis
```

The final big idea:

> **Temporal alignment from DTW + geometric compression from subspaces + quantum estimability from overlap estimation.**

That is the massive version of the project.

---

# 19. Immediate Next Commands / Tasks

## Task 1: Add exact projection-affinity baseline

Script:

```bash
python scripts/04_run_global_subspace.py \
  --dataset msr_action3d \
  --distance projection_affinity chordal mean_angle max_angle \
  --r-values 1 2 3 4 5 6 8 10 12 \
  --seeds 0 1 2 3 4
```

## Task 2: Implement exact Local Subspace-DTW

Script target:

```bash
python scripts/05_run_local_sdtw.py \
  --dataset msr_action3d \
  --window-lengths 5 10 15 20 \
  --strides 2 5 10 \
  --r-values 1 2 3 4 6 \
  --dtw-windows none 0.1 0.2 \
  --distance projection_affinity \
  --seeds 0 1 2 3 4
```

## Task 3: Generate local subspace heatmaps

```bash
python scripts/11_generate_figures.py \
  --figure local_sdtw_heatmap \
  --dataset msr_action3d
```

## Task 4: Run SWAP global convergence with more shots

```bash
python scripts/06_run_swap_global.py \
  --dataset msr_action3d \
  --r-values 2 3 4 \
  --shots 128 256 512 1024 2048 4096 \
  --subset true \
  --train-per-class 5 \
  --test-per-class 3 \
  --seeds 0 1 2 3 4
```

## Task 5: Run SWAP Local-SDTW on best exact settings

```bash
python scripts/07_run_swap_local_sdtw.py \
  --dataset msr_action3d \
  --window-length 10 \
  --stride 5 \
  --rank 2 \
  --shots 512 1024 2048 \
  --subset true \
  --train-per-class 3 \
  --test-per-class 2 \
  --seeds 0 1 2
```

## Task 6: Add UTKinect support

```bash
python scripts/00_prepare_dataset.py --dataset utkinect
python scripts/01_validate_dataset.py --dataset utkinect
```

Then repeat classical and quantum subset experiments.

---

# 20. Final Project Name Options

Best names:

1. **Q-SDTW**
   - Quantum Subspace Dynamic Time Warping

2. **Q-MoSA**
   - Quantum Motion Subspace Affinity

3. **QS-HAR**
   - Quantum Subspace Human Action Recognition

4. **Q-LSA**
   - Quantum Local Subspace Alignment

Recommended:

```text
Q-SDTW: Quantum-Estimable Subspace Dynamic Time Warping
```

This is the strongest name because it connects directly to the master's DTW research and the new quantum subspace work.
