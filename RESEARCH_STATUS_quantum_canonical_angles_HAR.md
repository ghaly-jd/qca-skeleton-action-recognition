# Research Status: Quantum-Estimated Canonical Angles for HAR

Last updated: 2026-05-20

Source plan: `RESEARCH_PLAN_quantum_canonical_angles_HAR.md`  
Process roadmap: `RESEARCH_PROCESS_quantum_canonical_angles_HAR.md`

## Status Legend

| Status | Meaning |
|---|---|
| Not started | No implementation or artifact exists yet. |
| In progress | Work has started, but the acceptance criteria are not met yet. |
| Blocked | Waiting on data, decision, access, dependency, or another task. |
| Done | Artifact exists and passes the listed done condition. |
| Optional | Useful for the strong version, but not required for the minimum publishable version. |

## Current Overall State

| Area | Status | Evidence / Notes |
|---|---|---|
| Research plan | Done | `RESEARCH_PLAN_quantum_canonical_angles_HAR.md` exists. |
| Process roadmap | Done | `RESEARCH_PROCESS_quantum_canonical_angles_HAR.md` created. |
| Status tracker | Done | This file created. |
| Codebase implementation | In progress | Phase 2 exact subspace-angle pipeline and Phase 3 classical baselines exist; quantum pipeline is next. |
| Main dataset results | In progress | MSR Action3D is processed; exact canonical-angle, raw DTW, PCA+DTW, and main comparison table exist. |
| Quantum experiments | Not started | Quantum circuit code is not implemented yet. |
| Paper draft | Not started | Paper folder and LaTeX draft are not created yet. |

## Immediate Next Milestone

Ready to show advisor/sensei when these are done:

| Task | Status | Evidence / Output |
|---|---|---|
| Clean repo structure | Done | Folders exist: `src/`, `scripts/`, `configs/`, `data/`, `results/`, `tests/`, `paper/`. |
| MSR Action3D processed | Done | `data/processed/msr_action3d/` and `results/tables/dataset_summary.csv`; 566 usable sequences, one all-zero sequence skipped. |
| Exact canonical-angle accuracy table | Done | `results/raw/canonical_angles_exact.csv`; best chordal result is rank 2, accuracy 0.7263, macro-F1 0.7081. |
| Raw DTW baseline | Done | `results/raw/dtw_baselines.csv`; best window is none, accuracy 0.8436, macro-F1 0.8253. |
| Accuracy vs rank plot | Done | `results/figures/accuracy_vs_rank.png`. |
| PCA+DTW baseline | Done | `results/raw/pca_dtw_baselines.csv`; best setting is k=32, window none, accuracy 0.8473, macro-F1 0.8299. |
| Main comparison table | Done | `results/tables/main_results.csv`; includes Raw DTW, PCA+DTW, and Exact Canonical Angles. |
| Quantum extension paragraph | Not started | Short method paragraph in paper notes or README. |

---

## Phase 0: Project Setup and Research Hygiene

| ID | Step | Status | Evidence / Output | Notes |
|---|---|---|---|---|
| 0.1 | Create repository folder structure | Done | `configs/`, `data/`, `src/`, `scripts/`, `results/`, `tests/`, `notebooks/`, `paper/` | Placeholder `.gitkeep` files preserve empty directories. |
| 0.2 | Add `requirements.txt` | Done | `requirements.txt` | Includes classical ML, plotting, test, and Qiskit dependencies. |
| 0.3 | Add `environment.yml` | Done | `environment.yml` | Conda environment with pip-installed Qiskit packages. |
| 0.4 | Add `pyproject.toml` | Done | `pyproject.toml` | Defines package metadata, package discovery, and pytest settings. |
| 0.5 | Add `.gitignore` | Done | `.gitignore` | Excludes caches, raw data, generated results, and paper build artifacts. |
| 0.6 | Add config files | Done | `configs/*.yaml` | Dataset, main experiment, quantum experiment, and figure configs exist. |
| 0.7 | Add utility modules | Done | `src/utils/*.py` | Seed, IO, logging, and path helpers exist. |
| 0.8 | Define result metadata schema | Done | `src/eval/result_writer.py` | Includes dataset, method, seed, parameters, accuracy, macro_f1, runtime, commit, timestamp. |

Done condition: `python scripts/check_setup.py` imports project modules and writes `results/raw/setup_check.csv`.

---

## Phase 1: MSR Action3D Dataset Pipeline

| ID | Step | Status | Evidence / Output | Notes |
|---|---|---|---|---|
| 1.1 | Place/download MSR Action3D raw data | Done | `data/raw/MSRAction3D/` | Loaded from `OneDrive_1_5-20-2026.zip`; raw files are gitignored. |
| 1.2 | Document dataset source and manual steps | Done | `README.md` | Includes zip extraction and prepare/validate commands. |
| 1.3 | Implement MSR loader | Done | `src/data/msr_loader.py` | Parses sequence id, subject id, action id, repetition, frames, joints, and confidence. |
| 1.4 | Implement invalid frame removal | Done | `src/data/preprocessing.py` | Removes non-finite/all-zero coordinate frames; skips fully invalid sequences. |
| 1.5 | Implement joint flattening | Done | Processed `T x 60` arrays | Joint-major order. |
| 1.6 | Implement root centering | Done | Preprocessed arrays | Uses hip/spine root mapping with zero-based joint index 6 for MSR. |
| 1.7 | Implement optional scale normalization | Done | Config-controlled preprocessing | Enabled in `configs/msr_action3d.yaml`. |
| 1.8 | Implement optional z-score normalization | Done | Config-controlled preprocessing | Sequence-level mode only; global split-aware standardization deferred. |
| 1.9 | Save processed MSR arrays | Done | `sequences.npz`, `labels.npy`, `subjects.npy`, `metadata.json` | Also writes `sequence_ids.npy` and `repetition_ids.npy`. |
| 1.10 | Create fixed cross-subject split | Done | `data/splits/msr_cross_subject.json` | Train: 1,3,5,7,9. Test: 2,4,6,8,10. |
| 1.11 | Create seeded split variants | Done | `msr_cross_subject_seed*.json` | Seeds 0 through 4. |
| 1.12 | Implement dataset validation | Done | `src/data/validation.py` | Checks counts, labels, subjects, frames, dimensions, and split integrity. |
| 1.13 | Add dataset preparation script | Done | `scripts/00_prepare_dataset.py` | CLI entry point. |
| 1.14 | Add dataset validation script | Done | `scripts/01_validate_dataset.py` | Writes dataset summary. |
| 1.15 | Generate dataset summary table | Done | `results/tables/dataset_summary.csv` | 566 sequences, 20 classes, 10 subjects, feature dim 60. |

Done condition: `python scripts/01_validate_dataset.py --dataset msr_action3d` prints a clean summary and writes `results/tables/dataset_summary.csv`.

---

## Phase 2: Exact Sequence-as-Subspace Pipeline

| ID | Step | Status | Evidence / Output | Notes |
|---|---|---|---|---|
| 2.1 | Implement sequence normalization helpers | Done | `src/features/normalization.py` | Sequence centering, z-score, frame L2 options. |
| 2.2 | Implement per-sequence SVD | Done | `src/features/sequence_subspace.py` | Returns `D x r` basis. |
| 2.3 | Validate basis orthonormality | Done | `tests/test_subspace.py` | Checks SVD basis shape and orthonormality. |
| 2.4 | Implement canonical angles | Done | `src/distances/canonical_angles.py` | Clips singular values before arccos. |
| 2.5 | Implement chordal distance | Done | `src/distances/subspace_distances.py` | Main distance. |
| 2.6 | Implement projection distance | Done | `src/distances/subspace_distances.py` | Ablation. |
| 2.7 | Implement mean/max/min angle distances | Done | `src/distances/subspace_distances.py` | Ablation. |
| 2.8 | Implement 1-NN classifier | Done | `src/eval/knn.py` | Similarity-metric evaluation only. |
| 2.9 | Implement metrics | Done | `src/eval/metrics.py` | Accuracy, macro-F1, and confusion matrix helper. |
| 2.10 | Add subspace and angle tests | Done | `tests/test_subspace.py`, `tests/test_canonical_angles.py` | Includes identical, orthogonal, and sign-flipped cases. |
| 2.11 | Add exact canonical-angle experiment script | Done | `scripts/03_run_subspace_angles.py` | Rank and distance sweeps. |
| 2.12 | Run MSR exact canonical-angle experiment | Done | `results/raw/canonical_angles_exact.csv` | 180 rows across 5 seeds, 9 ranks, and 4 distances. |
| 2.13 | Generate accuracy vs rank plot | Done | `results/figures/accuracy_vs_rank.png` | First key figure. |

Done condition: `results/raw/canonical_angles_exact.csv` exists and contains seed/rank/distance results.

---

## Phase 3: Classical Baselines

| ID | Step | Status | Evidence / Output | Notes |
|---|---|---|---|---|
| 3.1 | Implement raw DTW | Done | `src/distances/dtw.py`, `tests/test_dtw.py` | Euclidean frame distance; NumPy/Numba/Torch backends. |
| 3.2 | Add path-length normalization | Done | `src/distances/dtw.py` | Enabled for main baseline setting. |
| 3.3 | Add DTW window ratios | Done | `results/raw/dtw_baselines.csv` | None, 0.1, 0.2. |
| 3.4 | Add DTW baseline script | Done | `scripts/02_run_dtw_baselines.py` | CLI entry point with CPU/GPU backend option. |
| 3.5 | Run raw DTW baseline | Done | `results/raw/dtw_baselines.csv`, `results/tables/dtw_baselines_summary.csv` | Seeds 0 through 4; best window none, accuracy 0.8436, macro-F1 0.8253. |
| 3.6 | Implement PCA projection | Done | `src/features/pca_projection.py`, `tests/test_pca_projection.py` | Fit only on training frames. |
| 3.7 | Run PCA+DTW baseline | Done | `results/raw/pca_dtw_baselines.csv`, `results/tables/pca_dtw_baselines_summary.csv` | Best setting: k=32, window none, accuracy 0.8473, macro-F1 0.8299. |
| 3.8 | Add optional VQD/QPCA+DTW baseline | Optional | Optional CSV rows | Do after core results. |
| 3.9 | Generate main comparison table | Done | `results/tables/main_results.csv` | Includes Raw DTW, PCA+DTW, and Exact Canonical Angles. |

Done condition: `results/tables/main_results.csv` includes at least Raw DTW, PCA+DTW, and Exact Canonical Angles.

---

## Phase 4: Quantum-Estimated Subspace Affinity

| ID | Step | Status | Evidence / Output | Notes |
|---|---|---|---|---|
| 4.1 | Choose quantum simulator stack | Not started | Config entry | Plan suggests Qiskit Aer. |
| 4.2 | Implement amplitude encoding | Not started | `src/quantum/state_preparation.py` | Pad 60D vectors to power-of-two length if needed. |
| 4.3 | Implement SWAP test circuit | Not started | `src/quantum/swap_test.py` | Estimates squared overlaps. |
| 4.4 | Test SWAP test on known vectors | Not started | `tests/test_quantum_overlap.py` | Identical and orthogonal sanity checks. |
| 4.5 | Implement overlap estimation wrapper | Not started | `src/quantum/overlap_estimation.py` | Common interface for quantum overlap methods. |
| 4.6 | Implement quantum subspace affinity | Not started | `src/distances/quantum_estimated_angles.py` | Use `r^2` overlap estimates per sequence pair. |
| 4.7 | Add quantum simulation script | Not started | `scripts/04_run_quantum_angles_sim.py` | Start with subset mode. |
| 4.8 | Run small subset experiment | Not started | `results/raw/quantum_subspace_affinity.csv` | r: 2,3,4. shots: 128,256,512,1024. |
| 4.9 | Compare quantum affinity with exact method | Not started | Analysis table or notes | Check behavior before full run. |
| 4.10 | Implement optional signed inner-product method | Optional | `src/quantum/hadamard_test.py` | Future work unless time allows. |

Done condition: quantum-estimated subspace affinity runs on an MSR subset and writes clean results.

---

## Phase 5: Shot Ablation and Exact-vs-Quantum Error

| ID | Step | Status | Evidence / Output | Notes |
|---|---|---|---|---|
| 5.1 | Add shot ablation mode/script | Not started | `scripts/05_run_shot_ablation.py` | Shots 64 through 2048. |
| 5.2 | Run shot ablation | Not started | `results/raw/shot_ablation.csv` | r values 2 and 4. |
| 5.3 | Add distance-error mode | Not started | Script mode | Compare exact vs quantum distances. |
| 5.4 | Sample sequence pairs | Not started | Pair list or deterministic sampling | Target 500 pairs. |
| 5.5 | Run exact-vs-quantum distance error | Not started | `results/raw/distance_error.csv` | r values 2,4,6. |
| 5.6 | Generate accuracy vs shots figure | Not started | `results/figures/shot_ablation.*` | Main quantum figure. |
| 5.7 | Generate distance error vs shots figure | Not started | `results/figures/distance_error.*` | Shows estimation quality. |

Done condition: paper can make a supported claim about shot-count effects.

---

## Phase 6: Normalization, Encoding, and Distance Ablations

| ID | Step | Status | Evidence / Output | Notes |
|---|---|---|---|---|
| 6.1 | Implement normalization grid runner | Not started | `scripts/07_run_encoding_ablation.py` | 16-setting initial grid. |
| 6.2 | Run encoding/normalization ablation | Not started | `results/raw/encoding_ablation.csv` | r values 2,4,6. |
| 6.3 | Run distance metric ablation | Not started | Rows in `canonical_angles_exact.csv` | Chordal, projection, mean, max. |
| 6.4 | Generate normalization heatmap | Not started | `results/figures/encoding_heatmap.*` | Paper Figure 6. |
| 6.5 | Generate distance ablation table | Not started | `results/tables/ablation_results.csv` | Table 3 source. |
| 6.6 | Select final normalization setting | Not started | Config update and notes | Must be justified by results. |

Done condition: main method settings are chosen from documented ablation results.

---

## Phase 7: Noise Robustness

| ID | Step | Status | Evidence / Output | Notes |
|---|---|---|---|---|
| 7.1 | Implement Gaussian joint noise | Not started | Noise transform code | Levels 0.00 to 0.10. |
| 7.2 | Implement random frame dropping | Not started | Noise transform code | Ratios 0.00 to 0.30. |
| 7.3 | Implement temporal jitter | Not started | Noise transform code | Ratios 0.00 to 0.10. |
| 7.4 | Implement missing joints | Not started | Noise transform code | Ratios 0.00 to 0.20. |
| 7.5 | Add noise robustness script | Not started | `scripts/06_run_noise_ablation.py` | Compare DTW, exact angles, quantum affinity. |
| 7.6 | Run noise robustness experiments | Not started | `results/raw/noise_robustness.csv` | Can be postponed if time is short. |
| 7.7 | Generate noise robustness figure | Not started | `results/figures/noise_robustness.*` | Paper Figure 7. |

Done condition: noise degradation curves exist for the compared methods.

---

## Phase 8: UTKinect Generalization

| ID | Step | Status | Evidence / Output | Notes |
|---|---|---|---|---|
| 8.1 | Place/download UTKinect raw data | Blocked | `data/raw/UTKinect/` | Needs dataset files. |
| 8.2 | Implement UTKinect loader | Not started | `src/data/utkinect_loader.py` | Match preprocessing interface. |
| 8.3 | Process UTKinect | Not started | `data/processed/utkinect/` | Save arrays and metadata. |
| 8.4 | Validate UTKinect | Not started | Dataset summary row | Use `scripts/01_validate_dataset.py`. |
| 8.5 | Create UTKinect splits | Not started | `data/splits/utkinect*.json` | Define reproducible protocol. |
| 8.6 | Run UTKinect raw DTW | Not started | Result CSV rows | Same metrics. |
| 8.7 | Run UTKinect exact canonical angles | Not started | Result CSV rows | Same rank sweep if feasible. |
| 8.8 | Run UTKinect best quantum setting | Not started | Result CSV rows | Use best setting from MSR first. |
| 8.9 | Update main results with UTKinect | Not started | `results/tables/main_results.csv` | Strong-version requirement. |

Done condition: main results table contains both MSR Action3D and UTKinect.

---

## Phase 9: Tables, Figures, and Statistical Reporting

| ID | Step | Status | Evidence / Output | Notes |
|---|---|---|---|---|
| 9.1 | Implement result writer | Not started | `src/eval/result_writer.py` | Standard output schema. |
| 9.2 | Implement statistical summaries | Not started | `src/eval/statistics.py` | Mean, std, Wilcoxon. |
| 9.3 | Generate dataset summary table | Done | `results/tables/dataset_summary.csv` | Table 1. |
| 9.4 | Generate main results table | Done | `results/tables/main_results.csv` | Table 2 source; currently MSR classical comparison only. |
| 9.5 | Generate ablation table | Not started | `results/tables/ablation_results.csv` | Table 3. |
| 9.6 | Generate method overview figure | Not started | `results/figures/method_overview.*` | Figure 1. |
| 9.7 | Generate accuracy vs rank figure | Not started | `results/figures/accuracy_vs_rank.*` | Figure 2. |
| 9.8 | Generate DTW vs canonical angle figure | Not started | `results/figures/dtw_vs_angles.*` | Figure 3. |
| 9.9 | Generate shot ablation figure | Not started | `results/figures/shot_ablation.*` | Figure 4. |
| 9.10 | Generate distance error figure | Not started | `results/figures/distance_error.*` | Figure 5. |
| 9.11 | Generate encoding heatmap | Not started | `results/figures/encoding_heatmap.*` | Figure 6. |
| 9.12 | Generate noise robustness figure | Not started | `results/figures/noise_robustness.*` | Figure 7. |
| 9.13 | Create paper results CSV | Not started | `results/paper_results.csv` | Single source for paper numbers. |

Done condition: every paper table and figure can be regenerated from scripts.

---

## Phase 10: Paper Writing

| ID | Step | Status | Evidence / Output | Notes |
|---|---|---|---|---|
| 10.1 | Create paper folder and LaTeX skeleton | Not started | `paper/main.tex`, `paper/sections/` | Use target venue format later. |
| 10.2 | Write Method section | Not started | `paper/sections/method.tex` | Write before introduction. |
| 10.3 | Write Experiments section | Not started | `paper/sections/experiments.tex` | Dataset, splits, metrics, configs. |
| 10.4 | Write Results section | Not started | `paper/sections/results.tex` | Insert generated tables and figures. |
| 10.5 | Write Related Work | Not started | `paper/sections/related_work.tex` | After method/results are clear. |
| 10.6 | Write Introduction | Not started | `paper/sections/introduction.tex` | Keep claims honest. |
| 10.7 | Write Abstract | Not started | `paper/sections/abstract.tex` | Last or near-last. |
| 10.8 | Write Limitations | Not started | `paper/sections/limitations.tex` | Include squared-overlap limitation. |
| 10.9 | Add references | Not started | `paper/refs.bib` | Canonical angles, HAR, DTW, quantum overlap. |
| 10.10 | Build paper PDF | Not started | `paper/main.pdf` | Fix warnings before submission. |

Done condition: a complete draft exists with generated tables and figures.

---

## Phase 11: Reproduction Package and Submission Prep

| ID | Step | Status | Evidence / Output | Notes |
|---|---|---|---|---|
| 11.1 | Create reproduction script | Not started | `scripts/reproduce_paper.sh` | Runs core pipeline. |
| 11.2 | Audit result metadata | Not started | Result CSVs include commit/timestamp/runtime | Required for reproducibility. |
| 11.3 | Freeze dependencies | Not started | Updated `requirements.txt` and `environment.yml` | Include exact versions near submission. |
| 11.4 | Run reproduction script from clean state | Not started | Successful script log | Needed before sharing. |
| 11.5 | Final README update | Not started | `README.md` | Include commands and project summary. |
| 11.6 | Tag release | Not started | `v0.1-qce-submission` | Only after results are stable. |
| 11.7 | Archive release | Optional | Zenodo DOI | Useful for submission. |

Done condition: repository can be shared with documented commands and a stable commit hash.

---

## Minimum Publishable Version Tracker

| Requirement | Status | Evidence / Output |
|---|---|---|
| MSR Action3D loaded cleanly | Done | `results/tables/dataset_summary.csv` |
| Raw DTW baseline | Done | `results/raw/dtw_baselines.csv` |
| Exact canonical-angle method | Done | `results/raw/canonical_angles_exact.csv` |
| Quantum-estimated subspace affinity | Not started | `results/raw/quantum_subspace_affinity.csv` |
| Shot ablation | Not started | `results/raw/shot_ablation.csv` |
| Clear limitations | Not started | Paper limitations section or notes |
| Reproducible scripts | Not started | `scripts/reproduce_paper.sh` |

## Strong Version Tracker

| Requirement | Status | Evidence / Output |
|---|---|---|
| MSR Action3D | In progress | Dataset processed; exact canonical-angle, raw DTW, PCA+DTW, and main classical comparison complete; quantum experiments next. |
| UTKinect | Blocked | Needs raw data. |
| Raw DTW | Done | `results/raw/dtw_baselines.csv` and `results/tables/dtw_baselines_summary.csv`. |
| PCA + DTW | Done | `results/raw/pca_dtw_baselines.csv` and `results/tables/pca_dtw_baselines_summary.csv`. |
| VQD/QPCA + DTW | Optional | Optional baseline CSV. |
| Exact canonical angles | Done | `results/raw/canonical_angles_exact.csv` and `results/tables/canonical_angles_exact_summary.csv`. |
| Quantum-estimated subspace affinity | Not started | Quantum affinity CSV. |
| Shot ablation | Not started | Shot ablation CSV. |
| Exact-vs-quantum distance error | Not started | Distance error CSV. |
| Encoding ablation | Not started | Encoding ablation CSV. |
| Noise robustness | Not started | Noise robustness CSV. |
| Multi-seed statistics | Not started | Tables with mean and std. |
| Full reproducibility package | Not started | Reproduction script and release tag. |

## Decision Log

| Date | Decision | Reason |
|---|---|---|
| 2026-05-19 | Build exact classical pipeline before quantum circuits. | The quantum method depends on reliable subspace extraction and exact classical comparison. |
| 2026-05-19 | Treat SWAP-test method as quantum-estimated subspace affinity, not exact canonical angles. | SWAP test estimates squared overlaps, not signed inner products. |
| 2026-05-19 | Use this status file as the project tracker. | Keeps implementation progress tied to the research plan. |
| 2026-05-19 | Complete Phase 0 with a smoke-check script. | `scripts/check_setup.py` verifies imports and writes a metadata-rich setup result row. |
| 2026-05-20 | Process MSR Action3D from supplied zip and skip all-zero recording `a13_s09_e02`. | Produces 566 usable `T x 60` sequences and a clean validation summary. |
| 2026-05-20 | Run raw DTW baseline with Torch CUDA backend. | Full 5-seed sweep across no window, 0.1, and 0.2 windows completed; unconstrained DTW is strongest so far. |
| 2026-05-20 | Run PCA+DTW baseline and generate first main comparison table. | PCA+DTW with k=32 and no DTW window is the strongest classical baseline so far. |
| 2026-05-20 | Complete exact canonical-angle rank/distance sweep on MSR Action3D. | Best chordal row is rank 2 with accuracy 0.7263 and macro-F1 0.7081; mean-angle rank 2 reaches accuracy 0.7372. |

## Open Blockers / Inputs Needed

| Blocker | Status | Needed Action |
|---|---|---|
| MSR Action3D raw data | Resolved | Supplied zip unpacked under `data/raw/MSRAction3D/`; raw files remain gitignored. |
| UTKinect raw data | Blocked | Add dataset files under `data/raw/UTKinect/` when ready for Phase 8. |
| Target venue template | Not started | Add IEEE Quantum Week/QCE template when paper writing begins. |
| Quantum dependency choice | Not started | Confirm Qiskit Aer or equivalent simulator during Phase 4. |

## How To Update This File

When a task is completed:

1. Change its status to `Done`.
2. Add the exact output path in `Evidence / Output`.
3. Add a note if any behavior differs from the original research plan.
4. Update the `Last updated` date.
5. Add important choices to the Decision Log.
