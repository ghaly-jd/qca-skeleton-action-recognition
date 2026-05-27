# STATUS — Q-SDTW Research Tracker

**Last updated:** 2026-05-27
**Active plan:** `RESEARCH_PROCESS_V2_few_shot_qsdtw.md`
**Operating guide:** `CLAUDE.md`

This file is the single source of truth for what is done, what is in progress, and what is next. It is updated after every completed step. Do not work from memory; work from this file.

---

## How to Read This File

- **One row per step.** Steps are numbered `<phase>.<step>` matching the step IDs in `CLAUDE.md`.
- **Status** is one of: `Not started`, `In progress`, `Done`, `Blocked`, `Skipped`.
- **Evidence** is the file path(s) the step produced. Without evidence, a step is not Done.
- **Notes** is a short comment (optional).

After each step, also append to the **Decision Log** at the bottom if you made any non-trivial choice.

---

## Status Legend

| Status | Meaning |
| --- | --- |
| Not started | No work begun |
| In progress | Started, acceptance criteria not yet met |
| Done | Acceptance criteria met, evidence linked |
| Blocked | Waiting on external input; blocker documented below |
| Skipped | Explicitly deferred (optional steps only) |

---

## Phase Summary

| Phase | Title | Status | Steps Done / Total | Gate |
| --- | --- | --- | --- | --- |
| 1 | Methodology Lockdown | In progress | 8 / 25 | Phase 1 acceptance (Step 1.25) |
| 2 | Few-Shot Centerpiece | Not started | 0 / 14 | Phase 2 acceptance (Step 2.14) |
| 3 | Mechanism + Generalization | Not started | 0 / 18 | None (continues into Phase 4) |
| 4 | Paper + Polish | Not started | 0 / 13 | Submission (Step 4.13) |

---

## Phase 1 — Methodology Lockdown

| Step | Description | Status | Evidence | Notes |
| --- | --- | --- | --- | --- |
| 1.1 | Bootstrap status tracker, bump seeds to 10 | Done | configs/experiment_main.yaml, configs/experiment_quantum.yaml, configs/msr_action3d.yaml, data/splits/msr_cross_subject_seed{5..9}.json | |
| 1.2 | Update result writer schema with `feature_mode` | Done | src/eval/result_writer.py (RESULT_FIELDNAMES, ResultRecord), scripts/check_setup.py | |
| 1.3 | Add `statistics` module (Wilcoxon, mean/std) | Done | src/eval/statistics.py, tests/test_statistics.py (13 passed) | |
| 1.4 | Add `scripts/aggregate_results.py` | Done | scripts/aggregate_results.py, src/eval/aggregate.py, tests/test_aggregate_results.py (8 passed) | |
| 1.5 | Implement motion features module | Done | src/features/motion_features.py, configs/msr_action3d.yaml (bone_hierarchy added) | |
| 1.6 | Add motion feature tests | Done | tests/test_motion_features.py (25 passed) | Fixed acceleration to use direct 2nd differences |
| 1.7 | Add `--feature-mode` to existing scripts | Done | scripts/02_run_dtw_baselines.py, scripts/03_run_subspace_angles.py, scripts/04_run_quantum_angles_sim.py | |
| 1.8 | Implement clean Local-SDTW module | Done | src/distances/local_sdtw.py; pytest (100 passed, 2 skipped); raw API smoke: identical=1.998e-16, random=0.6561 | Added raw-array API while preserving precomputed-subspace path |
| 1.9 | Add Local-SDTW tests | Not started | | |
| 1.10 | Implement MLP baseline | Not started | | |
| 1.11 | Implement LSTM baseline | Not started | | |
| 1.12 | Implement Random Forest baseline | Not started | | |
| 1.13 | Implement KDTW baseline | Not started | | |
| 1.14 | Implement GAK baseline | Not started | | Optional |
| 1.15 | Add `scripts/09_run_classical_baselines.py` | Not started | | |
| 1.16 | Profile and optimize quantum simulation | Not started | | |
| 1.17 | Run classical baselines (full data, 10 seeds, all feature modes) | Not started | | |
| 1.18 | Re-run DTW baselines (full data, 10 seeds, all feature modes) | Not started | | |
| 1.19 | Run Exact Local-SDTW (full data, 10 seeds, all feature modes) | Not started | | |
| 1.20 | Run SWAP Local-SDTW on full data | Not started | | Long run |
| 1.21 | Aggregate Phase 1 results | Not started | | |
| 1.22 | Generate Phase 1 comparison figure | Not started | | |
| 1.23 | Pre-flight few-shot probe (K=1, 1 seed) | Not started | | **Gate** before Phase 2 |
| 1.24 | Write `phase1_findings.md` | Not started | | |
| 1.25 | Phase 1 acceptance check | Not started | | **Gate** |

---

## Phase 2 — Few-Shot Centerpiece

| Step | Description | Status | Evidence | Notes |
| --- | --- | --- | --- | --- |
| 2.1 | Implement `FewShotProtocol` class | Not started | | |
| 2.2 | Add few-shot protocol tests | Not started | | |
| 2.3 | Add `scripts/10_run_few_shot.py` | Not started | | |
| 2.4 | Run classical few-shot baselines | Not started | | |
| 2.5 | Pre-train (or load) ST-GCN backbone | Not started | | |
| 2.6 | Implement ProtoNet wrapper | Not started | | |
| 2.7 | Implement fine-tuned ST-GCN baseline | Not started | | |
| 2.8 | Run ProtoNet-STGCN few-shot | Not started | | |
| 2.9 | Run fine-tuned ST-GCN few-shot | Not started | | |
| 2.10 | Run Q-SDTW few-shot at all K | Not started | | Headline result |
| 2.11 | Aggregate Phase 2 results | Not started | | |
| 2.12 | Generate few-shot curve figure | Not started | | |
| 2.13 | Write `phase2_findings.md` | Not started | | |
| 2.14 | Phase 2 acceptance check | Not started | | **Gate** |

---

## Phase 3 — Mechanism + Generalization

| Step | Description | Status | Evidence | Notes |
| --- | --- | --- | --- | --- |
| 3.1 | Implement noise-injection comparators | Not started | | |
| 3.2 | Add `scripts/12_run_shot_noise_ablation.py` | Not started | | |
| 3.3 | Run shot-noise-as-regularizer ablation | Not started | | |
| 3.4 | Generate shot-noise figure | Not started | | |
| 3.5 | Implement Hadamard test | Not started | | |
| 3.6 | Add Hadamard test tests | Not started | | |
| 3.7 | Add Q-SDTW-H method variant | Not started | | |
| 3.8 | Run Q-SDTW-H on MSR | Not started | | |
| 3.9 | Add UTKinect loader | Not started | | |
| 3.10 | Add UTD-MHAD loader | Not started | | |
| 3.11 | Process and validate UTKinect | Not started | | |
| 3.12 | Process and validate UTD-MHAD | Not started | | |
| 3.13 | Run few-shot on UTKinect | Not started | | |
| 3.14 | Run few-shot on UTD-MHAD | Not started | | |
| 3.15 | Add `scripts/13_run_early_recognition.py` | Not started | | |
| 3.16 | Run early recognition experiment | Not started | | |
| 3.17 | Generate cross-dataset and early-recognition figures | Not started | | |
| 3.18 | Write `phase3_findings.md` | Not started | | |

---

## Phase 4 — Paper + Polish

| Step | Description | Status | Evidence | Notes |
| --- | --- | --- | --- | --- |
| 4.1 | Create paper folder structure | Not started | | |
| 4.2 | Write Method section | Not started | | |
| 4.3 | Write Experiments section | Not started | | |
| 4.4 | Write Results section | Not started | | |
| 4.5 | Write Related Work | Not started | | |
| 4.6 | Write Introduction | Not started | | |
| 4.7 | Write Abstract | Not started | | |
| 4.8 | Write Limitations | Not started | | |
| 4.9 | Finalize all figures | Not started | | |
| 4.10 | Failure analysis | Not started | | |
| 4.11 | Write `scripts/reproduce_paper.sh` | Not started | | |
| 4.12 | Internal review | Not started | | |
| 4.13 | Tag release and submit | Not started | | |

---

## Decision Log

Append a row to this table whenever a non-obvious choice is made (e.g., choosing pyskl vs from-scratch ST-GCN, dropping an optional step, picking a specific window size, etc).

| Date | Decision | Rationale | Step |
| --- | --- | --- | --- |
| 2026-05-27 | Adopt few-shot framing as primary contribution | Full-data MSR Action3D unwinnable vs deep-learning SOTA; few-shot is defensible | Planning |
| 2026-05-27 | 10 seeds default, not 5 | Tighter CIs for small effect sizes | 1.1 |
| 2026-05-27 | Drop NTU-RGB+D from scope | Scope discipline; UTKinect + UTD-MHAD sufficient | Planning |
| 2026-05-27 | Preserve legacy `LocalSubspaceSequence` Local-SDTW path while adding the Step 1.8 raw-array API | Existing exact and SWAP scripts already depend on precomputed local subspaces; dispatch keeps those paths stable | 1.8 |

---

## Open Blockers

Use this table for anything preventing progress. Resolve and clear.

| Blocker | Step | Date raised | Resolution / Next action |
| --- | --- | --- | --- |
| | | | |

---

## Notes for the Next Session

(Free-form scratch area — short notes the next session might need. Clear when no longer relevant.)

- First session: start with Step 1.1.
- The current best results are still being treated as preliminary until Phase 1 re-runs validate them with multi-seed + Wilcoxon.
- Avoid touching `data/processed/` and `data/raw/` — those are dataset artifacts and not part of the code change scope.
