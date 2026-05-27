# Research Status: Q-SDTW Upgrade

Last updated: 2026-05-26

Source process: `Q_SDTW_RESEARCH_PROCESS.md`

## Current Overall State

| Area | Status | Evidence / Output | Notes |
|---|---|---|---|
| Upgrade plan | Done | `Q_SDTW_Massive_Research_Upgrade_Plan.md` | User-provided plan. |
| Implementation process | Done | `Q_SDTW_RESEARCH_PROCESS.md` | Detailed process created from the upgrade plan and current repo state. |
| Q-SDTW status tracker | In progress | `Q_SDTW_RESEARCH_STATUS.md` | This file tracks the upgrade implementation. |
| Exact global projection affinity | Done | `results/raw/global_subspace_affinity.csv`, `results/tables/global_subspace_affinity_summary.csv` | 45 raw rows; best setting `r=2`, accuracy `0.726277`, macro-F1 `0.708104`. |
| Local motion subspace extraction | Done | `src/features/local_subspace.py`, `tests/test_local_subspace.py` | Supports position, velocity, and position+velocity features. |
| Exact Local Subspace-DTW | Done | `src/distances/local_sdtw.py`, `tests/test_local_sdtw.py` | Supports projection affinity and canonical-angle local costs; projection-affinity pairwise sweep now has `torch_cuda` backend support. |
| MSR Local-SDTW smoke run | Done | `results/raw/local_subspace_dtw_smoke.csv`, `results/tables/local_subspace_dtw_smoke_summary.csv` | `L=10`, stride `5`, rank `2`, seed `0`, limited train/test check passed. |
| MSR Local-SDTW full CUDA sweep | Done | `results/raw/local_subspace_dtw_msr_action3d.csv`, `results/tables/local_subspace_dtw_summary_msr_action3d.csv` | 855 raw rows; best setting `L=5`, stride `2`, rank `1`, DTW window `0.1`, accuracy `0.847273`, macro-F1 `0.833853`. |
| Local-SDTW tables and figures | Done | `results/tables/local_subspace_dtw_best_msr_action3d.csv`, `results/figures/local_sdtw_accuracy_heatmap_msr_action3d.png`, `results/figures/local_sdtw_runtime_heatmap_msr_action3d.png` | First summary table and heatmaps generated from the full CUDA sweep. |
| SWAP global convergence | Done for small subset | `results/raw/swap_global_exact_comparison_msr_action3d.csv`, `results/tables/swap_global_exact_comparison_summary_msr_action3d.csv`, SWAP convergence PNGs | Formal small-subset sweep is complete through 4096 shots; medium-subset sweep remains optional/pending. |
| SWAP Local-SDTW | Done for full MSR split | `results/raw/swap_local_sdtw_full_msr_action3d.csv`, `results/tables/swap_local_sdtw_full_summary_msr_action3d.csv`, SWAP Local-SDTW PNGs | Full split run complete for seeds `0 1 2`, shots `512 1024 2048`; best accuracy row `shots=512`, accuracy `0.852121`, macro-F1 `0.838612`; best convergence row `shots=2048`, distance MAE `0.004975`. |
| UTKinect support | Not started | Planned: `src/data/utkinect_loader.py` | Generalization phase. |

## Phase Tracker

| Phase | Status | Evidence / Output |
|---|---|---|
| 1. Formalize exact global projection affinity | Done | `results/raw/global_subspace_affinity.csv`, `results/tables/global_subspace_affinity_summary.csv` |
| 2. Implement local motion subspace extraction | Done | `src/features/local_subspace.py`, `tests/test_local_subspace.py` |
| 3. Implement exact Local Subspace-DTW | Done | `src/distances/local_sdtw.py`, `tests/test_local_sdtw.py` |
| 4. Run exact Local-SDTW on MSR | Done | `results/raw/local_subspace_dtw_msr_action3d.csv`, `results/tables/local_subspace_dtw_summary_msr_action3d.csv` |
| 5. Generate Local-SDTW tables and figures | Done | `results/tables/local_subspace_dtw_best_msr_action3d.csv`, `results/tables/q_sdtw_main_results.csv`, Local-SDTW heatmap PNGs |
| 6. Formalize SWAP global convergence | Done for small subset | `results/raw/swap_global_exact_comparison_msr_action3d.csv`; `results/tables/swap_global_exact_comparison_summary_msr_action3d.csv`; SWAP convergence PNGs |
| 7. Implement SWAP Local-SDTW | Done for full MSR split | `results/raw/swap_local_sdtw_full_msr_action3d.csv`; `results/tables/swap_local_sdtw_full_summary_msr_action3d.csv`; SWAP Local-SDTW convergence PNGs |
| 8. Add UTKinect support | Not started | `data/processed/utkinect/` |
| 9. Add UTD-MHAD support | Not started | `data/processed/utd_mhad/` |
| 10. Noise and Aer validation | Not started | `results/raw/noise_simulation_msr_action3d.csv` |
| 11. QAE efficiency experiment | Optional | `results/raw/qae_overlap_efficiency.csv` |
| 12. Statistical reporting and reproducibility | Not started | `scripts/reproduce_q_sdtw.sh` |

## Immediate Checklist

| Task | Status | Evidence / Output |
|---|---|---|
| Add local subspace extraction module | Done | `src/features/local_subspace.py` |
| Add local subspace tests | Done | `tests/test_local_subspace.py` |
| Add exact Local-SDTW distance module | Done | `src/distances/local_sdtw.py` |
| Add exact Local-SDTW tests | Done | `tests/test_local_sdtw.py` |
| Add exact global affinity script | Done | `scripts/05_run_global_subspace_affinity.py` |
| Run exact global projection-affinity full sweep | Done | `results/raw/global_subspace_affinity.csv`; 45 rows |
| Add exact Local-SDTW experiment script | Done | `scripts/06_run_local_sdtw.py` |
| Add CUDA backend for projection-affinity Local-SDTW | Done | `src/distances/local_sdtw.py`; `scripts/06_run_local_sdtw.py --backend torch_cuda` |
| Add Local-SDTW best-table generation | Done | `scripts/08_generate_tables.py --table local_sdtw` |
| Add Local-SDTW heatmap generation | Done | `scripts/12_generate_figures.py --figure local_sdtw_heatmaps` |
| Run focused unit tests | Done | `23 passed, 2 skipped` for local + quantum overlap tests. |
| Run MSR Local-SDTW smoke check | Done | `results/raw/local_subspace_dtw_smoke.csv` |
| Run MSR Local-SDTW full CUDA sweep | Done | `results/raw/local_subspace_dtw_msr_action3d.csv`; 855 rows |
| Generate first Local-SDTW full-sweep summary and heatmaps | Done | `results/tables/local_subspace_dtw_best_msr_action3d.csv`; `results/figures/local_sdtw_accuracy_heatmap_msr_action3d.png`; `results/figures/local_sdtw_runtime_heatmap_msr_action3d.png` |
| Add formal SWAP global convergence wrapper | Done | `scripts/07_run_swap_global_convergence.py` |
| Add SWAP global convergence figure generation | Done | `scripts/12_generate_figures.py --figure swap_global_convergence` |
| Run SWAP global convergence smoke check | Done | Temporary `/tmp/swap_global_convergence_smoke*.csv` and `/tmp/swap_global_*_smoke.png` files |
| Run formal SWAP global convergence small-subset sweep | Done | `results/raw/swap_global_exact_comparison_msr_action3d.csv`; 90 rows |
| Generate formal SWAP global convergence figures | Done | `results/figures/swap_global_mae_vs_shots_msr_action3d.png`; `results/figures/swap_global_accuracy_vs_shots_msr_action3d.png`; `results/figures/swap_global_prediction_agreement_msr_action3d.png` |
| Run formal SWAP global convergence medium-subset sweep | Pending | Optional target: same output schema with `train_per_class=5`, `test_per_class=3` |
| Add SWAP Local-SDTW distance module | Done | `src/distances/quantum_local_sdtw.py` |
| Add SWAP Local-SDTW tests | Done | `tests/test_quantum_local_sdtw.py` |
| Add SWAP Local-SDTW experiment script | Done | `scripts/09_run_swap_local_sdtw.py` |
| Add SWAP Local-SDTW reporting hooks | Done | `scripts/08_generate_tables.py`; `scripts/12_generate_figures.py --figure swap_local_sdtw_convergence` |
| Run SWAP Local-SDTW smoke check | Done | `results/raw/swap_local_sdtw_smoke.csv`; `results/tables/swap_local_sdtw_smoke_summary.csv`; `results/raw/swap_local_sdtw_pair_diagnostics_smoke.csv` |
| Run SWAP Local-SDTW all-class subset | Done | `results/raw/swap_local_sdtw_comparison_msr_action3d.csv`; 9 rows |
| Generate SWAP Local-SDTW convergence figures | Done | `results/figures/swap_local_sdtw_mae_vs_shots_msr_action3d.png`; `results/figures/swap_local_sdtw_accuracy_vs_shots_msr_action3d.png`; `results/figures/swap_local_sdtw_prediction_agreement_msr_action3d.png` |
| Run SWAP Local-SDTW full MSR split | Done | `results/raw/swap_local_sdtw_full_msr_action3d.csv`; 9 rows |
| Generate SWAP Local-SDTW full-run figures and table row | Done | `results/figures/swap_local_sdtw_*_msr_action3d.png`; `results/tables/q_sdtw_main_results.csv` |

## Verification Log

| Date | Check | Result |
|---|---|---|
| 2026-05-24 | `pytest tests/test_local_subspace.py tests/test_local_sdtw.py` | `14 passed` |
| 2026-05-24 | `pytest tests/test_local_subspace.py tests/test_local_sdtw.py tests/test_quantum_overlap.py` | `23 passed, 2 skipped` |
| 2026-05-24 | Global projection-affinity smoke run | Wrote `results/raw/global_subspace_affinity_smoke.csv` and summary CSV. |
| 2026-05-24 | Exact Local-SDTW smoke run | Wrote `results/raw/local_subspace_dtw_smoke.csv` and summary CSV. |
| 2026-05-24 | `pytest` | `46 passed, 2 skipped` |
| 2026-05-24 | `python -m py_compile scripts/08_generate_tables.py scripts/12_generate_figures.py` | Passed. |
| 2026-05-24 | Local-SDTW post-processing smoke using `results/tables/local_subspace_dtw_smoke_summary.csv` | Wrote temporary `/tmp` best-table and heatmap files. |
| 2026-05-24 | `pytest tests/test_local_sdtw.py` | `9 passed`; includes Torch CPU parity with NumPy projection-affinity Local-SDTW. |
| 2026-05-24 | Tiny CLI smoke with `scripts/06_run_local_sdtw.py --backend torch_cpu` | Wrote temporary `/tmp/local_sdtw_torch_cpu_smoke*.csv`; CSV includes backend/device/dtype fields. |
| 2026-05-24 | Full MSR Local-SDTW CUDA sweep | Wrote 855 rows to `results/raw/local_subspace_dtw_msr_action3d.csv` and summary CSV. |
| 2026-05-25 | `python scripts/08_generate_tables.py --table local_sdtw` | Wrote `results/tables/local_subspace_dtw_best_msr_action3d.csv` and `results/tables/global_vs_local_subspace_msr_action3d.csv`. |
| 2026-05-25 | `python scripts/12_generate_figures.py --figure local_sdtw_heatmaps` | Wrote Local-SDTW accuracy and runtime heatmaps. |
| 2026-05-25 | `python scripts/08_generate_tables.py --table q_sdtw_main` | Wrote `results/tables/q_sdtw_main_results.csv`. |
| 2026-05-26 | Full global projection-affinity sweep | Wrote 45 rows to `results/raw/global_subspace_affinity.csv` and summary CSV. |
| 2026-05-26 | `python scripts/08_generate_tables.py --table local_sdtw` and `--table q_sdtw_main` | Regenerated comparison tables with the full global projection-affinity row. |
| 2026-05-26 | `python -m py_compile scripts/07_run_swap_global_convergence.py scripts/12_generate_figures.py` | Passed. |
| 2026-05-26 | SWAP global convergence tiny smoke run | Wrote temporary `/tmp/swap_global_convergence_smoke*.csv`; MAE decreased from `0.080347` at 64 shots to `0.057049` at 128 shots on the tiny subset. |
| 2026-05-26 | `python scripts/12_generate_figures.py --figure swap_global_convergence` on smoke summary | Wrote temporary `/tmp/swap_global_mae_smoke.png`, `/tmp/swap_global_accuracy_smoke.png`, and `/tmp/swap_global_agreement_smoke.png`. |
| 2026-05-26 | `pytest tests/test_quantum_overlap.py tests/test_local_sdtw.py` | `18 passed, 2 skipped` |
| 2026-05-26 | `python scripts/12_generate_figures.py --figure local_sdtw_heatmaps` with `/tmp` outputs | Passed; wrote temporary Local-SDTW accuracy/runtime heatmaps. |
| 2026-05-26 | Formal SWAP global convergence small-subset sweep | Wrote 90 rows to `results/raw/swap_global_exact_comparison_msr_action3d.csv` and 18 summary rows to `results/tables/swap_global_exact_comparison_summary_msr_action3d.csv`. |
| 2026-05-26 | `python scripts/12_generate_figures.py --figure swap_global_convergence` | Wrote formal SWAP global MAE, accuracy, and agreement figures. |
| 2026-05-26 | `python scripts/08_generate_tables.py --table q_sdtw_main --quantum-global-summary results/tables/swap_global_exact_comparison_summary_msr_action3d.csv` | Refreshed `results/tables/q_sdtw_main_results.csv`; current best SWAP global row is `r=2`, `shots=2048`, accuracy `0.675000`, macro-F1 `0.645524`. |
| 2026-05-26 | `python -m py_compile src/distances/quantum_local_sdtw.py scripts/09_run_swap_local_sdtw.py tests/test_quantum_local_sdtw.py` | Passed. |
| 2026-05-26 | `pytest tests/test_quantum_local_sdtw.py` | `6 passed` |
| 2026-05-26 | Tiny SWAP Local-SDTW CLI smoke with two classes and `/tmp` outputs | Wrote temporary raw, summary, and pair-diagnostic CSVs. |
| 2026-05-26 | `python scripts/12_generate_figures.py --figure swap_local_sdtw_convergence` on smoke summary | Wrote temporary SWAP Local-SDTW MAE, accuracy, and agreement figures. |
| 2026-05-26 | SWAP Local-SDTW repository smoke with three classes | Wrote `results/raw/swap_local_sdtw_smoke.csv`, `results/tables/swap_local_sdtw_smoke_summary.csv`, and `results/raw/swap_local_sdtw_pair_diagnostics_smoke.csv`; distance MAE decreased `0.017510 -> 0.012767 -> 0.005310` for shots `128 -> 256 -> 512`. |
| 2026-05-26 | `pytest tests/test_quantum_local_sdtw.py tests/test_quantum_overlap.py tests/test_local_sdtw.py` | `24 passed, 2 skipped` |
| 2026-05-26 | `pytest` | `54 passed, 2 skipped` |
| 2026-05-26 | SWAP Local-SDTW all-class subset run | Wrote 9 rows to `results/raw/swap_local_sdtw_comparison_msr_action3d.csv`, 3 summary rows to `results/tables/swap_local_sdtw_comparison_summary_msr_action3d.csv`, and 21,600 pair diagnostics to `results/raw/swap_local_sdtw_pair_diagnostics_msr_action3d.csv`. |
| 2026-05-26 | `python scripts/12_generate_figures.py --figure swap_local_sdtw_convergence` | Wrote SWAP Local-SDTW MAE, accuracy, and agreement figures. |
| 2026-05-26 | `python scripts/08_generate_tables.py --table q_sdtw_main --quantum-global-summary results/tables/swap_global_exact_comparison_summary_msr_action3d.csv` | Refreshed `results/tables/q_sdtw_main_results.csv`; current best SWAP Local-SDTW row is `shots=2048`, accuracy `0.750000`, macro-F1 `0.725556`. |
| 2026-05-26 | `python -m py_compile scripts/08_generate_tables.py scripts/12_generate_figures.py scripts/09_run_swap_local_sdtw.py src/distances/quantum_local_sdtw.py` | Passed. |
| 2026-05-27 | SWAP Local-SDTW full MSR split run | Wrote 9 rows to `results/raw/swap_local_sdtw_full_msr_action3d.csv` and 3 summary rows to `results/tables/swap_local_sdtw_full_summary_msr_action3d.csv`; each setting used 291 train and 275 test sequences. |
| 2026-05-27 | `python scripts/12_generate_figures.py --figure swap_local_sdtw_convergence --swap-local-summary-input results/tables/swap_local_sdtw_full_summary_msr_action3d.csv` | Rewrote SWAP Local-SDTW convergence figures from the full split summary. |
| 2026-05-27 | `python scripts/08_generate_tables.py --table q_sdtw_main --quantum-global-summary results/tables/swap_global_exact_comparison_summary_msr_action3d.csv --swap-local-summary results/tables/swap_local_sdtw_full_summary_msr_action3d.csv` | Refreshed `results/tables/q_sdtw_main_results.csv`; current best full-split SWAP Local-SDTW row is `shots=512`, accuracy `0.852121`, macro-F1 `0.838612`. |
| 2026-05-27 | `python -m py_compile scripts/08_generate_tables.py scripts/12_generate_figures.py scripts/09_run_swap_local_sdtw.py src/distances/quantum_local_sdtw.py` | Passed. |

## Decision Log

| Date | Decision | Reason |
|---|---|---|
| 2026-05-24 | Create a separate Q-SDTW status file. | The upgrade is large enough to track separately from the original global canonical-angle status. |
| 2026-05-24 | Keep existing script numbers and reserve `scripts/08_generate_tables.py` for tables. | Avoid collisions with current working scripts. |
| 2026-05-24 | Implement exact Local-SDTW before quantum Local-SDTW. | The quantum method needs an exact local-cost baseline on the same settings. |
| 2026-05-24 | Treat `skipped_sequences` as invalid/frame-filtered sequences, not CLI-limited sequences. | Smoke limits should not look like dataset preprocessing failures. |
| 2026-05-24 | Verify Local-SDTW table/figure tooling against smoke data but write verification artifacts to `/tmp`. | Avoid presenting smoke plots as full-sweep research outputs. |
| 2026-05-24 | Add CUDA acceleration only for the projection-affinity Local-SDTW path first. | This is the Phase 4 main sweep metric; canonical-angle local metrics can stay on the exact NumPy path for ablations. |
| 2026-05-25 | Treat `L=5`, stride `2`, rank `1`, DTW window `0.1` as the current exact Local-SDTW setting for follow-up ablations. | It is the best full-sweep setting by accuracy and macro-F1: `0.847273` accuracy, `0.833853` macro-F1. |
| 2026-05-26 | Formalize Phase 6 as a wrapper around the existing exact-vs-SWAP global comparison code. | Reuses the validated comparison path while adding formal output names, subset-size-aware summaries, and convergence figures. |
| 2026-05-26 | Keep SWAP Local-SDTW pair diagnostics in a separate CSV. | Per-pair local cost-matrix diagnostics are useful but can grow quickly; the main raw CSV stays one row per seed/shot setting. |

## Open Blockers / Inputs Needed

| Blocker | Status | Needed Action |
|---|---|---|
| UTKinect raw data | Blocked | Add dataset files before Phase 8. |
| UTD-MHAD raw data | Blocked | Add dataset files before Phase 9. |
| QAE implementation choice | Deferred | SWAP Local-SDTW subset now exists; decide whether QAE is worth adding after paper/table pass. |
| Formal SWAP global medium-subset run | Optional/Pending | Run Phase 6 command with `train_per_class=5`, `test_per_class=3`, shots through `4096` if we want a larger global-SWAP subset result. |
| SWAP Local-SDTW broader run | Optional/Pending | Consider seeds `0 1 2 3 4` or higher shots if more statistical confidence is needed; use `--no-pair-output` for full split runs. |

## How To Update This File

When a task is completed:

1. Change the task status to `Done`.
2. Add the exact output path.
3. Add a decision-log row for important design choices.
4. Update `Last updated`.
