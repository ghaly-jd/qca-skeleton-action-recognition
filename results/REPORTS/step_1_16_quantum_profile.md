# Step 1.16 Quantum Simulation Profile

Date: 2026-05-27

## Goal

Make the global quantum-estimated subspace-affinity sweep tractable before the Phase 1 full-data quantum runs.

## Baseline

Command:

```bash
python scripts/04_run_quantum_angles_sim.py --dataset msr_action3d --r-values 2 --shots 1024 --seeds 0 --subset false --limit-train 60 --limit-test 40 --output /tmp/quantum_step_1_16_before_60x40.csv --summary-output /tmp/quantum_step_1_16_before_60x40_summary.csv
```

Result row:

```text
train=60, test=40, rank=2, shots=1024, runtime_sec=1.09
```

Initial cProfile on a 10x10 subset showed scalar overlap estimation dominating the classification path:

```text
pairwise_quantum_subspace_affinity_distances: 0.064s
quantum_subspace_affinity_distance: 100 calls
estimate_subspace_squared_overlaps: 100 calls
SwapTestOverlapEstimator.estimate: 400 calls
estimate_squared_overlap_swap_test: 400 calls
```

## Changes

- Added vectorized basis-stack SWAP sampling in `src/quantum/overlap_estimation.py`.
- Replaced per-pair/per-basis-vector loops in `pairwise_quantum_subspace_affinity_distances` with one `np.einsum` overlap tensor and one vectorized binomial draw.
- Added a per-sequence `SubspaceBasisCache` in `scripts/04_run_quantum_angles_sim.py`.
- Added `--num-workers` seed-level multiprocessing support for `scripts/04_run_quantum_angles_sim.py`.

## Verification

Required 60x40 timing:

```bash
python -m cProfile -o /tmp/quantum_step_1_16_after.prof scripts/04_run_quantum_angles_sim.py --dataset msr_action3d --r-values 2 --shots 1024 --seeds 0 --subset false --limit-train 60 --limit-test 40 --output /tmp/quantum_step_1_16_after_60x40.csv --summary-output /tmp/quantum_step_1_16_after_60x40_summary.csv
```

Result row:

```text
train=60, test=40, rank=2, shots=1024, runtime_sec=0.085687
num_overlap_estimates=9600
```

Full-data one-seed timing:

```bash
python scripts/04_run_quantum_angles_sim.py --dataset msr_action3d --r-values 2 --shots 1024 --seeds 0 --subset false --output /tmp/quantum_step_1_16_full_seed0.csv --summary-output /tmp/quantum_step_1_16_full_seed0_summary.csv
```

Result row:

```text
train=291, test=274, rank=2, shots=1024, runtime_sec=0.973906
num_overlap_estimates=318936
```

Multiprocessing smoke:

```bash
python scripts/04_run_quantum_angles_sim.py --dataset msr_action3d --r-values 2 --shots 128 --seeds 0 1 --subset false --limit-train 10 --limit-test 10 --num-workers 2 --output /tmp/quantum_step_1_16_workers.csv --summary-output /tmp/quantum_step_1_16_workers_summary.csv
```

Full tests:

```text
119 passed, 2 skipped, 2 warnings
```

## Outcome

The required 60x40 subset gate is well below 60 seconds, and the full-data one-seed rank=2 shots=1024 gate is well below 30 minutes on a single CPU core.
