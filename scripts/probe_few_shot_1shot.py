#!/usr/bin/env python
"""Pre-flight 1-shot few-shot probe (Step 1.23).

Evaluates Raw DTW, Exact Local-SDTW, and SWAP Local-SDTW in a minimal
K=1, 20-way, 50-episode protocol on MSR Action3D (test split, seed 0).

This is intentionally a one-off script — the proper few-shot protocol
lives in Phase 2 (src/eval/few_shot.py). The goal here is to confirm
that a gap exists between Q-SDTW and classical baselines before
committing to Phase 2.

Usage
-----
    python scripts/probe_few_shot_1shot.py
    python scripts/probe_few_shot_1shot.py --shots 2048 --n-episodes 50
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.validation import load_processed_dataset
from src.distances.dtw import pairwise_dtw_distances
from src.distances.local_sdtw import _extract_local_bases, dtw_distance_from_cost_matrix
from src.eval.metrics import accuracy, macro_f1
from src.eval.result_writer import get_git_commit, utc_timestamp
from src.features.motion_features import apply_feature_mode
from src.quantum.overlap_estimation import SwapTestOverlapEstimator
from src.utils.io import read_yaml
from src.utils.logging import get_logger
from src.utils.paths import ensure_parent_dir, project_path

logger = get_logger("probe_few_shot_1shot")


# ---------------------------------------------------------------------------
# Episode generation
# ---------------------------------------------------------------------------


def generate_episodes(
    labels: np.ndarray,
    *,
    n_way: int,
    k_shot: int,
    n_query: int,
    n_episodes: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Generate random K-shot N-way episodes.

    Each episode is a dict with keys:
        support_idx: list[int]  — indices into the dataset
        support_y:   list[int]
        query_idx:   list[int]
        query_y:     list[int]
        class_map:   dict[original_label -> episode_label 0..N-1]
    """
    unique_classes = np.unique(labels)
    if n_way > len(unique_classes):
        raise ValueError(
            f"n_way={n_way} exceeds number of classes ({len(unique_classes)})."
        )

    # Build per-class index lists
    class_to_indices: dict[int, list[int]] = {
        c: np.where(labels == c)[0].tolist() for c in unique_classes
    }

    episodes = []
    for _ in range(n_episodes):
        chosen_classes = rng.choice(unique_classes, size=n_way, replace=False)
        class_map = {int(c): i for i, c in enumerate(chosen_classes)}

        support_idx, support_y = [], []
        query_idx, query_y = [], []

        for c in chosen_classes:
            pool = class_to_indices[int(c)].copy()
            rng.shuffle(pool)
            needed = k_shot + n_query
            if len(pool) < needed:
                # Sample with replacement if class is too small
                chosen = rng.choice(pool, size=needed, replace=True).tolist()
            else:
                chosen = pool[:needed]
            for idx in chosen[:k_shot]:
                support_idx.append(idx)
                support_y.append(class_map[int(c)])
            for idx in chosen[k_shot:k_shot + n_query]:
                query_idx.append(idx)
                query_y.append(class_map[int(c)])

        episodes.append(
            {
                "support_idx": support_idx,
                "support_y": support_y,
                "query_idx": query_idx,
                "query_y": query_y,
                "class_map": class_map,
            }
        )
    return episodes


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------


def _predict_1nn(dist_matrix: np.ndarray, support_y: list[int]) -> list[int]:
    """1-NN classification from a query × support distance matrix."""
    nearest = np.argmin(dist_matrix, axis=1)
    return [support_y[i] for i in nearest]


def evaluate_raw_dtw(episodes: list[dict], seqs: list[np.ndarray]) -> tuple[float, float]:
    """Evaluate Raw DTW 1-NN on each episode using batched pairwise computation."""
    episode_accs = []
    for ep_i, ep in enumerate(episodes):
        s_idx, s_y = ep["support_idx"], ep["support_y"]
        q_idx, q_y = ep["query_idx"], ep["query_y"]
        support_seqs = [seqs[i] for i in s_idx]
        query_seqs   = [seqs[i] for i in q_idx]
        dist = pairwise_dtw_distances(query_seqs, support_seqs, backend="numpy")
        preds = _predict_1nn(dist, s_y)
        episode_accs.append(accuracy(q_y, preds))
        if (ep_i + 1) % 10 == 0:
            logger.info("  Raw DTW episode %d/%d — running acc=%.4f",
                        ep_i + 1, len(episodes), np.mean(episode_accs))
    return float(np.mean(episode_accs)), float(np.std(episode_accs, ddof=1))


def _build_bases(seqs: list[np.ndarray], indices: list[int],
                 window_size: int, stride: int, rank: int) -> list[np.ndarray]:
    """Extract local subspace bases for a list of indices."""
    bases_list = []
    for idx in indices:
        bases = _extract_local_bases(seqs[idx], window_size=window_size,
                                      stride=stride, rank=rank)
        bases_list.append(bases)  # shape: (M, D, rank)
    return bases_list


def _local_sdtw_cost_matrix_from_bases(
    q_bases: np.ndarray,
    s_bases: np.ndarray,
    overlap_fn,
) -> float:
    """Compute Local-SDTW distance between two precomputed basis stacks."""
    M, N = len(q_bases), len(s_bases)
    rank = q_bases.shape[-1]
    cost = np.empty((M, N), dtype=np.float64)
    for i in range(M):
        for j in range(N):
            # projection affinity = sum of squared overlaps / rank
            total = 0.0
            for a in range(rank):
                for b in range(rank):
                    total += overlap_fn(q_bases[i, :, a], s_bases[j, :, b])
            affinity = total / rank
            cost[i, j] = 1.0 - affinity
    return dtw_distance_from_cost_matrix(cost, normalize_by_path_length=True)


def evaluate_local_sdtw(
    episodes: list[dict],
    seqs: list[np.ndarray],
    *,
    window_size: int,
    stride: int,
    rank: int,
    overlap_fn,
) -> tuple[float, float]:
    """Evaluate Local-SDTW 1-NN on each episode."""
    all_needed = set()
    for ep in episodes:
        all_needed.update(ep["support_idx"])
        all_needed.update(ep["query_idx"])
    all_needed = sorted(all_needed)

    logger.info("Pre-computing %d bases (window=%d, stride=%d, rank=%d)",
                len(all_needed), window_size, stride, rank)
    bases_cache: dict[int, np.ndarray] = {}
    for idx in all_needed:
        bases_cache[idx] = _extract_local_bases(
            seqs[idx], window_size=window_size, stride=stride, rank=rank
        )

    episode_accs = []
    for ep_i, ep in enumerate(episodes):
        s_idx, s_y = ep["support_idx"], ep["support_y"]
        q_idx, q_y = ep["query_idx"], ep["query_y"]
        dist = np.zeros((len(q_idx), len(s_idx)))
        for qi, qidx in enumerate(q_idx):
            for si, sidx in enumerate(s_idx):
                dist[qi, si] = _local_sdtw_cost_matrix_from_bases(
                    bases_cache[qidx], bases_cache[sidx], overlap_fn
                )
        preds = _predict_1nn(dist, s_y)
        episode_accs.append(accuracy(q_y, preds))
        if (ep_i + 1) % 10 == 0:
            logger.info("  episode %d/%d — running acc=%.4f",
                        ep_i + 1, len(episodes), np.mean(episode_accs))
    return float(np.mean(episode_accs)), float(np.std(episode_accs, ddof=1))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--n-way",     type=int, default=20)
    p.add_argument("--k-shot",    type=int, default=1)
    p.add_argument("--n-query",   type=int, default=10)
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--window-size", type=int, default=15)
    p.add_argument("--stride",    type=int, default=5)
    p.add_argument("--rank",      type=int, default=2)
    p.add_argument("--shots",     type=int, default=2048)
    p.add_argument("--feature-mode", default="position_velocity")
    p.add_argument("--output",    default="results/raw/probe_1shot.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("1-shot probe — %d-way, %d episodes, seed=%d, feature=%s",
                args.n_way, args.n_episodes, args.seed, args.feature_mode)

    dataset_config = read_yaml(project_path("configs/msr_action3d.yaml"))
    dataset = load_processed_dataset(
        project_path(dataset_config["dataset"]["processed_dir"])
    )

    # Use all sequences (both train and test) for episode sampling
    # The cross-subject split is for full-data eval; for few-shot we pool all.
    # For this probe, use all 566 sequences.
    all_seqs_raw = dataset.sequences
    all_labels   = dataset.labels

    seqs = [apply_feature_mode(X, args.feature_mode, dataset_config)
            for X in all_seqs_raw]

    rng = np.random.default_rng(args.seed)
    episodes = generate_episodes(
        all_labels,
        n_way=args.n_way,
        k_shot=args.k_shot,
        n_query=args.n_query,
        n_episodes=args.n_episodes,
        rng=rng,
    )
    logger.info("Generated %d episodes", len(episodes))

    git_commit = get_git_commit()
    timestamp  = utc_timestamp()

    rows = []

    # ── Raw DTW ──────────────────────────────────────────────────────────────
    logger.info("Evaluating Raw DTW …")
    t0 = perf_counter()
    acc_dtw, std_dtw = evaluate_raw_dtw(episodes, seqs)
    rt_dtw = perf_counter() - t0
    logger.info("Raw DTW: acc=%.4f ± %.4f  (%.1fs)", acc_dtw, std_dtw, rt_dtw)
    rows.append({
        "method": "raw_dtw",
        "feature_mode": args.feature_mode,
        "k_shot": args.k_shot,
        "n_way": args.n_way,
        "n_episodes": args.n_episodes,
        "seed": args.seed,
        "accuracy_mean": round(acc_dtw, 6),
        "accuracy_std": round(std_dtw, 6),
        "runtime_sec": round(rt_dtw, 3),
        "parameters": f"window=none",
        "git_commit": git_commit,
        "timestamp": timestamp,
    })

    # ── Exact Local-SDTW ─────────────────────────────────────────────────────
    exact_overlap = lambda u, v: float((u @ v) ** 2)
    logger.info("Evaluating Exact Local-SDTW (w=%d, s=%d, r=%d) …",
                args.window_size, args.stride, args.rank)
    t0 = perf_counter()
    acc_exact, std_exact = evaluate_local_sdtw(
        episodes, seqs,
        window_size=args.window_size, stride=args.stride, rank=args.rank,
        overlap_fn=exact_overlap,
    )
    rt_exact = perf_counter() - t0
    logger.info("Exact Local-SDTW: acc=%.4f ± %.4f  (%.1fs)", acc_exact, std_exact, rt_exact)
    rows.append({
        "method": "local_sdtw_exact",
        "feature_mode": args.feature_mode,
        "k_shot": args.k_shot,
        "n_way": args.n_way,
        "n_episodes": args.n_episodes,
        "seed": args.seed,
        "accuracy_mean": round(acc_exact, 6),
        "accuracy_std": round(std_exact, 6),
        "runtime_sec": round(rt_exact, 3),
        "parameters": f"window={args.window_size},stride={args.stride},rank={args.rank}",
        "git_commit": git_commit,
        "timestamp": timestamp,
    })

    # ── SWAP Local-SDTW ───────────────────────────────────────────────────────
    estimator = SwapTestOverlapEstimator(shots=args.shots, simulator="sampling",
                                         seed=args.seed, cache=True)
    swap_overlap = lambda u, v: estimator.estimate(u, v).overlap_squared
    logger.info("Evaluating SWAP Local-SDTW (shots=%d, w=%d, s=%d, r=%d) …",
                args.shots, args.window_size, args.stride, args.rank)
    t0 = perf_counter()
    acc_swap, std_swap = evaluate_local_sdtw(
        episodes, seqs,
        window_size=args.window_size, stride=args.stride, rank=args.rank,
        overlap_fn=swap_overlap,
    )
    rt_swap = perf_counter() - t0
    logger.info("SWAP Local-SDTW: acc=%.4f ± %.4f  (%.1fs)", acc_swap, std_swap, rt_swap)
    rows.append({
        "method": "local_sdtw_swap",
        "feature_mode": args.feature_mode,
        "k_shot": args.k_shot,
        "n_way": args.n_way,
        "n_episodes": args.n_episodes,
        "seed": args.seed,
        "accuracy_mean": round(acc_swap, 6),
        "accuracy_std": round(std_swap, 6),
        "runtime_sec": round(rt_swap, 3),
        "parameters": f"window={args.window_size},stride={args.stride},rank={args.rank},shots={args.shots}",
        "git_commit": git_commit,
        "timestamp": timestamp,
    })

    # ── Write results ─────────────────────────────────────────────────────────
    output_path = ensure_parent_dir(project_path(args.output))
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows to %s", len(rows), output_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== 1-Shot Probe Results ===")
    print(f"{'Method':<25} {'Acc Mean':>10} {'Acc Std':>9} {'Gap vs DTW':>12}")
    print("-" * 60)
    for r in rows:
        gap = r["accuracy_mean"] - acc_dtw if r["method"] != "raw_dtw" else 0.0
        gap_str = f"{gap:+.4f}" if r["method"] != "raw_dtw" else "    —"
        print(f"{r['method']:<25} {r['accuracy_mean']:>10.4f} {r['accuracy_std']:>9.4f} {gap_str:>12}")


if __name__ == "__main__":
    main()
