"""Few-shot episode generation and evaluation (Step 2.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from src.eval.metrics import accuracy as _accuracy, macro_f1 as _macro_f1


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    """One N-way K-shot episode.

    Labels ``support_y`` and ``query_y`` are remapped to episode-local
    indices 0..n_way-1.  ``original_classes[i]`` gives the original dataset
    class ID for episode class ``i``.
    """

    support_X: list[np.ndarray]    # k_shot * n_way sequences, shape (T_i, D)
    support_y: list[int]           # k_shot * n_way labels in 0..n_way-1
    query_X: list[np.ndarray]      # n_query * n_way sequences
    query_y: list[int]             # n_query * n_way labels in 0..n_way-1
    original_classes: list[int]    # original dataset class IDs, length n_way


@dataclass
class EvaluationResult:
    """Aggregated result from evaluating a method over many episodes."""

    per_episode_accuracy: list[float] = field(default_factory=list)
    per_episode_macro_f1: list[float] = field(default_factory=list)

    @property
    def mean_accuracy(self) -> float:
        return float(np.mean(self.per_episode_accuracy))

    @property
    def std_accuracy(self) -> float:
        n = len(self.per_episode_accuracy)
        return float(np.std(self.per_episode_accuracy, ddof=1)) if n > 1 else 0.0

    @property
    def mean_macro_f1(self) -> float:
        return float(np.mean(self.per_episode_macro_f1))

    @property
    def std_macro_f1(self) -> float:
        n = len(self.per_episode_macro_f1)
        return float(np.std(self.per_episode_macro_f1, ddof=1)) if n > 1 else 0.0

    def __len__(self) -> int:
        return len(self.per_episode_accuracy)


# ---------------------------------------------------------------------------
# Method protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class FewShotMethod(Protocol):
    """Interface every few-shot classifier must implement."""

    def predict(
        self,
        support_X: list[np.ndarray],
        support_y: list[int],
        query_X: list[np.ndarray],
    ) -> list[int]:
        """Return predicted episode-local labels for each element in query_X."""
        ...


# ---------------------------------------------------------------------------
# Dataset protocol
# ---------------------------------------------------------------------------


class _DatasetLike(Protocol):
    """Duck-typed dataset with sequences and labels."""

    sequences: list[np.ndarray]
    labels: Any  # np.ndarray or list[int]


# ---------------------------------------------------------------------------
# FewShotProtocol
# ---------------------------------------------------------------------------


class FewShotProtocol:
    """Deterministic N-way K-shot episode generator and evaluator.

    Parameters
    ----------
    n_way:
        Number of classes sampled per episode.
    k_shot:
        Number of support samples per class per episode.
    n_query:
        Number of query samples per class per episode.
    n_episodes:
        Total episodes to generate per ``generate_episodes`` call.
    episode_seed:
        Seed for the episode RNG.  Same seed → identical episodes regardless
        of which method will be evaluated (needed for valid paired comparison).
    """

    def __init__(
        self,
        n_way: int,
        k_shot: int,
        n_query: int,
        n_episodes: int,
        episode_seed: int,
    ) -> None:
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        self.episode_seed = episode_seed

    # ------------------------------------------------------------------
    # Episode generation
    # ------------------------------------------------------------------

    def generate_episodes(
        self,
        dataset: _DatasetLike | tuple[list[np.ndarray], Any],
    ) -> list[Episode]:
        """Sample ``n_episodes`` episodes from *dataset*.

        *dataset* may be any object with ``.sequences`` and ``.labels``
        attributes (e.g. ``ProcessedDataset``), or a 2-tuple
        ``(sequences, labels)``.

        Labels in returned episodes are remapped to episode-local indices
        0..n_way-1.  Sampling is without replacement per class within each
        episode; if a class has fewer than ``k_shot + n_query`` samples it
        is sampled with replacement (logged as a warning, should not happen
        on MSR with k≤5 + n_query=15 and ≥20 samples per class).
        """
        if isinstance(dataset, tuple):
            sequences, labels = dataset
        else:
            sequences = dataset.sequences
            labels = dataset.labels

        labels_arr = np.asarray(labels, dtype=int)
        unique_classes = np.unique(labels_arr)
        n_classes = len(unique_classes)

        if n_classes < self.n_way:
            raise ValueError(
                f"n_way={self.n_way} > number of classes ({n_classes})."
            )

        min_samples = self.k_shot + self.n_query
        class_to_indices: dict[int, list[int]] = {}
        for cls in unique_classes:
            idxs = np.where(labels_arr == cls)[0].tolist()
            class_to_indices[int(cls)] = idxs

        rng = np.random.default_rng(self.episode_seed)
        episodes: list[Episode] = []

        for _ in range(self.n_episodes):
            chosen = rng.choice(unique_classes, size=self.n_way, replace=False)
            # Map original class id → local episode index 0..n_way-1
            class_map = {int(c): i for i, c in enumerate(chosen)}

            support_X: list[np.ndarray] = []
            support_y: list[int] = []
            query_X: list[np.ndarray] = []
            query_y: list[int] = []

            for cls in chosen:
                pool = class_to_indices[int(cls)].copy()
                replace = len(pool) < min_samples
                sampled = rng.choice(pool, size=min_samples, replace=replace).tolist()

                for idx in sampled[: self.k_shot]:
                    support_X.append(sequences[idx])
                    support_y.append(class_map[int(cls)])

                for idx in sampled[self.k_shot :]:
                    query_X.append(sequences[idx])
                    query_y.append(class_map[int(cls)])

            episodes.append(
                Episode(
                    support_X=support_X,
                    support_y=support_y,
                    query_X=query_X,
                    query_y=query_y,
                    original_classes=[int(c) for c in chosen],
                )
            )

        return episodes

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        method: FewShotMethod,
        episodes: list[Episode],
    ) -> EvaluationResult:
        """Evaluate *method* on *episodes* and return per-episode metrics.

        *method* must implement ``predict(support_X, support_y, query_X)``.
        Episodes must have been generated by this or an identical protocol
        so that per-episode indices are aligned across methods.
        """
        per_acc: list[float] = []
        per_f1: list[float] = []

        for episode in episodes:
            y_pred = method.predict(
                episode.support_X, episode.support_y, episode.query_X
            )
            y_true = np.asarray(episode.query_y, dtype=int)
            y_pred_arr = np.asarray(y_pred, dtype=int)
            per_acc.append(_accuracy(y_true, y_pred_arr))
            per_f1.append(_macro_f1(y_true, y_pred_arr))

        return EvaluationResult(
            per_episode_accuracy=per_acc,
            per_episode_macro_f1=per_f1,
        )
