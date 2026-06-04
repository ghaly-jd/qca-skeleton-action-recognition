"""Tests for FewShotProtocol (Step 2.2)."""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.few_shot import EvaluationResult, Episode, FewShotMethod, FewShotProtocol


# ---------------------------------------------------------------------------
# Synthetic dataset helpers
# ---------------------------------------------------------------------------


def _make_dataset(
    n_classes: int = 20,
    samples_per_class: int = 30,
    T: int = 15,
    D: int = 10,
    seed: int = 42,
) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    for cls in range(n_classes):
        for _ in range(samples_per_class):
            sequences.append(rng.standard_normal((T, D)))
            labels.append(cls)
    return sequences, np.asarray(labels, dtype=int)


# ---------------------------------------------------------------------------
# Determinism: same seed → identical episodes
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_episodes(self):
        seqs, labels = _make_dataset()
        proto_a = FewShotProtocol(n_way=5, k_shot=1, n_query=5, n_episodes=20, episode_seed=0)
        proto_b = FewShotProtocol(n_way=5, k_shot=1, n_query=5, n_episodes=20, episode_seed=0)

        eps_a = proto_a.generate_episodes((seqs, labels))
        eps_b = proto_b.generate_episodes((seqs, labels))

        assert len(eps_a) == len(eps_b)
        for ea, eb in zip(eps_a, eps_b):
            assert ea.support_y == eb.support_y
            assert ea.query_y == eb.query_y
            assert ea.original_classes == eb.original_classes
            for xa, xb in zip(ea.support_X, eb.support_X):
                np.testing.assert_array_equal(xa, xb)

    def test_different_seeds_different_episodes(self):
        seqs, labels = _make_dataset()
        proto_a = FewShotProtocol(n_way=5, k_shot=1, n_query=5, n_episodes=20, episode_seed=0)
        proto_b = FewShotProtocol(n_way=5, k_shot=1, n_query=5, n_episodes=20, episode_seed=1)

        eps_a = proto_a.generate_episodes((seqs, labels))
        eps_b = proto_b.generate_episodes((seqs, labels))

        # Classes drawn should differ in at least one episode
        all_same = all(ea.original_classes == eb.original_classes for ea, eb in zip(eps_a, eps_b))
        assert not all_same, "Different seeds should produce different episodes"


# ---------------------------------------------------------------------------
# Support / query are disjoint
# ---------------------------------------------------------------------------


class TestDisjoint:
    def test_support_query_disjoint(self):
        seqs, labels = _make_dataset(samples_per_class=30)
        proto = FewShotProtocol(n_way=5, k_shot=3, n_query=5, n_episodes=10, episode_seed=7)
        episodes = proto.generate_episodes((seqs, labels))

        # Build a lookup: sequence id by array identity (id())
        seq_id_map = {id(s): i for i, s in enumerate(seqs)}

        for ep in episodes:
            support_ids = {seq_id_map.get(id(x)) for x in ep.support_X}
            query_ids = {seq_id_map.get(id(x)) for x in ep.query_X}
            overlap = support_ids & query_ids
            assert len(overlap) == 0, f"Support and query share {len(overlap)} sequences"


# ---------------------------------------------------------------------------
# All N classes represented
# ---------------------------------------------------------------------------


class TestClassCoverage:
    def test_all_classes_in_support(self):
        seqs, labels = _make_dataset()
        n_way = 5
        k_shot = 2
        proto = FewShotProtocol(n_way=n_way, k_shot=k_shot, n_query=10, n_episodes=20, episode_seed=3)
        episodes = proto.generate_episodes((seqs, labels))

        for ep in episodes:
            classes_in_support = set(ep.support_y)
            assert classes_in_support == set(range(n_way)), (
                f"Support does not cover all {n_way} episode classes: {classes_in_support}"
            )

    def test_all_classes_in_query(self):
        seqs, labels = _make_dataset()
        n_way = 5
        proto = FewShotProtocol(n_way=n_way, k_shot=1, n_query=5, n_episodes=20, episode_seed=4)
        episodes = proto.generate_episodes((seqs, labels))

        for ep in episodes:
            classes_in_query = set(ep.query_y)
            assert classes_in_query == set(range(n_way)), (
                f"Query does not cover all {n_way} episode classes: {classes_in_query}"
            )


# ---------------------------------------------------------------------------
# Episode count
# ---------------------------------------------------------------------------


class TestEpisodeCount:
    def test_episode_count(self):
        seqs, labels = _make_dataset()
        n_episodes = 37
        proto = FewShotProtocol(n_way=5, k_shot=1, n_query=5, n_episodes=n_episodes, episode_seed=0)
        episodes = proto.generate_episodes((seqs, labels))
        assert len(episodes) == n_episodes

    def test_support_query_sizes(self):
        seqs, labels = _make_dataset()
        n_way, k_shot, n_query = 5, 3, 7
        proto = FewShotProtocol(n_way=n_way, k_shot=k_shot, n_query=n_query, n_episodes=10, episode_seed=0)
        episodes = proto.generate_episodes((seqs, labels))

        for ep in episodes:
            assert len(ep.support_X) == n_way * k_shot
            assert len(ep.support_y) == n_way * k_shot
            assert len(ep.query_X) == n_way * n_query
            assert len(ep.query_y) == n_way * n_query
            assert len(ep.original_classes) == n_way


# ---------------------------------------------------------------------------
# Dataset interface: accepts (seqs, labels) tuple and duck-typed object
# ---------------------------------------------------------------------------


class TestDatasetInterface:
    def test_tuple_interface(self):
        seqs, labels = _make_dataset()
        proto = FewShotProtocol(n_way=5, k_shot=1, n_query=5, n_episodes=5, episode_seed=0)
        episodes = proto.generate_episodes((seqs, labels))
        assert len(episodes) == 5

    def test_duck_typed_dataset(self):
        class FakeDataset:
            def __init__(self, seqs, labels):
                self.sequences = seqs
                self.labels = labels

        seqs, labels = _make_dataset()
        proto = FewShotProtocol(n_way=5, k_shot=1, n_query=5, n_episodes=5, episode_seed=0)
        ds = FakeDataset(seqs, labels)
        episodes = proto.generate_episodes(ds)
        assert len(episodes) == 5

    def test_tuple_and_dataset_agree(self):
        seqs, labels = _make_dataset()

        class FakeDataset:
            def __init__(self, s, l):
                self.sequences = s
                self.labels = l

        proto = FewShotProtocol(n_way=5, k_shot=1, n_query=5, n_episodes=10, episode_seed=99)
        eps_tuple = proto.generate_episodes((seqs, labels))
        eps_ds = proto.generate_episodes(FakeDataset(seqs, labels))

        for ea, eb in zip(eps_tuple, eps_ds):
            assert ea.support_y == eb.support_y
            assert ea.original_classes == eb.original_classes


# ---------------------------------------------------------------------------
# evaluate() method
# ---------------------------------------------------------------------------


class _NearestSupportMethod:
    """Trivial 1-NN by array distance — used to test evaluate()."""

    def predict(
        self,
        support_X: list[np.ndarray],
        support_y: list[int],
        query_X: list[np.ndarray],
    ) -> list[int]:
        preds = []
        for qx in query_X:
            dists = [float(np.linalg.norm(qx - sx)) for sx in support_X]
            preds.append(support_y[int(np.argmin(dists))])
        return preds


class _PerfectMethod:
    """Always returns the correct query labels (simulates 100% accuracy)."""

    def predict(
        self,
        support_X: list[np.ndarray],
        support_y: list[int],
        query_X: list[np.ndarray],
    ) -> list[int]:
        # For test only: we stash query_y on the episode externally
        return self._query_y  # type: ignore[attr-defined]


class TestEvaluate:
    def test_evaluate_returns_correct_structure(self):
        seqs, labels = _make_dataset()
        proto = FewShotProtocol(n_way=5, k_shot=1, n_query=5, n_episodes=10, episode_seed=0)
        episodes = proto.generate_episodes((seqs, labels))
        result = proto.evaluate(_NearestSupportMethod(), episodes)

        assert isinstance(result, EvaluationResult)
        assert len(result.per_episode_accuracy) == 10
        assert len(result.per_episode_macro_f1) == 10
        assert 0.0 <= result.mean_accuracy <= 1.0
        assert result.std_accuracy >= 0.0

    def test_evaluate_aggregate_std_zero_for_identical_accuracy(self):
        """If all episodes return the same accuracy, std should be 0."""
        seqs, labels = _make_dataset(n_classes=5, samples_per_class=20)

        class ConstantMethod:
            def predict(self, support_X, support_y, query_X):
                # Return first class for everything (constant prediction)
                return [support_y[0]] * len(query_X)

        proto = FewShotProtocol(n_way=5, k_shot=1, n_query=5, n_episodes=5, episode_seed=0)
        episodes = proto.generate_episodes((seqs, labels))
        result = proto.evaluate(ConstantMethod(), episodes)

        assert result.std_accuracy == pytest.approx(0.0, abs=1e-10)

    def test_protocol_compatible_across_methods(self):
        """Identical seeds → episodes are the same → paired comparison is valid."""
        seqs, labels = _make_dataset()
        proto1 = FewShotProtocol(n_way=5, k_shot=1, n_query=5, n_episodes=8, episode_seed=42)
        proto2 = FewShotProtocol(n_way=5, k_shot=1, n_query=5, n_episodes=8, episode_seed=42)

        eps1 = proto1.generate_episodes((seqs, labels))
        eps2 = proto2.generate_episodes((seqs, labels))

        method = _NearestSupportMethod()
        r1 = proto1.evaluate(method, eps1)
        r2 = proto2.evaluate(method, eps2)

        np.testing.assert_array_almost_equal(
            r1.per_episode_accuracy, r2.per_episode_accuracy
        )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_n_way_too_large(self):
        seqs, labels = _make_dataset(n_classes=5)
        proto = FewShotProtocol(n_way=10, k_shot=1, n_query=5, n_episodes=5, episode_seed=0)
        with pytest.raises(ValueError, match="n_way"):
            proto.generate_episodes((seqs, labels))
