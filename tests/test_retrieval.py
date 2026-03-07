"""Tests for ragtest.retrieval — IR metric functions."""

from ragtest.retrieval import mrr, ndcg_at_k, precision_at_k, recall_at_k


class TestRecallAtK:
    def test_perfect_recall(self) -> None:
        retrieved = ["a", "b", "c"]
        relevant = ["a", "b"]
        assert recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_partial_recall(self) -> None:
        retrieved = ["a", "x", "y"]
        relevant = ["a", "b"]
        assert recall_at_k(retrieved, relevant, k=3) == 0.5

    def test_zero_recall(self) -> None:
        retrieved = ["x", "y", "z"]
        relevant = ["a", "b"]
        assert recall_at_k(retrieved, relevant, k=3) == 0.0

    def test_k_limits_results(self) -> None:
        retrieved = ["x", "a", "b"]
        relevant = ["a", "b"]
        # Only top-1 considered; "a" is at position 2
        assert recall_at_k(retrieved, relevant, k=1) == 0.0

    def test_empty_relevant(self) -> None:
        assert recall_at_k(["a", "b"], [], k=5) == 0.0


class TestPrecisionAtK:
    def test_perfect_precision(self) -> None:
        retrieved = ["a", "b"]
        relevant = ["a", "b", "c"]
        assert precision_at_k(retrieved, relevant, k=2) == 1.0

    def test_half_precision(self) -> None:
        retrieved = ["a", "x"]
        relevant = ["a", "b"]
        assert precision_at_k(retrieved, relevant, k=2) == 0.5

    def test_zero_k(self) -> None:
        assert precision_at_k(["a"], ["a"], k=0) == 0.0


class TestMRR:
    def test_first_position(self) -> None:
        assert mrr(["a", "b", "c"], ["a"]) == 1.0

    def test_second_position(self) -> None:
        assert mrr(["x", "a", "c"], ["a"]) == 0.5

    def test_third_position(self) -> None:
        assert mrr(["x", "y", "a"], ["a"]) == pytest.approx(1.0 / 3.0)

    def test_not_found(self) -> None:
        assert mrr(["x", "y", "z"], ["a"]) == 0.0


class TestNDCGAtK:
    def test_perfect_ranking(self) -> None:
        # All relevant docs at the top
        retrieved = ["a", "b", "x", "y"]
        relevant = ["a", "b"]
        assert ndcg_at_k(retrieved, relevant, k=4) == 1.0

    def test_imperfect_ranking(self) -> None:
        # Relevant doc at position 2 instead of 1
        retrieved = ["x", "a"]
        relevant = ["a"]
        score = ndcg_at_k(retrieved, relevant, k=2)
        # IDCG = 1/log2(2) = 1.0, DCG = 1/log2(3) ~= 0.6309
        assert 0.6 < score < 0.65

    def test_zero_k(self) -> None:
        assert ndcg_at_k(["a"], ["a"], k=0) == 0.0

    def test_empty_relevant(self) -> None:
        assert ndcg_at_k(["a", "b"], [], k=5) == 0.0


# pytest is imported at module level for approx
import pytest  # noqa: E402
