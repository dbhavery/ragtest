"""Tests for ragtest.generation — generation quality metrics."""

import pytest

from ragtest.generation import (
    answer_length_ratio,
    completeness_score,
    faithfulness_score,
    relevance_score,
)


class TestFaithfulness:
    def test_fully_grounded(self) -> None:
        answer = "reset the password and go to settings"
        context = "To reset the password, go to settings and click reset."
        score = faithfulness_score(answer, context)
        assert score == 1.0

    def test_partially_grounded(self) -> None:
        answer = "The password is reset in the dashboard."
        context = "To reset the password, go to settings."
        score = faithfulness_score(answer, context)
        # "dashboard" is not in context, so score < 1.0
        assert 0.0 < score < 1.0

    def test_empty_answer(self) -> None:
        assert faithfulness_score("", "some context") == 1.0

    def test_no_overlap(self) -> None:
        assert faithfulness_score("xyz abc", "completely different text") == 0.0


class TestRelevance:
    def test_full_relevance(self) -> None:
        answer = "How do I reset my password? You go to settings."
        question = "How do I reset my password?"
        score = relevance_score(answer, question)
        # All question tokens appear in the answer
        assert score == 1.0

    def test_empty_question(self) -> None:
        assert relevance_score("some answer", "") == 1.0


class TestCompleteness:
    def test_full_coverage(self) -> None:
        answer = "Go to Settings then Security then Reset Password."
        expected = "Go to Settings > Security > Reset Password"
        score = completeness_score(answer, expected)
        assert score == 1.0

    def test_partial_coverage(self) -> None:
        answer = "Go to Settings."
        expected = "Go to Settings > Security > Reset Password"
        score = completeness_score(answer, expected)
        assert 0.0 < score < 1.0

    def test_empty_expected(self) -> None:
        assert completeness_score("anything", "") == 1.0


class TestAnswerLengthRatio:
    def test_same_length(self) -> None:
        answer = "one two three"
        expected = "four five six"
        assert answer_length_ratio(answer, expected) == pytest.approx(1.0)

    def test_double_length(self) -> None:
        answer = "one two three four five six"
        expected = "one two three"
        assert answer_length_ratio(answer, expected) == pytest.approx(2.0)

    def test_empty_expected(self) -> None:
        assert answer_length_ratio("some answer", "") == 0.0
