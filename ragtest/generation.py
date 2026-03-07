"""Generation quality metrics — heuristic-based (no LLM calls required).

These metrics use keyword / token overlap to approximate faithfulness,
relevance, and completeness.  They are fast, deterministic, and require
zero external dependencies.
"""

from __future__ import annotations

import re


def _tokenize(text: str) -> list[str]:
    """Lowercase and split text into alphanumeric tokens.

    Args:
        text: Input string.

    Returns:
        List of lowercased tokens.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def faithfulness_score(answer: str, context: str) -> float:
    """Measure whether the answer only uses information from the context.

    Computed as the fraction of answer tokens that also appear in the context.
    A score of 1.0 means every word in the answer is grounded in the context.

    Args:
        answer: The generated answer.
        context: The retrieved context (concatenated chunks).

    Returns:
        Score in [0.0, 1.0].  Returns 1.0 when the answer is empty.
    """
    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return 1.0
    context_tokens = set(_tokenize(context))
    grounded = sum(1 for t in answer_tokens if t in context_tokens)
    return grounded / len(answer_tokens)


def relevance_score(answer: str, question: str) -> float:
    """Measure whether the answer addresses the question.

    Computed as the fraction of question tokens that appear in the answer.
    Higher means the answer echoes more of the question's key terms.

    Args:
        answer: The generated answer.
        question: The original question.

    Returns:
        Score in [0.0, 1.0].  Returns 1.0 when the question is empty.
    """
    question_tokens = _tokenize(question)
    if not question_tokens:
        return 1.0
    answer_tokens = set(_tokenize(answer))
    matched = sum(1 for t in question_tokens if t in answer_tokens)
    return matched / len(question_tokens)


def completeness_score(answer: str, expected_answer: str) -> float:
    """Measure how much of the expected answer is covered.

    Computed as the fraction of expected-answer tokens found in the actual
    answer.  A score of 1.0 means every keyword in the expected answer
    appears in the generated answer.

    Args:
        answer: The generated answer.
        expected_answer: The ground-truth expected answer.

    Returns:
        Score in [0.0, 1.0].  Returns 1.0 when expected_answer is empty.
    """
    expected_tokens = _tokenize(expected_answer)
    if not expected_tokens:
        return 1.0
    answer_tokens = set(_tokenize(answer))
    covered = sum(1 for t in expected_tokens if t in answer_tokens)
    return covered / len(expected_tokens)


def answer_length_ratio(answer: str, expected_answer: str) -> float:
    """Ratio of answer length to expected answer length.

    Values close to 1.0 indicate similar verbosity.  Values > 1.0 mean the
    answer is longer; values < 1.0 mean shorter.

    Args:
        answer: The generated answer.
        expected_answer: The ground-truth expected answer.

    Returns:
        Non-negative float.  Returns 0.0 when expected_answer is empty.
    """
    expected_len = len(expected_answer.split())
    if expected_len == 0:
        return 0.0
    return len(answer.split()) / expected_len
