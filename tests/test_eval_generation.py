"""Tests for eval_generation.py's offline helpers."""

from collections import Counter

from evaluation import TEST_QUERIES
from eval_generation import _is_daily_limit, stratified_sample


def test_daily_limit_detected():
    msg = (
        "Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` "
        "... on tokens per day (TPD): Limit 200000, Used 199798', 'code': 'rate_limit_exceeded'}}"
    )
    assert _is_daily_limit(Exception(msg))


def test_per_minute_limit_is_not_daily():
    msg = "Rate limit reached ... on tokens per minute (TPM): Limit 8000, Used 6002"
    assert not _is_daily_limit(Exception(msg))


def test_full_run_is_interleaved_across_cities():
    full = stratified_sample(TEST_QUERIES, 0)
    assert sorted(q["qid"] for q in full) == sorted(q["qid"] for q in TEST_QUERIES)

    assert len(Counter(q.get("city") for q in full[:12])) >= 10


def test_limit_sample_is_deterministic():
    assert stratified_sample(TEST_QUERIES, 15, seed=7) == stratified_sample(
        TEST_QUERIES, 15, seed=7
    )


def test_make_judge_routes_by_model_id():
    from eval_generation import ClaudeJudge, GroqJudge, make_judge

    assert isinstance(make_judge("claude-sonnet-5"), ClaudeJudge)
    assert isinstance(make_judge("openai/gpt-oss-120b"), GroqJudge)
