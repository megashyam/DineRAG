"""Tests for evaluation.py retrieval metrics, statistics and qrels handling."""

import json

import pytest

from evaluation import (
    TEST_QUERIES,
    _normalize,
    bootstrap_ci,
    hit_at_k,
    load_qrels,
    location_accuracy,
    mrr_at_k,
    normalize_city,
    paired_bootstrap_ci,
    percentile,
    precision_at_k,
    split_judged,
    win_loss_tie,
)


def _res(*ids):
    return [{"business_id": b, "restaurant": f"name-{b}"} for b in ids]


# ── Name / city normalization ──────────────────────────────────────────────

def test_normalize_strips_accents_and_branch_suffixes():
    assert _normalize("Café Du Monde") == "cafe du monde"
    assert _normalize("Hattie B's Hot Chicken - Nashville") == "hattie b's hot chicken"
    assert _normalize("Shells Seafood Restaurant- Carrollwood") == "shells seafood restaurant"


@pytest.mark.parametrize("apostrophe", ["’", "‘", "ʼ", "`"])
def test_normalize_maps_curly_apostrophes_to_straight(apostrophe):
    assert _normalize(f"Hattie B{apostrophe}s Hot Chicken") == _normalize("Hattie B's Hot Chicken")


def test_normalize_city_saint_and_punctuation():
    assert normalize_city("St. Petersburg") == "saint petersburg"
    assert normalize_city("St Louis") == "saint louis"
    assert normalize_city("Town 'N' Country") == "town n country"
    # "st" only expands as a whole leading word, not inside other words
    assert normalize_city("West Chester") == "west chester"


# ── Qrels ──────────────────────────────────────────────────────────────────

def test_test_queries_have_unique_qids_and_no_inline_labels():
    qids = [q["qid"] for q in TEST_QUERIES]
    assert len(qids) == len(set(qids))
    assert all("relevant" not in q for q in TEST_QUERIES)


def test_load_qrels_returns_sets(tmp_path):
    path = tmp_path / "qrels.json"
    path.write_text(json.dumps({"q1": ["a", "b"], "q2": []}), encoding="utf-8")
    assert load_qrels(path) == {"q1": {"a", "b"}, "q2": set()}


def test_split_judged_excludes_unjudged_and_empty_queries():
    queries = [{"qid": "q1", "query": "x"}, {"qid": "q2", "query": "y"}, {"qid": "q3", "query": "z"}]
    qrels = {"q1": {"a"}, "q2": set()}
    scored, excluded = split_judged(queries, qrels)
    assert [q["qid"] for q in scored] == ["q1"]
    assert {q["qid"]: reason for q, reason in excluded} == {
        "q2": "0 relevant in pool",
        "q3": "not judged",
    }


# ── Metrics (business_id based) ────────────────────────────────────────────

def test_mrr_at_k_rewards_earlier_hits():
    assert mrr_at_k(_res("a", "x"), {"a"}, k=5) == 1.0
    assert mrr_at_k(_res("x", "a"), {"a"}, k=5) == 0.5


def test_mrr_at_k_zero_when_no_hit_in_k():
    assert mrr_at_k(_res("x", "y", "z", "w", "v", "a"), {"a"}, k=5) == 0.0


def test_metrics_match_on_business_id_not_name():
    # Same display name, different business: must not count.
    results = [{"business_id": "other-branch", "restaurant": "Gumbo Shop"}]
    assert hit_at_k(results, {"gumbo-shop-id"}, k=5) == 0.0
    # Results without a business_id never match.
    assert hit_at_k([{"restaurant": "Gumbo Shop"}], {"gumbo-shop-id"}, k=5) == 0.0


def test_hit_at_k():
    assert hit_at_k(_res("x", "a"), {"a"}, k=2) == 1.0
    assert hit_at_k(_res("x", "a"), {"a"}, k=1) == 0.0


def test_precision_at_k_denominator_is_always_k():
    # Precision divides by k, not by however many results came back — a
    # system returning 2 results (both relevant) scores P@5 = 0.4, not 1.0.
    assert precision_at_k(_res("a", "b"), {"a", "b"}, k=5) == pytest.approx(2 / 5)


def test_precision_at_k_empty_results():
    assert precision_at_k([], {"a"}, k=5) == 0.0


# ── Statistics ─────────────────────────────────────────────────────────────

def test_percentile_nearest_rank():
    vals = list(range(1, 21))  # 1..20
    assert percentile(vals, 50) == 10
    assert percentile(vals, 95) == 19
    assert percentile(vals, 100) == 20
    assert percentile([7.0], 95) == 7.0
    assert percentile([], 95) == 0.0


def test_paired_bootstrap_ci_on_constant_difference():
    a = [1.0, 0.5, 0.0, 1.0]
    b = [0.5, 0.0, -0.5, 0.5]
    delta, lo, hi = paired_bootstrap_ci(a, b)
    assert delta == pytest.approx(0.5)
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(0.5)


def test_paired_bootstrap_ci_brackets_mean_and_rejects_unequal_lengths():
    a = [1.0, 0.0, 1.0, 0.5, 0.0, 1.0]
    b = [0.5, 0.0, 0.0, 0.5, 1.0, 0.5]
    delta, lo, hi = paired_bootstrap_ci(a, b)
    assert lo <= delta <= hi
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.0], [1.0, 0.0])


def test_win_loss_tie():
    assert win_loss_tie([1.0, 0.5, 0.0, 0.2], [0.5, 0.5, 1.0, 0.2]) == (1, 1, 2)


def test_bootstrap_ci_empty():
    assert bootstrap_ci([]) == (0.0, 0.0)


# ── Location accuracy ──────────────────────────────────────────────────────

def test_location_accuracy_metro_area_matching():
    # Cherry Hill, NJ counts as a hit for expected_city="philadelphia".
    results = [{"city": "cherry hill"}, {"city": "philadelphia"}, {"city": "chicago"}]
    assert location_accuracy(results, "philadelphia") == pytest.approx(2 / 3)


def test_location_accuracy_shared_suburb_counts_for_each_metro():
    # Brentwood is in both the Nashville and Saint Louis lists; the old
    # reverse map was last-write-wins and scored it as Saint Louis only.
    assert location_accuracy([{"city": "Brentwood"}], "nashville") == 1.0
    assert location_accuracy([{"city": "Brentwood"}], "saint_louis") == 1.0
    assert location_accuracy([{"city": "Wilmington"}], "philadelphia") == 1.0
    assert location_accuracy([{"city": "Wilmington"}], "wilmington") == 1.0


def test_location_accuracy_normalizes_saint():
    assert location_accuracy([{"city": "St. Petersburg"}], "tampa") == 1.0
    assert location_accuracy([{"city": "St. Louis"}], "saint_louis") == 1.0


def test_location_accuracy_none_when_no_expected_city():
    assert location_accuracy([{"city": "philadelphia"}], None) is None
