"""Tests for eval_labels.py pool reading, validation and qrels import.

Import must write nothing while any row is unjudged or invalid.
"""

import csv
import json

import pytest

from evaluation import TEST_QUERIES
from eval_labels import POOL_COLUMNS, best_snippets, build_qrels, import_pool, read_pool

Q1, Q2 = TEST_QUERIES[0]["qid"], TEST_QUERIES[1]["qid"]


def _row(qid, bid, relevant, name="X"):
    row = {c: "" for c in POOL_COLUMNS}
    row.update(qid=qid, business_id=bid, name=name, relevant=relevant)
    return row


def _write(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=POOL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_build_qrels_collects_relevant_ids_per_query():
    rows = [
        _row(Q1, "a", "y"),
        _row(Q1, "b", "n"),
        _row(Q1, "c", "Y "),  # case/whitespace tolerant
        _row(Q2, "d", "n"),
    ]
    qrels, errors = build_qrels(rows)
    assert errors == []
    # Q2 is judged but has no relevant business: kept as an empty list so
    # evaluation.py can report it as excluded.
    assert qrels == {Q1: ["a", "c"], Q2: []}


def test_build_qrels_rejects_blank_and_invalid_judgments():
    rows = [
        _row(Q1, "a", "y"),
        _row(Q1, "b", "", name="Blank Place"),
        _row(Q1, "c", "maybe"),
        _row("not-a-real-qid", "d", "y"),
        _row(Q2, "", "y"),
    ]
    _, errors = build_qrels(rows)
    assert len(errors) == 4
    assert any("Blank Place" in e and "not judged" in e for e in errors)
    assert any("invalid value 'maybe'" in e for e in errors)
    assert any("unknown qid" in e for e in errors)
    assert any("missing business_id" in e for e in errors)


def test_import_pool_writes_expected_qrels(tmp_path):
    pool, out = tmp_path / "pool.csv", tmp_path / "qrels.json"
    _write(pool, [_row(Q1, "a", "y"), _row(Q1, "b", "n"), _row(Q2, "c", "y")])
    import_pool(pool, out)
    assert json.loads(out.read_text(encoding="utf-8")) == {Q1: ["a"], Q2: ["c"]}


def test_import_pool_refuses_to_write_with_blanks(tmp_path):
    pool, out = tmp_path / "pool.csv", tmp_path / "qrels.json"
    _write(pool, [_row(Q1, "a", "y"), _row(Q1, "b", "")])
    with pytest.raises(SystemExit) as exc:
        import_pool(pool, out)
    assert exc.value.code == 1
    assert not out.exists()


def test_read_pool_tolerates_excel_bom(tmp_path):
    pool = tmp_path / "pool.csv"
    _write(pool, [_row(Q1, "a", "y")])
    pool.write_bytes(b"\xef\xbb\xbf" + pool.read_bytes())
    rows = read_pool(pool)
    assert rows[0]["qid"] == Q1


def test_best_snippets_prefers_reviews_matching_query_terms():
    chunk = (
        "passage: positive customer reviews for X mention:\n"
        " -- Nice staff and parking.\n\n-- The tacos al pastor are the best tacos around."
    )
    snippets = best_snippets([chunk], {"tacos"}, n=1)
    assert snippets == ["The tacos al pastor are the best tacos around."]
