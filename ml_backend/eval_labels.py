"""
Relevance-label builder for evaluation.py.

  pool    Writes eval_data/label_pool.csv: per query, the union of the top-depth
          results from each strategy plus resolved legacy name labels, with an
          empty `relevant` column. Keeps judgments from an existing pool file.
  import  Validates that every row is judged y/n and writes eval_data/qrels.json
          ({qid: [business_id, ...]}).

Commit both files: the CSV is the judgment audit trail; evaluation.py and CI
read the JSON.

Usage:
  python ml_backend/eval_labels.py pool --url http://127.0.0.1:8000 [--depth 10]
  python ml_backend/eval_labels.py import
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import config
from evaluation import (
    STRATEGIES,
    TEST_QUERIES,
    _METRO_NORMALIZED,
    _normalize,
    normalize_city,
    retrieve,
)

ROOT = _REPO_ROOT
POOL_PATH = ROOT / "eval_data" / "label_pool.csv"
QRELS_OUT = ROOT / "eval_data" / "qrels.json"
BUSINESS_JSON = ROOT / "yelp_academic_dataset_business.json"

POOL_COLUMNS = [
    "qid",
    "query",
    "business_id",
    "name",
    "city",
    "categories",
    "stars",
    "review_count",
    "is_open",
    "best_rank",
    "surfaced_by",
    "old_label",
    "snippet_1",
    "snippet_2",
    "relevant",
]
VALID_JUDGMENTS = {"y", "n"}
SNIPPET_CHARS = 280

LEGACY_NAME_LABELS: Dict[str, List[str]] = {
    "best-tacos-in-philadelphia": [
        "El Purepecha",
        "South Philly Barbacoa",
        "Blue Corn",
        "Mission Taqueria",
    ],
    "romantic-italian-dinner-philadelphia": [
        "Bistro Romano",
        "Gran Caffe L'Aquila",
        "L'Angolo Ristorante",
        "Vetri Cucina",
    ],
    "late-night-bars-philadelphia": [
        "Glory Beer Bar & Kitchen",
        "South",
        "Barclay Prime",
        "Cuba Libre Restaurant & Rum Bar",
        "La Calaca Feliz",
    ],
    "best-cheesesteak-philadelphia": [
        "Dalessandro's Steaks & Hoagies",
        "Jim's South St",
        "John's Roast Pork",
        "Max's Steaks",
        "Sonny's Famous Steaks",
    ],
    "brunch-spots-philadelphia": [
        "Cafe Lift",
        "Cafe La Maude",
        "On Point Bistro",
        "Sabrina's Café",
        "Honey's Sit N Eat",
    ],
    "sushi-philadelphia": [
        "Hikari Sushi",
        "Vic Sushi Bar",
        "Bleu Sushi",
        "Royal Sushi & Izakaya",
        "Tomo Sushi & Ramen",
    ],
    "vegan-restaurants-philadelphia": [
        "Vedge",
        "Charlie Was a Sinner",
        "V Street",
        "HipCityVeg",
    ],
    "rooftop-bars-philadelphia": ["Harp & Crown", "The Continental Mid-town"],
    "best-ramen-in-philly": ["Terakawa Ramen", "Tomo Sushi & Ramen", "Ramen House"],
    "best-hot-chicken-nashville": [
        "Hattie B's Hot Chicken",
        "Prince's Hot Chicken Shack",
        "Prince's Hot Chicken South",
        "Music City Chicken",
    ],
    "romantic-dinner-nashville": [
        "The Optimist",
        "Merchants",
        "The Standard At The Smith House",
        "Jeff Ruby's Steakhouse",
    ],
    "live-music-bars-nashville": [
        "Jason Aldean's Kitchen + Rooftop Bar",
        "Bourbon Street Blues & Boogie Bar",
        "Skull's Rainbow Room",
        "Ole Smoky Distillery",
    ],
    "best-bbq-nashville": [
        "Martin's Bar-B-Que Joint",
        "HoneyFire BBQ",
        "Charcoal Cowboys BBQ",
    ],
    "brunch-nashville": [
        "The Garden Brunch Cafe",
        "Tavern",
        "Another Broken Egg Cafe",
        "Big Bad Breakfast",
    ],
    "best-cuban-food-tampa": ["Cuban Foodies", "Box Of Cubans", "La Teresita Cafe"],
    "seafood-restaurants-tampa": [
        "Shells Seafood Restaurant",
        "Eddie V's Prime Seafood",
        "Heights Seafood",
        "Oystercatchers",
    ],
    "best-pizza-tampa": ["Fabrica Pizza", "Eddie & Sam's NY Pizza", "Due Amici"],
    "sushi-tampa": ["Sushi Cafe", "Soho Sushi", "Matoi Sushi", "Izakaya Tori"],
    "best-gumbo-new-orleans": ["Gumbo Shop", "Restaurant Rebirth", "Li'l Dizzy's Cafe"],
    "late-night-food-new-orleans": ["Daisy Dukes Express", "Olde Nola Cookery"],
    "best-beignets-new-orleans": ["Café Du Monde", "Cafe Beignet on Royal Street"],
    "romantic-dinner-new-orleans": [
        "Palace Café",
        "Mr. B's Bistro",
        "Coquette",
        "Broussard's",
        "Doris Metropolitan",
        "Café Amelie",
        "Herbsaint",
    ],
    "best-brunch-indianapolis": ["Milktooth", "Cafe Patachou", "Spoke & Steele"],
    "romantic-dinner-indianapolis": [
        "St. Elmo Steak House",
        "Bluebeard",
        "Beholder",
        "Tinker Street",
    ],
    "craft-beer-bars-indianapolis-indiana": [
        "Bier Brewery",
        "Sun King Brewing",
        "Metazoa Brewing",
        "Ellison Brewing Company",
        "Half Liter Beer & BBQ Hall",
    ],
    "best-mexican-food-tucson": [
        "El Charro Café",
        "Guadalajara Grill",
        "Barrio Bread",
        "El Rustico",
        "Taqueria Pico De Gallo",
        "Mi Nidito Restaurant",
    ],
    "brunch-tucson-arizona": [
        "Prep & Pastry",
        "Cup Cafe",
        "47 Scott",
        "Sonoran Brunch Company",
        "The Little One",
        "5 Points Market & Restaurant",
    ],
    "best-bbq-tucson": [
        "Brushfire BBQ",
        "Smokey Mo",
        "Holy Smokin Butts BBQ",
        "Kiss Of Smoke BBQ",
    ],
    "best-breakfast-reno-nevada": ["Peg's Glorified Ham n Eggs", "Squeeze In"],
    "craft-beer-bars-reno": ["The Brewer's Cabinet", "Bricks Restaurant & Bar"],
    "romantic-dinner-reno": ["Beaujolais Bistro"],
    "best-brunch-boise-idaho": [
        "Goldy's Breakfast Bistro",
        "Fork",
        "Egg Mann and Earl",
        "High Note",
    ],
    "craft-beer-boise": [
        "Bittercreek Alehouse",
        "Woodland Empire Ale Craft",
        "Brixx Craft House",
        "Bier:Thirty Bottle & Bistro",
        "Cloud 9 Brewery",
    ],
    "romantic-dinner-santa-barbara": [
        "Bouchon Santa Barbara",
        "The Lark",
        "Olio e Limone",
    ],
    "best-brunch-santa-barbara-california": ["Scarlett Begonia", "Barbareno"],
    "fine-dining-edmonton-alberta": ["Hardware Grill", "Corso 32", "RGE RD"],
    "best-brunch-edmonton": ["Cafe De Ville", "Bundok", "Pip", "SugarBowl"],
    "best-bbq-saint-louis": [
        "Salt + Smoke",
        "Pappy's Smokehouse",
        "Bogart's Smokehouse",
    ],
    "romantic-dinner-st-louis-missouri": [
        "Sidney Street Cafe",
        "Tony's",
        "Balaban's Wine Cellar",
        "Lombardo's Trattoria",
        "Baileys' Chocolate Bar",
    ],
    "best-vietnamese-food-saint-louis": ["Mai Lee"],
    "best-restaurants-wilmington-delaware": [
        "Bardea Food & Drink",
        "Harry's Seafood Bar & Grille",
        "Domaine Hudson",
        "Kid Shelleen's Charcoal House & Saloon",
        "The Reef Seafood & Steak Restaurant",
    ],
    "romantic-dinner-wilmington-de": ["Bardea Food & Drink", "Domaine Hudson"],
    "cozy-coffee-shop-to-study": [
        "The Broad Street Grind",
        "Chapterhouse Café & Gallery",
        "Picasso's Coffee House",
        "Kaffeine Coffee",
        "Calvin Fletcher's Coffee Company",
        "Caffeina Roasting Company",
        "'feine",
    ],
    "family-friendly-pizza-place": [
        "Your Pie",
        "Little Anthony Pizza",
        "Uncle Maddios Pizza",
        "Sal's Family Pizza",
        "Untouchables Pasta & Pizza",
    ],
    "best-ramen-spots": [
        "Ramen House",
        "Terakawa Ramen",
        "Ramen Ray",
        "Raijin Ramen",
        "Uncommon Ramen",
    ],
    "spicy-food-lovers-restaurant": [
        "Spicy Affair",
        "Thai Spice",
        "Spice Indian Cuisine",
    ],
    "upscale-steakhouse": [
        "St. Elmo Steak House",
        "Jeff Ruby's Steakhouse",
        "Eddie V's Prime Seafood",
        "Barclay Prime",
        "Ruth's Chris Steak House",
        "Del Frisco's Double Eagle Steakhouse",
    ],
}

_STOPWORDS = {
    "a",
    "an",
    "and",
    "the",
    "in",
    "of",
    "to",
    "for",
    "with",
    "best",
    "top",
    "good",
    "great",
    "place",
    "places",
    "spot",
    "spots",
    "restaurant",
    "restaurants",
    "food",
    "near",
    "me",
}


#  Legacy label resolution


def _query_metro(city: Optional[str]) -> Optional[Set[str]]:
    if city is None:
        return None
    return _METRO_NORMALIZED.get(city, {normalize_city(city.replace("_", " "))})


def resolve_legacy_labels(
    corpus: Dict[str, Dict],
) -> Tuple[Dict[str, Dict[str, str]], List[Tuple[str, str, str]]]:
    """Resolves LEGACY_NAME_LABELS to business_ids in each query's metro.

    Tries an exact normalized-name match, then a word-boundary prefix match
    (hint prefixed with "~").

    Returns:
        ({qid: {business_id: hint}}, [(qid, name, reason)] for unresolved labels).
    """
    by_name: Dict[str, List[str]] = defaultdict(list)
    for bid, b in corpus.items():
        by_name[_normalize(b["name"])].append(bid)

    def in_metro(bid: str, metro: Optional[Set[str]]) -> bool:
        return metro is None or normalize_city(corpus[bid]["city"]) in metro

    queries = {q["qid"]: q for q in TEST_QUERIES}
    resolved: Dict[str, Dict[str, str]] = defaultdict(dict)
    dropped: List[Tuple[str, str, str]] = []
    for qid, names in LEGACY_NAME_LABELS.items():
        metro = _query_metro(queries[qid]["city"])
        for name in names:
            norm = _normalize(name)
            exact = by_name.get(norm, [])
            hits = [b for b in exact if in_metro(b, metro)]
            hint = name
            if not hits:
                hits = [
                    bid
                    for other, bids in by_name.items()
                    if other.startswith(norm + " ") or norm.startswith(other + " ")
                    for bid in bids
                    if in_metro(bid, metro)
                ]
                hint = f"~{name}"
            if not hits:
                if exact:
                    cities = sorted({corpus[b]["city"] for b in exact})
                    dropped.append((qid, name, f"wrong city ({', '.join(cities)})"))
                else:
                    dropped.append((qid, name, "not in corpus"))
                continue
            for bid in hits:
                resolved[qid][bid] = hint
    return resolved, dropped


#  Enrichment


def _query_terms(query: str, city: Optional[str]) -> Set[str]:
    tokens = set(re.findall(r"[a-z0-9']+", query.lower()))
    city_tokens = set(re.findall(r"[a-z]+", (city or "").replace("_", " ")))
    terms = {t for t in tokens if t not in _STOPWORDS and t not in city_tokens}
    for t in list(terms):
        terms.update(config.QUERY_SYNONYMS.get(t, []))
    return terms


def best_snippets(chunks: List[str], terms: Set[str], n: int = 2) -> List[str]:
    """Returns the n reviews with the most query-term overlap, truncated to SNIPPET_CHARS."""
    reviews = []
    for chunk in chunks:
        parts = chunk.split("--")
        reviews.extend(p.strip() for p in parts[1:] if p.strip())
    if not reviews:
        return []

    def score(review: str) -> int:
        words = set(re.findall(r"[a-z0-9']+", review.lower()))
        return len(words & terms)

    ranked = sorted(reviews, key=score, reverse=True)
    return [re.sub(r"\s+", " ", r)[:SNIPPET_CHARS] for r in ranked[:n]]


def load_corpus() -> Dict[str, Dict]:
    """Returns {business_id: {name, city, review_chunks}} from the metadata parquet."""
    import pandas as pd

    df = pd.read_parquet(
        ROOT / config.METADATA_PATH,
        columns=["business_id", "restaurant", "city", "variable", "chunk"],
    )
    corpus: Dict[str, Dict] = {}
    for bid, grp in df.groupby("business_id", sort=False):
        first = grp.iloc[0]
        corpus[bid] = {
            "name": first["restaurant"],
            "city": first["city"] or "",
            "review_chunks": grp.loc[
                grp["variable"] != "chunked_profile", "chunk"
            ].tolist(),
        }
    return corpus


def load_business_details(path: Path, ids: Set[str]) -> Dict[str, Dict]:
    """Returns {business_id: Yelp business record} for ids; empty if path is missing."""
    details = {}
    if not path.exists():
        print(f"  ⚠ {path.name} not found — categories/stars/is_open will be blank")
        return details
    with open(path, encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            if b["business_id"] in ids:
                details[b["business_id"]] = b
    return details


#  Pool I/O


def read_pool(path: Path) -> List[Dict[str, str]]:
    """Reads the pool CSV as UTF-8 (BOM allowed), falling back to cp1252."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        print(f"  ⚠ {path.name} is not UTF-8; reading as cp1252 (re-saved by Excel?)")
        text = path.read_text(encoding="cp1252")
    return list(csv.DictReader(text.splitlines()))


def write_pool(path: Path, rows: List[Dict]) -> None:
    """Writes rows to path as UTF-8 CSV with POOL_COLUMNS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=POOL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def build_pool(url: str, depth: int, out: Path, extra_rrf_k: List[int]) -> None:
    """Builds and writes the label pool CSV to out.

    Args:
        url: Retriever base URL.
        depth: Results pooled per query and run.
        out: Pool CSV path; judgments already in it are kept.
        extra_rrf_k: Additional k_rrf values to pool no-rerank results at.

    Exits if retrieval fails or results lack business_id.
    """
    print("  Loading corpus metadata...")
    corpus = load_corpus()

    legacy, dropped = resolve_legacy_labels(corpus)
    if dropped:
        print(f"\n  Old name labels dropped ({len(dropped)}):")
        for qid, name, reason in dropped:
            print(f"    - {qid}: {name!r} — {reason}")

    # (strategy label, do_rerank, k_rrf)
    runs = [(s["name"], s["do_rerank"], config.RRF_K) for s in STRATEGIES]
    runs += [(f"no-rerank k={k}", False, k) for k in extra_rrf_k if k != config.RRF_K]
    short = {"Hybrid + Rerank": "rerank", "Hybrid (no rerank)": "no-rerank"}

    # qid -> business_id -> {"ranks": [(label, rank)], "old_label": str}
    pool: Dict[str, Dict[str, Dict]] = {q["qid"]: {} for q in TEST_QUERIES}
    print(
        f"\n  Retrieving top-{depth} for {len(TEST_QUERIES)} queries × {len(runs)} runs (uncached)..."
    )
    for i, q in enumerate(TEST_QUERIES):
        for label, do_rerank, k in runs:
            results, _, server_ms, _ = retrieve(
                url,
                q["query"],
                top_k=depth,
                do_rerank=do_rerank,
                k_rrf=k,
                no_cache=True,
            )
            if server_ms is None:
                sys.exit(
                    f"Retrieval failed for {q['query']!r} — is the retriever up at {url}?"
                )
            for rank, r in enumerate(results, start=1):
                bid = r.get("business_id")
                if not bid:
                    sys.exit(
                        "Retriever results have no business_id — restart the retriever "
                        "from current code."
                    )
                entry = pool[q["qid"]].setdefault(bid, {"ranks": [], "old_label": ""})
                entry["ranks"].append((short.get(label, label), rank))
        for bid, name in legacy.get(q["qid"], {}).items():
            pool[q["qid"]].setdefault(bid, {"ranks": [], "old_label": ""})[
                "old_label"
            ] = name
        print(
            f"    [{i+1:02d}/{len(TEST_QUERIES)}] {q['query'][:55]:<55} pool={len(pool[q['qid']])}"
        )

    all_ids = {bid for cands in pool.values() for bid in cands}
    print(f"\n  Loading business details for {len(all_ids)} businesses...")
    details = load_business_details(BUSINESS_JSON, all_ids)

    previous: Dict[Tuple[str, str], str] = {}
    if out.exists():
        for row in read_pool(out):
            if (row.get("relevant") or "").strip():
                previous[(row["qid"], row["business_id"])] = row["relevant"].strip()
        print(f"  Carrying over {len(previous)} existing judgments from {out.name}")

    rows = []
    for q in TEST_QUERIES:
        terms = _query_terms(q["query"], q["city"])
        cands = pool[q["qid"]]
        order = sorted(
            cands.items(),
            key=lambda kv: min((r for _, r in kv[1]["ranks"]), default=10**6),
        )
        for bid, info in order:
            c = corpus.get(bid, {"name": "?", "city": "", "review_chunks": []})
            d = details.get(bid, {})
            snippets = best_snippets(c["review_chunks"], terms) + ["", ""]
            surfaced = [f"{label}#{rank}" for label, rank in info["ranks"]]
            if info["old_label"]:
                surfaced.append("old_label")
            rows.append(
                {
                    "qid": q["qid"],
                    "query": q["query"],
                    "business_id": bid,
                    "name": c["name"],
                    "city": c["city"],
                    "categories": d.get("categories") or "",
                    "stars": d.get("stars", ""),
                    "review_count": d.get("review_count", ""),
                    "is_open": d.get("is_open", ""),
                    "best_rank": min((r for _, r in info["ranks"]), default=""),
                    "surfaced_by": "; ".join(surfaced),
                    "old_label": info["old_label"],
                    "snippet_1": snippets[0],
                    "snippet_2": snippets[1],
                    "relevant": previous.get((q["qid"], bid), ""),
                }
            )

    write_pool(out, rows)
    todo = sum(1 for r in rows if not r["relevant"])
    print(f"\n  Wrote {len(rows)} rows to {out}  ({todo} to judge)")
    print(
        "  Fill `relevant` with y/n, then run: python ml_backend/eval_labels.py import"
    )


#  Import


def build_qrels(rows: List[Dict[str, str]]) -> Tuple[Dict[str, List[str]], List[str]]:
    """Validates judged pool rows.

    Returns:
        ({qid: [relevant business_ids]}, errors). qrels is valid only if errors is empty.
    """
    known_qids = [q["qid"] for q in TEST_QUERIES]
    errors: List[str] = []
    relevant: Dict[str, Set[str]] = defaultdict(set)
    seen_qids: Set[str] = set()

    for i, row in enumerate(rows, start=2):
        qid = (row.get("qid") or "").strip()
        bid = (row.get("business_id") or "").strip()
        judgment = (row.get("relevant") or "").strip().lower()
        where = f"line {i} ({qid} / {row.get('name') or bid})"
        if qid not in known_qids:
            errors.append(f"{where}: unknown qid")
            continue
        if not bid:
            errors.append(f"{where}: missing business_id")
            continue
        if not judgment:
            errors.append(f"{where}: not judged")
            continue
        if judgment not in VALID_JUDGMENTS:
            errors.append(
                f"{where}: invalid value {row.get('relevant')!r} (use y or n)"
            )
            continue
        seen_qids.add(qid)
        if judgment == "y":
            relevant[qid].add(bid)

    qrels = {qid: sorted(relevant[qid]) for qid in known_qids if qid in seen_qids}
    return qrels, errors


def import_pool(pool_path: Path, out: Path) -> None:
    """Writes qrels JSON from the judged pool to out; exits 1 without writing on errors."""
    rows = read_pool(pool_path)
    qrels, errors = build_qrels(rows)
    if errors:
        print(
            f"  ✗ {len(errors)} problem(s) in {pool_path.name} — nothing written:",
            file=sys.stderr,
        )
        for e in errors[:50]:
            print(f"    - {e}", file=sys.stderr)
        if len(errors) > 50:
            print(f"    ... and {len(errors) - 50} more", file=sys.stderr)
        sys.exit(1)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(qrels, f, indent=2)
        f.write("\n")

    empty = [qid for qid, ids in qrels.items() if not ids]
    n_rel = sum(len(ids) for ids in qrels.values())
    print(f"  Wrote {out}: {len(qrels)} queries, {n_rel} relevant judgments")
    if empty:
        print(
            f"  {len(empty)} queries have 0 relevant (evaluation.py will list and exclude them):"
        )
        for qid in empty:
            print(f"    - {qid}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DineRAG relevance-label builder")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pool = sub.add_parser("pool", help="Build the candidate pool CSV to judge.")
    p_pool.add_argument(
        "--url", default="http://127.0.0.1:8000", help="Retriever base URL"
    )
    p_pool.add_argument("--depth", type=int, default=10)
    p_pool.add_argument("--out", type=Path, default=POOL_PATH)
    p_pool.add_argument(
        "--extra-rrf-k",
        type=_int_list,
        default=[],
        help="Also pool no-rerank results at these k_rrf values (e.g. 10,30,100,150), "
        "so an RRF sweep isn't scored against candidates it was never pooled from.",
    )

    p_imp = sub.add_parser(
        "import", help="Validate the judged pool and write qrels.json."
    )
    p_imp.add_argument("--pool", type=Path, default=POOL_PATH)
    p_imp.add_argument("--out", type=Path, default=QRELS_OUT)

    args = parser.parse_args()
    if args.cmd == "pool":
        build_pool(args.url, args.depth, args.out, args.extra_rrf_k)
    else:
        import_pool(args.pool, args.out)
