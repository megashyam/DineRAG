"""
Retrieval evaluation for the /retrieve endpoint: MRR@5, Hit@3/5, P@5, latency.

Labels: eval_data/qrels.json ({qid: [business_id, ...]}), built by
eval_labels.py. Matched by business_id. Unpooled businesses count as
non-relevant; queries with no relevant business are excluded.

Requests use no_cache=True. Strategies are compared per query (paired
bootstrap CI on ΔMRR@5, win/loss/tie). Location accuracy only sanity-checks
the retriever's city filter.
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
from tabulate import tabulate

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")


QRELS_PATH = Path(_REPO_ROOT) / "eval_data" / "qrels.json"

STRATEGIES: List[Dict] = [
    {"name": "Hybrid + Rerank", "do_rerank": True},
    {"name": "Hybrid (no rerank)", "do_rerank": False},
]


METRO_AREAS: Dict[str, List[str]] = {
    "philadelphia": [
        "philadelphia",
        "phila",
        "philadephia",
        # PA suburbs
        "abington",
        "ardmore",
        "bala cynwyd",
        "berwyn",
        "blue bell",
        "broomall",
        "bryn mawr",
        "chalfont",
        "cheltenham",
        "chester",
        "coatesville",
        "collegeville",
        "conshohocken",
        "devon",
        "downingtown",
        "doylestown",
        "drexel hill",
        "exton",
        "fort washington",
        "frazer",
        "glenside",
        "havertown",
        "horsham",
        "jenkintown",
        "king of prussia",
        "lansdale",
        "lansdowne",
        "limerick",
        "malvern",
        "manayunk",
        "media",
        "montgomeryville",
        "narberth",
        "newtown square",
        "norristown",
        "oreland",
        "paoli",
        "phoenixville",
        "plymouth meeting",
        "pottstown",
        "quakertown",
        "royersford",
        "skippack",
        "souderton",
        "springfield",
        "swarthmore",
        "upper darby",
        "wayne",
        "west chester",
        "willow grove",
        "wynnewood",
        "yardley",
        # NJ suburbs
        "cherry hill",
        "voorhees",
        "haddonfield",
        "moorestown",
        "marlton",
        "mount laurel",
        "medford",
        "collingswood",
        "haddon township",
        "haddon heights",
        "audubon",
        "bellmawr",
        "barrington",
        "lawnside",
        "gloucester city",
        "deptford",
        "woodbury",
        "sewell",
        "turnersville",
        "blackwood",
        "sicklerville",
        "lindenwold",
        "clementon",
        "pitman",
        "swedesboro",
        "bordentown",
        "burlington",
        "mount holly",
        "lumberton",
        "cinnaminson",
        "palmyra",
        "maple shade",
        "berlin",
        "atco",
        "hammonton",
        "langhorne",
        "bensalem",
        "bristol",
        "levittown",
        "croydon",
        "newtown",
        "morrisville",
        "pennsauken",
        "camden",
        "gibbsboro",
        "glassboro",
        "franklinville",
        "elmer",
        "woolwich township",
        "west deptford",
        "westmont",
        "willingboro",
        "delran",
        "riverton",
        # DE suburbs
        "wilmington",
        "claymont",
        "christiana",
        "newark",
    ],
    "tampa": [
        "tampa",
        "south tampa",
        "carrollwood",
        "citrus park",
        "westchase",
        "town n country",
        "tampa palms",
        "clearwater",
        "clearwater beach",
        "st pete",
        "saint petersburg",
        "dunedin",
        "safety harbor",
        "tarpon springs",
        "palm harbor",
        "oldsmar",
        "wesley chapel",
        "new port richey",
        "port richey",
        "holiday",
        "zephyrhills",
        "plant city",
        "brandon",
        "valrico",
        "seffner",
        "lutz",
        "land o lakes",
        "spring hill",
        "brooksville",
        "seminole",
        "largo",
        "pinellas park",
        "kenneth city",
        "south pasadena",
        "st pete beach",
        "treasure island",
        "madeira beach",
        "indian shores",
        "indian rocks beach",
        "belleair bluffs",
        "north redington beach",
        "redington shores",
        "apollo beach",
        "ruskin",
        "wimauma",
        "sun city center",
        "gibsonton",
        "riverview",
        "lithia",
    ],
    "nashville": [
        "nashville",
        "hendersonville",
        "goodlettsville",
        "madison",
        "old hickory",
        "hermitage",
        "antioch",
        "nolensville",
        "brentwood",
        "franklin",
        "spring hill",
        "columbia",
        "gallatin",
        "white house",
        "portland",
        "joelton",
        "whites creek",
        "belle meade",
        "berry hill",
        "kingston springs",
        "ashland city",
        "la vergne",
        "smyrna",
        "murfreesboro",
        "lebanon",
        "mount juliet",
    ],
    "new_orleans": [
        "new orleans",
        "metairie",
        "kenner",
        "harahan",
        "river ridge",
        "jefferson",
        "belle chasse",
        "gretna",
        "westwego",
        "marrero",
        "harvey",
        "terrytown",
        "violet",
        "arabi",
        "chalmette",
        "meraux",
        "saint rose",
        "luling",
    ],
    "indianapolis": [
        "indianapolis",
        "carmel",
        "fishers",
        "noblesville",
        "westfield",
        "zionsville",
        "brownsburg",
        "avon",
        "plainfield",
        "mooresville",
        "martinsville",
        "greenwood",
        "bargersville",
        "whitestown",
        "speedway",
        "beech grove",
        "lawrence",
        "castleton",
        "new palestine",
        "greenfield",
        "mccordsville",
        "camby",
    ],
    "tucson": [
        "tucson",
        "marana",
        "sahuarita",
        "oro valley",
        "catalina",
        "vail",
        "corona de tucson",
        "green valley",
        "mount lemmon",
    ],
    "reno": [
        "reno",
        "sparks",
        "spanish springs",
        "verdi",
        "cold springs",
        "sun valley",
        "virginia city",
    ],
    "boise": [
        "boise",
        "boise city",
        "eagle",
        "meridian",
        "nampa",
        "caldwell",
        "kuna",
        "star",
        "middleton",
    ],
    "santa_barbara": [
        "santa barbara",
        "goleta",
        "montecito",
        "carpinteria",
        "summerland",
        "isla vista",
        "santa ynez",
    ],
    "edmonton": [
        "edmonton",
        "st albert",
        "saint albert",
        "sherwood park",
        "spruce grove",
        "leduc",
        "beaumont",
        "fort saskatchewan",
    ],
    "saint_louis": [
        "saint louis",
        "st louis",
        "chesterfield",
        "ballwin",
        "creve coeur",
        "manchester",
        "kirkwood",
        "webster groves",
        "maplewood",
        "clayton",
        "ladue",
        "town and country",
        "des peres",
        "fenton",
        "valley park",
        "affton",
        "mehlville",
        "sappington",
        "sunset hills",
        "overland",
        "richmond heights",
        "university city",
        "olivette",
        "frontenac",
        "brentwood",
        "rock hill",
        "collinsville",
        "belleville",
        "o fallon",
        "swansea",
        "caseyville",
        "mascoutah",
        "fairview heights",
        "cahokia",
        "east saint louis",
        "saint charles",
        "saint peters",
    ],
    "wilmington": [
        "wilmington",
        "claymont",
        "christiana",
        "newark",
        "hockessin",
        "pike creek",
        "talleyville",
        "wilmington manor",
    ],
}


def normalize_city(city: str) -> str:
    """Lowercases, maps "St."/"St" to "saint", and replaces punctuation with spaces."""
    s = (city or "").lower().strip()
    s = re.sub(r"\bst\.?(?=\s)", "saint", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_METRO_NORMALIZED: Dict[str, Set[str]] = {
    metro: {normalize_city(s) for s in suburbs}
    for metro, suburbs in METRO_AREAS.items()
}


def _normalize(name: str) -> str:
    """Normalizes a restaurant name for exact matching in eval_labels.py."""
    s = name.lower().strip()
    for src, dst in [
        ("àáâãäå", "a"),
        ("èéêë", "e"),
        ("ìíîï", "i"),
        ("òóôõö", "o"),
        ("ùúûü", "u"),
        ("ñ", "n"),
        ("ç", "c"),
    ]:
        for c in src:
            s = s.replace(c, dst)

    s = re.sub(r"[‘’ʼ´`]", "'", s)
    # Strip city/branch suffixes
    s = re.sub(
        r"\s*[-–]\s*(nashville|philadelphia|tampa|new orleans|houston|south|north|"
        r"east|west|downtown|carrollwood|lower broadway|brandon|south philly|"
        r"uptown|midtown|brentwood|metairie).*$",
        "",
        s,
    )
    s = re.sub(r"[^a-z0-9' ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


#  Test queries


TEST_QUERIES: List[Dict] = [
    #  Philadelphia / South Jersey
    {
        "qid": "best-tacos-in-philadelphia",
        "query": "best tacos in Philadelphia",
        "category": "cuisine",
        "city": "philadelphia",
    },
    {
        "qid": "romantic-italian-dinner-philadelphia",
        "query": "romantic Italian dinner Philadelphia",
        "category": "occasion+cuisine",
        "city": "philadelphia",
    },
    {
        "qid": "late-night-bars-philadelphia",
        "query": "late night bars Philadelphia",
        "category": "time+type",
        "city": "philadelphia",
    },
    {
        "qid": "best-cheesesteak-philadelphia",
        "query": "best cheesesteak Philadelphia",
        "category": "landmark",
        "city": "philadelphia",
    },
    {
        "qid": "brunch-spots-philadelphia",
        "query": "brunch spots Philadelphia",
        "category": "mealtime",
        "city": "philadelphia",
    },
    {
        "qid": "sushi-philadelphia",
        "query": "sushi Philadelphia",
        "category": "cuisine",
        "city": "philadelphia",
    },
    {
        "qid": "vegan-restaurants-philadelphia",
        "query": "vegan restaurants Philadelphia",
        "category": "dietary",
        "city": "philadelphia",
    },
    {
        "qid": "rooftop-bars-philadelphia",
        "query": "rooftop bars Philadelphia",
        "category": "ambiance",
        "city": "philadelphia",
    },
    {
        "qid": "best-ramen-in-philly",
        "query": "best ramen in Philly",
        "category": "cuisine+noisy",
        "city": "philadelphia",
    },
    {
        "qid": "cheap-eats-philadelphia",
        "query": "cheap eats Philadelphia",
        "category": "budget",
        "city": "philadelphia",
    },
    {
        "qid": "gluten-free-restaurants-philadelphia",
        "query": "gluten free restaurants Philadelphia",
        "category": "dietary",
        "city": "philadelphia",
    },
    #  Nashville
    {
        "qid": "best-hot-chicken-nashville",
        "query": "best hot chicken Nashville",
        "category": "landmark",
        "city": "nashville",
    },
    {
        "qid": "romantic-dinner-nashville",
        "query": "romantic dinner Nashville",
        "category": "occasion",
        "city": "nashville",
    },
    {
        "qid": "live-music-bars-nashville",
        "query": "live music bars Nashville",
        "category": "ambiance",
        "city": "nashville",
    },
    {
        "qid": "best-bbq-nashville",
        "query": "best BBQ Nashville",
        "category": "cuisine",
        "city": "nashville",
    },
    {
        "qid": "brunch-nashville",
        "query": "brunch Nashville",
        "category": "mealtime",
        "city": "nashville",
    },
    {
        "qid": "best-tacos-nashville-tennessee",
        "query": "best tacos Nashville Tennessee",
        "category": "cuisine",
        "city": "nashville",
    },
    {
        "qid": "coffee-shops-nashville",
        "query": "coffee shops Nashville",
        "category": "type",
        "city": "nashville",
    },
    #  Tampa
    {
        "qid": "best-cuban-food-tampa",
        "query": "best Cuban food Tampa",
        "category": "cuisine",
        "city": "tampa",
    },
    {
        "qid": "seafood-restaurants-tampa",
        "query": "seafood restaurants Tampa",
        "category": "cuisine",
        "city": "tampa",
    },
    {
        "qid": "best-pizza-tampa",
        "query": "best pizza Tampa",
        "category": "cuisine",
        "city": "tampa",
    },
    {
        "qid": "sushi-tampa",
        "query": "sushi Tampa",
        "category": "cuisine",
        "city": "tampa",
    },
    {
        "qid": "outdoor-dining-tampa-waterfront",
        "query": "outdoor dining Tampa waterfront",
        "category": "ambiance",
        "city": "tampa",
    },
    {
        "qid": "family-friendly-restaurants-tampa",
        "query": "family friendly restaurants Tampa",
        "category": "occasion",
        "city": "tampa",
    },
    #  New Orleans
    {
        "qid": "best-gumbo-new-orleans",
        "query": "best gumbo New Orleans",
        "category": "landmark",
        "city": "new_orleans",
    },
    {
        "qid": "late-night-food-new-orleans",
        "query": "late night food New Orleans",
        "category": "time",
        "city": "new_orleans",
    },
    {
        "qid": "best-beignets-new-orleans",
        "query": "best beignets New Orleans",
        "category": "landmark",
        "city": "new_orleans",
    },
    {
        "qid": "romantic-dinner-new-orleans",
        "query": "romantic dinner New Orleans",
        "category": "occasion",
        "city": "new_orleans",
    },
    {
        "qid": "best-po-boy-new-orleans",
        "query": "best po boy New Orleans",
        "category": "landmark",
        "city": "new_orleans",
    },
    {
        "qid": "jazz-bars-with-food-nola",
        "query": "jazz bars with food NOLA",
        "category": "ambiance+noisy",
        "city": "new_orleans",
    },
    #  Indianapolis
    {
        "qid": "best-brunch-indianapolis",
        "query": "best brunch Indianapolis",
        "category": "mealtime",
        "city": "indianapolis",
    },
    {
        "qid": "romantic-dinner-indianapolis",
        "query": "romantic dinner Indianapolis",
        "category": "occasion",
        "city": "indianapolis",
    },
    {
        "qid": "best-tacos-indianapolis",
        "query": "best tacos Indianapolis",
        "category": "cuisine",
        "city": "indianapolis",
    },
    {
        "qid": "craft-beer-bars-indianapolis-indiana",
        "query": "craft beer bars Indianapolis Indiana",
        "category": "type",
        "city": "indianapolis",
    },
    {
        "qid": "sushi-indianapolis",
        "query": "sushi Indianapolis",
        "category": "cuisine",
        "city": "indianapolis",
    },
    #  Tucson
    {
        "qid": "best-mexican-food-tucson",
        "query": "best Mexican food Tucson",
        "category": "cuisine",
        "city": "tucson",
    },
    {
        "qid": "brunch-tucson-arizona",
        "query": "brunch Tucson Arizona",
        "category": "mealtime",
        "city": "tucson",
    },
    {
        "qid": "best-bbq-tucson",
        "query": "best BBQ Tucson",
        "category": "cuisine",
        "city": "tucson",
    },
    {
        "qid": "romantic-dinner-tucson",
        "query": "romantic dinner Tucson",
        "category": "occasion",
        "city": "tucson",
    },
    {
        "qid": "coffee-shops-tucson",
        "query": "coffee shops Tucson",
        "category": "type",
        "city": "tucson",
    },
    #  Reno
    {
        "qid": "best-breakfast-reno-nevada",
        "query": "best breakfast Reno Nevada",
        "category": "mealtime",
        "city": "reno",
    },
    {
        "qid": "craft-beer-bars-reno",
        "query": "craft beer bars Reno",
        "category": "type",
        "city": "reno",
    },
    {
        "qid": "romantic-dinner-reno",
        "query": "romantic dinner Reno",
        "category": "occasion",
        "city": "reno",
    },
    {
        "qid": "best-tacos-reno-nv",
        "query": "best tacos Reno NV",
        "category": "cuisine+noisy",
        "city": "reno",
    },
    #  Boise
    {
        "qid": "best-brunch-boise-idaho",
        "query": "best brunch Boise Idaho",
        "category": "mealtime",
        "city": "boise",
    },
    {
        "qid": "craft-beer-boise",
        "query": "craft beer Boise",
        "category": "type",
        "city": "boise",
    },
    {
        "qid": "romantic-dinner-boise",
        "query": "romantic dinner Boise",
        "category": "occasion",
        "city": "boise",
    },
    {
        "qid": "best-pizza-boise",
        "query": "best pizza Boise",
        "category": "cuisine",
        "city": "boise",
    },
    #  Santa Barbara
    {
        "qid": "romantic-dinner-santa-barbara",
        "query": "romantic dinner Santa Barbara",
        "category": "occasion",
        "city": "santa_barbara",
    },
    {
        "qid": "best-brunch-santa-barbara-california",
        "query": "best brunch Santa Barbara California",
        "category": "mealtime",
        "city": "santa_barbara",
    },
    {
        "qid": "seafood-santa-barbara",
        "query": "seafood Santa Barbara",
        "category": "cuisine",
        "city": "santa_barbara",
    },
    {
        "qid": "wine-bars-santa-barbara",
        "query": "wine bars Santa Barbara",
        "category": "type",
        "city": "santa_barbara",
    },
    #  Edmonton
    {
        "qid": "fine-dining-edmonton-alberta",
        "query": "fine dining Edmonton Alberta",
        "category": "occasion",
        "city": "edmonton",
    },
    {
        "qid": "best-brunch-edmonton",
        "query": "best brunch Edmonton",
        "category": "mealtime",
        "city": "edmonton",
    },
    {
        "qid": "best-ramen-edmonton",
        "query": "best ramen Edmonton",
        "category": "cuisine",
        "city": "edmonton",
    },
    {
        "qid": "pho-edmonton-canada",
        "query": "pho Edmonton Canada",
        "category": "cuisine",
        "city": "edmonton",
    },
    #  Saint Louis
    {
        "qid": "best-bbq-saint-louis",
        "query": "best BBQ Saint Louis",
        "category": "cuisine",
        "city": "saint_louis",
    },
    {
        "qid": "romantic-dinner-st-louis-missouri",
        "query": "romantic dinner St Louis Missouri",
        "category": "occasion+noisy",
        "city": "saint_louis",
    },
    {
        "qid": "best-vietnamese-food-saint-louis",
        "query": "best Vietnamese food Saint Louis",
        "category": "cuisine",
        "city": "saint_louis",
    },
    {
        "qid": "brunch-saint-louis",
        "query": "brunch Saint Louis",
        "category": "mealtime",
        "city": "saint_louis",
    },
    #  Wilmington
    {
        "qid": "best-restaurants-wilmington-delaware",
        "query": "best restaurants Wilmington Delaware",
        "category": "cuisine",
        "city": "wilmington",
    },
    {
        "qid": "romantic-dinner-wilmington-de",
        "query": "romantic dinner Wilmington DE",
        "category": "occasion+noisy",
        "city": "wilmington",
    },
    {
        "qid": "brunch-wilmington",
        "query": "brunch Wilmington",
        "category": "mealtime",
        "city": "wilmington",
    },
    #  No-city queries
    {
        "qid": "cozy-coffee-shop-to-study",
        "query": "cozy coffee shop to study",
        "category": "ambiance",
        "city": None,
    },
    {
        "qid": "family-friendly-pizza-place",
        "query": "family friendly pizza place",
        "category": "occasion+cuisine",
        "city": None,
    },
    {
        "qid": "best-ramen-spots",
        "query": "best ramen spots",
        "category": "cuisine",
        "city": None,
    },
    {
        "qid": "spicy-food-lovers-restaurant",
        "query": "spicy food lovers restaurant",
        "category": "cuisine",
        "city": None,
    },
    {
        "qid": "outdoor-seating-restaurants-with-dogs-allowed",
        "query": "outdoor seating restaurants with dogs allowed",
        "category": "ambiance+constraint",
        "city": None,
    },
    {
        "qid": "halal-restaurants",
        "query": "halal restaurants",
        "category": "dietary",
        "city": None,
    },
    {
        "qid": "upscale-steakhouse",
        "query": "upscale steakhouse",
        "category": "occasion+cuisine",
        "city": None,
    },
]

#  Relevance labels


def load_qrels(path: Path = QRELS_PATH) -> Dict[str, Set[str]]:
    """Returns {qid: set of relevant business_ids} from the qrels JSON."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {qid: set(ids) for qid, ids in raw.items()}


def split_judged(
    queries: List[Dict], qrels: Dict[str, Set[str]]
) -> Tuple[List[Dict], List[Tuple[Dict, str]]]:
    """Returns (queries with ≥1 relevant business, [(query, exclusion reason)])."""
    scored, excluded = [], []
    for q in queries:
        if q["qid"] not in qrels:
            excluded.append((q, "not judged"))
        elif not qrels[q["qid"]]:
            excluded.append((q, "0 relevant in pool"))
        else:
            scored.append(q)
    return scored, excluded


#  Metrics


def _is_relevant(result: Dict, relevant: Set[str]) -> bool:
    return result.get("business_id") in relevant


def mrr_at_k(results: List[Dict], relevant: Set[str], k: int = 5) -> float:
    for i, r in enumerate(results[:k]):
        if _is_relevant(r, relevant):
            return 1.0 / (i + 1)
    return 0.0


def hit_at_k(results: List[Dict], relevant: Set[str], k: int) -> float:
    return float(any(_is_relevant(r, relevant) for r in results[:k]))


def precision_at_k(results: List[Dict], relevant: Set[str], k: int) -> float:
    """Returns relevant hits in the top k divided by k (not by len(results))."""
    if not results:
        return 0.0
    hits = sum(1 for r in results[:k] if _is_relevant(r, relevant))
    return hits / k


def percentile(values: List[float], p: float) -> float:
    """Returns the nearest-rank percentile; p is in 0-100."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def bootstrap_ci(
    values: List[float], n_boot: int = 5000, ci: float = 0.95
) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(42)
    means = sorted(
        sum(rng.choice(values) for _ in range(len(values))) / len(values)
        for _ in range(n_boot)
    )
    lo_idx = int((1 - ci) / 2 * n_boot)
    hi_idx = int((1 + ci) / 2 * n_boot)
    return (means[lo_idx], means[min(hi_idx, n_boot - 1)])


def paired_bootstrap_ci(
    a: List[float], b: List[float], n_boot: int = 5000, ci: float = 0.95
) -> Tuple[float, float, float]:
    """Returns (mean, lo, hi) of paired differences a - b, bootstrapped over queries."""
    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    diffs = [x - y for x, y in zip(a, b)]
    if not diffs:
        return (0.0, 0.0, 0.0)
    lo, hi = bootstrap_ci(diffs, n_boot=n_boot, ci=ci)
    return (sum(diffs) / len(diffs), lo, hi)


def win_loss_tie(a: List[float], b: List[float]) -> Tuple[int, int, int]:
    wins = sum(1 for x, y in zip(a, b) if x > y)
    losses = sum(1 for x, y in zip(a, b) if x < y)
    return wins, losses, len(a) - wins - losses


def location_accuracy(
    results: List[Dict], expected_city: Optional[str]
) -> Optional[float]:
    """Returns the fraction of results in expected_city's metro area, or None."""
    if expected_city is None or not results:
        return None
    metro_key = expected_city.lower()
    metro = _METRO_NORMALIZED.get(
        metro_key, {normalize_city(metro_key.replace("_", " "))}
    )
    matches = sum(1 for r in results if normalize_city(r.get("city") or "") in metro)
    return matches / len(results)


#  HTTP retrieval


def retrieve(
    url: str,
    query: str,
    top_k: int,
    do_rerank: bool,
    k_rrf: int = config.RRF_K,
    initial_k: int = config.INITIAL_K,
    max_duplicates: int = config.MAX_DUPLICATES,
    no_cache: bool = True,
) -> Tuple[List[Dict], float, Optional[float], Dict]:
    """Returns (results, client_ms, server_ms, response_body); server_ms is None on failure."""
    payload = {
        "query": query,
        "top_k": top_k,
        "do_rerank": do_rerank,
        "k_rrf": k_rrf,
        "initial_k": initial_k,
        "max_duplicates": max_duplicates,
        "no_cache": no_cache,
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(f"{url}/retrieve", json=payload, timeout=120)
        resp.raise_for_status()
        client_ms = (time.perf_counter() - t0) * 1000
        data = resp.json()
        if isinstance(data, list):
            return data, client_ms, None, {}
        return (
            data.get("results", []),
            client_ms,
            float(data.get("retrieval_ms") or 0.0),
            data,
        )
    except Exception as e:
        client_ms = (time.perf_counter() - t0) * 1000
        print(f"    ⚠ retrieve failed: {e}")
        return [], client_ms, None, {}


#  Main evaluator


def run_eval(
    url: str,
    queries: List[Dict],
    qrels: Dict[str, Set[str]],
    strategies: List[Dict],
    params: Dict,
    verbose: bool = False,
) -> List[Dict]:
    """Runs each query under each strategy; returns one record per (strategy, query)."""
    top_k = params["top_k"]
    all_records = []

    for strat in strategies:
        print(f"\n Strategy: {strat['name']}  (k_rrf={params['k_rrf']}) {''*20}")
        records = []
        for i, test in enumerate(queries):
            query = test["query"]
            relevant = qrels[test["qid"]]
            exp_city = test.get("city")

            results, client_ms, server_ms, body = retrieve(
                url,
                query,
                top_k=top_k,
                do_rerank=strat["do_rerank"],
                k_rrf=params["k_rrf"],
                initial_k=params["initial_k"],
                max_duplicates=params["max_duplicates"],
                no_cache=True,
            )
            if results and not all(r.get("business_id") for r in results):
                raise RuntimeError(
                    "Retriever results have no business_id — the retriever at "
                    f"{url} predates ID-based eval; restart it from current code."
                )

            mrr = mrr_at_k(results, relevant, k=5)
            record = {
                "qid": test["qid"],
                "query": query,
                "category": test.get("category", "other"),
                "city": exp_city,
                "strategy": strat["name"],
                **params,
                "n_relevant": len(relevant),
                "mrr5": mrr,
                "hit3": hit_at_k(results, relevant, k=3),
                "hit5": hit_at_k(results, relevant, k=5),
                "p5": precision_at_k(results, relevant, k=5),
                "loc_acc": location_accuracy(results, exp_city),
                "latency_client_ms": client_ms,
                "latency_server_ms": server_ms,
                "error": server_ms is None,
                "out_of_coverage": bool(body.get("out_of_coverage", False)),
                "results": len(results),
                "returned": [
                    {
                        "business_id": r.get("business_id"),
                        "restaurant": r.get("restaurant") or r.get("name") or "?",
                        "city": r.get("city"),
                        "relevant": _is_relevant(r, relevant),
                    }
                    for r in results[:top_k]
                ],
            }
            records.append(record)

            status = "⚠" if record["error"] else ("✅" if mrr > 0 else "❌")
            srv = f"{server_ms:.0f}ms" if server_ms is not None else "—"
            if verbose:
                print(f"  {status} [{i+1:02d}] {query}")
                loc = record["loc_acc"]
                print(
                    f"       MRR@5={mrr:.3f}  Hit@5={record['hit5']:.0f}  P@5={record['p5']:.3f}  "
                    f"loc={f'{loc:.2f}' if loc is not None else 'N/A'}  "
                    f"client={client_ms:.0f}ms  server={srv}"
                )
                if mrr == 0 and results:
                    print(
                        f"       got: {', '.join(x['restaurant'] for x in record['returned'][:3])}"
                    )
            else:
                print(
                    f"  {status} [{i+1:02d}/{len(queries)}] {query[:55]:<55} "
                    f"MRR={mrr:.2f}  {client_ms:.0f}ms (server {srv})"
                )

            time.sleep(0.2)

        all_records.extend(records)
        _print_strategy_summary(strat["name"], records)

    return all_records


def evaluate(
    url: str,
    top_k: int = 5,
    k_rrf: int = config.RRF_K,
    initial_k: int = config.INITIAL_K,
    max_duplicates: int = config.MAX_DUPLICATES,
    sweep_rrf: Optional[List[int]] = None,
    verbose: bool = False,
    out: Optional[str] = None,
    use_mlflow: bool = False,
    mlflow_experiment: str = "dinerag-retrieval-eval",
    qrels_path: Path = QRELS_PATH,
) -> List[Dict]:
    try:
        qrels = load_qrels(qrels_path)
    except FileNotFoundError:
        print(
            f"No relevance labels at {qrels_path}.\n"
            "Build them first:\n"
            "  python ml_backend/eval_labels.py pool --url <retriever>   # writes eval_data/label_pool.csv\n"
            "  (fill the `relevant` column with y/n)\n"
            "  python ml_backend/eval_labels.py import                    # writes eval_data/qrels.json",
            file=sys.stderr,
        )
        sys.exit(2)
    qrels_sha = hashlib.sha256(Path(qrels_path).read_bytes()).hexdigest()[:12]

    scored, excluded = split_judged(TEST_QUERIES, qrels)

    print(f"\n{'='*72}")
    print("  DineRAG Retrieval Evaluation")
    print(f"{'='*72}")
    print(f"  Retriever : {url}")
    print(
        f"  top_k={top_k}  initial_k={initial_k}  max_duplicates={max_duplicates}  "
        f"k_rrf={'sweep ' + ','.join(map(str, sweep_rrf)) if sweep_rrf else k_rrf}"
    )
    print(
        f"  Labels    : {qrels_path.name} (sha {qrels_sha}) — pooled top-10, judged by hand, "
        f"matched on business_id"
    )
    print(f"  Scored    : {len(scored)} / {len(TEST_QUERIES)} queries")
    if excluded:
        print(f"  Excluded  : {len(excluded)}")
        for q, reason in excluded:
            print(f"     - [{reason}] {q['query']}")
    print("  Cache     : bypassed (no_cache=True) — latencies are real retrievals")
    print(f"{'='*72}\n")

    if not scored:
        print("  Nothing to score.")
        return []

    print("  [warm-up] sending warm-up query...")
    retrieve(url, "best pizza Philadelphia", top_k=top_k, do_rerank=True, no_cache=True)
    print("  [warm-up] done\n")

    base_params = {
        "top_k": top_k,
        "initial_k": initial_k,
        "max_duplicates": max_duplicates,
    }
    meta = {
        "retriever_url": url,
        "qrels_sha": qrels_sha,
        "n_queries": len(scored),
        "n_excluded": len(excluded),
    }

    if sweep_rrf:

        strategies = [s for s in STRATEGIES if not s["do_rerank"]]
        print(
            "  Sweep note: k_rrf only affects fused order, which the reranker "
            "fully re-sorts, so only the no-rerank strategy is swept.\n"
        )
        all_records = []
        for k in sweep_rrf:
            params = {**base_params, "k_rrf": k}
            recs = run_eval(url, scored, qrels, strategies, params, verbose)
            all_records.extend(recs)
            if use_mlflow:
                _log_to_mlflow(
                    mlflow_experiment,
                    f"{strategies[0]['name']} k_rrf={k}",
                    recs,
                    params,
                    meta,
                )
        _print_sweep(all_records, sweep_rrf, strategies[0]["name"])
    else:
        params = {**base_params, "k_rrf": k_rrf}
        all_records = run_eval(url, scored, qrels, STRATEGIES, params, verbose)
        paired = _paired_stats(
            all_records, STRATEGIES[0]["name"], STRATEGIES[1]["name"]
        )
        _print_comparison(all_records, STRATEGIES, paired)
        _print_per_city(all_records, STRATEGIES[0]["name"])
        _print_per_category(all_records, STRATEGIES[0]["name"])
        _print_failures(all_records, STRATEGIES[0]["name"])
        if use_mlflow:
            for strat in STRATEGIES:
                recs = [r for r in all_records if r["strategy"] == strat["name"]]
                _log_to_mlflow(
                    mlflow_experiment,
                    strat["name"],
                    recs,
                    params,
                    meta,
                    paired=paired if strat is STRATEGIES[0] else None,
                )

    n_err = sum(1 for r in all_records if r["error"])
    if n_err:
        print(
            f"\n  ⚠ {n_err} request(s) failed and were scored as misses — check the retriever."
        )

    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(
                {"meta": {**meta, **base_params}, "records": all_records}, f, indent=2
            )
        print(f"\n  Results written to: {out}")

    return all_records


#  Helpers


def _agg(records: List[Dict], key: str):
    return [r[key] for r in records if r[key] is not None]


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def _latency_stats(records: List[Dict], key: str) -> Dict[str, float]:
    vals = [r[key] for r in records if not r["error"] and r[key] is not None]
    return {
        "mean": _mean(vals),
        "p50": percentile(vals, 50),
        "p95": percentile(vals, 95),
    }


def _paired_stats(records: List[Dict], a_name: str, b_name: str) -> Optional[Dict]:
    """Returns per-query ΔMRR@5 stats of strategy a over b, or None if no shared queries."""
    a = {r["qid"]: r["mrr5"] for r in records if r["strategy"] == a_name}
    b = {r["qid"]: r["mrr5"] for r in records if r["strategy"] == b_name}
    qids = [q for q in a if q in b]
    if not qids:
        return None
    av, bv = [a[q] for q in qids], [b[q] for q in qids]
    delta, lo, hi = paired_bootstrap_ci(av, bv)
    w, l, t = win_loss_tie(av, bv)
    return {
        "a": a_name,
        "b": b_name,
        "n": len(qids),
        "delta_mrr5": delta,
        "ci_low": lo,
        "ci_high": hi,
        "wins": w,
        "losses": l,
        "ties": t,
    }


def _log_to_mlflow(
    experiment: str,
    run_name: str,
    records: List[Dict],
    params: Dict,
    meta: Dict,
    paired: Optional[Dict] = None,
) -> None:
    """Logs one MLflow run; params are the values sent to /retrieve, not config defaults."""
    import mlflow

    if "MLFLOW_TRACKING_URI" not in os.environ:
        mlruns_dir = Path(_REPO_ROOT) / "mlruns"
        mlruns_dir.mkdir(exist_ok=True)
        db_path = (mlruns_dir / "mlflow.db").as_posix()
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "strategy": records[0]["strategy"] if records else run_name,
                "no_cache": True,
                "label_source": "pooled-manual-business_id",
                **params,
                **meta,
            }
        )
        mrr_vals = _agg(records, "mrr5")
        ci = bootstrap_ci(mrr_vals)
        loc_vals = _agg(records, "loc_acc")
        client = _latency_stats(records, "latency_client_ms")
        server = _latency_stats(records, "latency_server_ms")
        metrics = {
            "mrr5": _mean(mrr_vals),
            "mrr5_ci_low": ci[0],
            "mrr5_ci_high": ci[1],
            "hit3": _mean(_agg(records, "hit3")),
            "hit5": _mean(_agg(records, "hit5")),
            "p5": _mean(_agg(records, "p5")),
            "loc_acc": _mean(loc_vals) if loc_vals else 0.0,
            "latency_client_mean_ms": client["mean"],
            "latency_client_p50_ms": client["p50"],
            "latency_client_p95_ms": client["p95"],
            "latency_server_mean_ms": server["mean"],
            "latency_server_p50_ms": server["p50"],
            "latency_server_p95_ms": server["p95"],
            "n_errors": sum(1 for r in records if r["error"]),
        }
        if paired:
            mlflow.log_param("paired_baseline", paired["b"])
            metrics.update(
                {
                    "paired_delta_mrr5": paired["delta_mrr5"],
                    "paired_delta_mrr5_ci_low": paired["ci_low"],
                    "paired_delta_mrr5_ci_high": paired["ci_high"],
                    "paired_wins": paired["wins"],
                    "paired_losses": paired["losses"],
                    "paired_ties": paired["ties"],
                }
            )
        mlflow.log_metrics(metrics)


def _print_strategy_summary(name: str, records: List[Dict]):
    mrr_vals = _agg(records, "mrr5")
    loc_vals = _agg(records, "loc_acc")
    mrr_ci = bootstrap_ci(mrr_vals)
    client = _latency_stats(records, "latency_client_ms")
    server = _latency_stats(records, "latency_server_ms")

    print(f"\n  Summary ({name}, n={len(mrr_vals)}):")
    print(
        f"    MRR@5  : {_mean(mrr_vals):.3f}  95% CI [{mrr_ci[0]:.3f}, {mrr_ci[1]:.3f}]"
    )
    print(f"    Hit@3  : {_mean(_agg(records, 'hit3')):.3f}")
    print(f"    Hit@5  : {_mean(_agg(records, 'hit5')):.3f}")
    print(f"    P@5    : {_mean(_agg(records, 'p5')):.3f}  (denominator=k)")
    print(
        f"    Loc acc: {_mean(loc_vals):.3f}  (city-filter sanity check, not a quality metric)"
    )
    print(
        f"    Latency: client {client['mean']:.0f}ms mean | p50 {client['p50']:.0f} | p95 {client['p95']:.0f}"
        f"   server {server['mean']:.0f}ms mean | p50 {server['p50']:.0f} | p95 {server['p95']:.0f}"
    )


def _print_comparison(
    records: List[Dict], strategies: List[Dict], paired: Optional[Dict]
):
    print(f"\n\n{'='*72}")
    print("  STRATEGY COMPARISON")
    print(f"{'='*72}\n")
    rows = []
    for strat in strategies:
        r = [x for x in records if x["strategy"] == strat["name"]]
        if not r:
            continue
        mrr_vals = _agg(r, "mrr5")
        ci = bootstrap_ci(mrr_vals)
        client = _latency_stats(r, "latency_client_ms")
        rows.append(
            [
                strat["name"],
                f"{_mean(mrr_vals):.3f}",
                f"[{ci[0]:.3f}, {ci[1]:.3f}]",
                f"{_mean(_agg(r, 'hit3')):.3f}",
                f"{_mean(_agg(r, 'hit5')):.3f}",
                f"{_mean(_agg(r, 'p5')):.3f}",
                f"{client['mean']:.0f}ms",
                f"{client['p95']:.0f}ms",
                len(mrr_vals),
            ]
        )
    print(
        tabulate(
            rows,
            headers=[
                "Strategy",
                "MRR@5",
                "95% CI",
                "Hit@3",
                "Hit@5",
                "P@5",
                "Mean Lat",
                "p95 Lat",
                "n",
            ],
            tablefmt="rounded_outline",
        )
    )
    if paired:
        print(
            f"\n  Paired ΔMRR@5 ({paired['a']} − {paired['b']}, n={paired['n']}): "
            f"{paired['delta_mrr5']:+.3f}  95% CI [{paired['ci_low']:+.3f}, {paired['ci_high']:+.3f}]"
        )
        print(
            f"  Per-query W/L/T: {paired['wins']}/{paired['losses']}/{paired['ties']}"
        )
        if paired["ci_low"] <= 0 <= paired["ci_high"]:
            print("  → CI includes 0: no significant difference at this sample size.")


def _print_sweep(records: List[Dict], ks: List[int], strategy: str):
    print(f"\n\n{'='*72}")
    print(f"  RRF k SWEEP  ({strategy})")
    print(f"{'='*72}\n")
    baseline = config.RRF_K if config.RRF_K in ks else ks[0]
    base = {r["qid"]: r["mrr5"] for r in records if r["k_rrf"] == baseline}
    rows = []
    for k in ks:
        r = [x for x in records if x["k_rrf"] == k]
        mrr_vals = _agg(r, "mrr5")
        ci = bootstrap_ci(mrr_vals)
        cur = {x["qid"]: x["mrr5"] for x in r}
        qids = [q for q in cur if q in base]
        delta, lo, hi = paired_bootstrap_ci(
            [cur[q] for q in qids], [base[q] for q in qids]
        )
        rows.append(
            [
                k,
                f"{_mean(mrr_vals):.3f}",
                f"[{ci[0]:.3f}, {ci[1]:.3f}]",
                f"{_mean(_agg(r, 'hit5')):.3f}",
                f"{_mean(_agg(r, 'p5')):.3f}",
                "—" if k == baseline else f"{delta:+.3f} [{lo:+.3f}, {hi:+.3f}]",
            ]
        )
    print(
        tabulate(
            rows,
            headers=[
                "k_rrf",
                "MRR@5",
                "95% CI",
                "Hit@5",
                "P@5",
                f"ΔMRR vs k={baseline} (paired CI)",
            ],
            tablefmt="rounded_outline",
        )
    )


def _print_per_city(records: List[Dict], strategy: str):
    print(f"\n\n{'='*72}")
    print(f"  PER-CITY BREAKDOWN  ({strategy})  — small n per row; indicative only")
    print(f"{'='*72}\n")
    r = [x for x in records if x["strategy"] == strategy]
    cities = sorted(set(x["city"] for x in r if x["city"]))
    rows = []
    for city in cities:
        cr = [x for x in r if x["city"] == city]
        mrr_vals = _agg(cr, "mrr5")
        loc_vals = _agg(cr, "loc_acc")
        rows.append(
            [
                city.replace("_", " ").title(),
                f"{_mean(mrr_vals):.3f}",
                f"{_mean(_agg(cr, 'hit5')):.3f}",
                f"{_mean(_agg(cr, 'p5')):.3f}",
                f"{_mean(loc_vals):.3f}" if loc_vals else "—",
                len(mrr_vals),
            ]
        )
    rows.sort(key=lambda x: x[1], reverse=True)
    print(
        tabulate(
            rows,
            headers=["City", "MRR@5", "Hit@5", "P@5", "Loc Acc", "n"],
            tablefmt="rounded_outline",
        )
    )


def _print_per_category(records: List[Dict], strategy: str):
    print(f"\n\n{'='*72}")
    print(f"  PER-CATEGORY BREAKDOWN  ({strategy})  — small n per row; indicative only")
    print(f"{'='*72}\n")
    r = [x for x in records if x["strategy"] == strategy]

    def base_cat(cat: str) -> str:
        return cat.split("+")[0]

    cats = sorted(set(base_cat(x["category"]) for x in r))
    rows = []
    for cat in cats:
        cr = [x for x in r if base_cat(x["category"]) == cat]
        mrr_vals = _agg(cr, "mrr5")
        rows.append(
            [
                cat,
                f"{_mean(mrr_vals):.3f}",
                f"{_mean(_agg(cr, 'hit5')):.3f}",
                f"{_mean(_agg(cr, 'p5')):.3f}",
                len(mrr_vals),
            ]
        )
    rows.sort(key=lambda x: x[1], reverse=True)
    print(
        tabulate(
            rows,
            headers=["Category", "MRR@5", "Hit@5", "P@5", "n"],
            tablefmt="rounded_outline",
        )
    )


def _print_failures(records: List[Dict], strategy: str):
    failures = [x for x in records if x["strategy"] == strategy and x["mrr5"] == 0.0]
    if not failures:
        print("\n  No zero-MRR queries. ✅")
        return
    print(f"\n\n{'='*72}")
    print(f"  FAILURE ANALYSIS — {len(failures)} zero-MRR queries  ({strategy})")
    print(f"{'='*72}\n")
    for f in failures:
        print(f"  ❌ [{f['city'] or 'no-city'}]  {f['query']}")
        if f["returned"]:
            print(f"     got: {', '.join(x['restaurant'] for x in f['returned'][:3])}")
        else:
            print("     got: (no results)")


#  Entry point


def _int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DineRAG Retrieval Evaluator")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="Retriever base URL (defaults to a local retriever, not production).",
    )
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--rrf-k", type=int, default=config.RRF_K)
    parser.add_argument("--initial-k", type=int, default=config.INITIAL_K)
    parser.add_argument("--max-duplicates", type=int, default=config.MAX_DUPLICATES)
    parser.add_argument(
        "--sweep-rrf",
        type=_int_list,
        default=None,
        help="Comma-separated k_rrf values to sweep, e.g. 10,30,60,100,150 "
        "(no-rerank strategy only; one MLflow run per value).",
    )
    parser.add_argument("--qrels", type=Path, default=QRELS_PATH)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Write per-query records (incl. returned business_ids) as JSON.",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Log per-strategy metrics to MLflow (local ./mlruns by default; "
        "set MLFLOW_TRACKING_URI to point elsewhere).",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="dinerag-retrieval-eval",
    )
    args = parser.parse_args()

    evaluate(
        url=args.url,
        top_k=args.top_k,
        k_rrf=args.rrf_k,
        initial_k=args.initial_k,
        max_duplicates=args.max_duplicates,
        sweep_rrf=args.sweep_rrf,
        verbose=args.verbose,
        out=args.out,
        use_mlflow=args.mlflow,
        mlflow_experiment=args.mlflow_experiment,
        qrels_path=args.qrels,
    )
