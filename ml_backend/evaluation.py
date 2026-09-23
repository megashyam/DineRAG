import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from tabulate import tabulate

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config

METRO_AREAS: Dict[str, List[str]] = {
    "philadelphia": [
        "philadelphia",
        "phila",
        "philadephia",
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
        "ardmore",
        "bala cynwyd",
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
        "pitman",
        "woolwich township",
        "west deptford",
        "westmont",
        "willingboro",
        "delran",
        "riverton",
        "palmyra",
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
        "st. louis",
        "chesterfield",
        "ballwin",
        "creve coeur",
        "ballwin",
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
        "collinsville",
        "cahokia",
        "east saint louis",
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


_SUBURB_TO_METRO: Dict[str, str] = {}
for metro, suburbs in METRO_AREAS.items():
    for suburb in suburbs:
        _SUBURB_TO_METRO[suburb.lower()] = metro


def _normalize(name: str) -> str:
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
    s = re.sub(r"[''`]", "'", s)

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


def name_match(returned: str, expected: str) -> bool:
    """Fuzzy match: normalise → exact, prefix (len≥3), or 2+-token subset."""
    r, e = _normalize(returned), _normalize(expected)
    if r == e:
        return True

    if len(r) >= 3 and len(e) >= 3:
        if r.startswith(e + " ") or e.startswith(r + " "):
            return True
    r_tok, e_tok = set(r.split()), set(e.split())
    shorter = r_tok if len(r_tok) <= len(e_tok) else e_tok
    longer = r_tok if len(r_tok) > len(e_tok) else e_tok
    if len(shorter) >= 2 and shorter.issubset(longer):
        return True
    return False


def in_relevant(result: Dict, relevant: set) -> bool:
    name = (result.get("restaurant") or result.get("name") or "").strip()
    return any(name_match(name, r) for r in relevant)


TEST_QUERIES: List[Dict] = [
    {
        "query": "best tacos in Philadelphia",
        "category": "cuisine",
        "city": "philadelphia",
        "relevant": [
            "El Purepecha",
            "South Philly Barbacoa",
            "Blue Corn",
            "Mission Taqueria",
        ],
    },
    {
        "query": "romantic Italian dinner Philadelphia",
        "category": "occasion+cuisine",
        "city": "philadelphia",
        "relevant": [
            "Bistro Romano",
            "Gran Caffe L'Aquila",
            "L'Angolo Ristorante",
            "Vetri Cucina",
        ],
    },
    {
        "query": "late night bars Philadelphia",
        "category": "time+type",
        "city": "philadelphia",
        "relevant": [
            "Butcher Bar",
            "Bar Hygge",
            "Harp & Crown",
            "Good Dog Bar",
            "Glory Beer Bar & Kitchen",
        ],
    },
    {
        "query": "best cheesesteak Philadelphia",
        "category": "landmark",
        "city": "philadelphia",
        "relevant": [
            "Dalessandro's Steaks & Hoagies",
            "Jim's South St",
            "John's Roast Pork",
            "Max's Steaks",
            "Sonny's Famous Steaks",
        ],
    },
    {
        "query": "brunch spots Philadelphia",
        "category": "mealtime",
        "city": "philadelphia",
        "relevant": [
            "Cafe Lift",
            "Cafe La Maude",
            "On Point Bistro",
            "Sabrina's Café",
            "Honey's Sit N Eat",
        ],
    },
    {
        "query": "sushi Philadelphia",
        "category": "cuisine",
        "city": "philadelphia",
        "relevant": [
            "Hikari Sushi",
            "Vic Sushi Bar",
            "Bleu Sushi",
            "Royal Sushi & Izakaya",
            "Tomo Sushi & Ramen",
        ],
    },
    {
        "query": "vegan restaurants Philadelphia",
        "category": "dietary",
        "city": "philadelphia",
        "relevant": ["Vedge", "Charlie Was a Sinner", "V Street", "HipCityVeg"],
    },
    {
        "query": "rooftop bars Philadelphia",
        "category": "ambiance",
        "city": "philadelphia",
        "relevant": ["The Continental Mid-town", "Harp & Crown"],
    },
    {
        "query": "best ramen in Philly",
        "category": "cuisine+noisy",
        "city": "philadelphia",
        "relevant": ["Terakawa Ramen", "Tomo Sushi & Ramen", "Ramen House"],
    },
    {
        "query": "cheap eats Philadelphia",
        "category": "budget",
        "city": "philadelphia",
        "relevant": [],
    },
    {
        "query": "gluten free restaurants Philadelphia",
        "category": "dietary",
        "city": "philadelphia",
        "relevant": [],
    },
    {
        "query": "best hot chicken Nashville",
        "category": "landmark",
        "city": "nashville",
        "relevant": [
            "Hattie B's Hot Chicken",
            "Prince's Hot Chicken Shack",
            "Prince's Hot Chicken South",
            "Music City Chicken",
        ],
    },
    {
        "query": "romantic dinner Nashville",
        "category": "occasion",
        "city": "nashville",
        "relevant": [
            "The Optimist",
            "Merchants",
            "The Standard At The Smith House",
            "Jeff Ruby's Steakhouse",
        ],
    },
    {
        "query": "live music bars Nashville",
        "category": "ambiance",
        "city": "nashville",
        "relevant": [
            "Jason Aldean's Kitchen + Rooftop Bar",
            "Bourbon Street Blues & Boogie Bar",
            "Skull's Rainbow Room",
            "Ole Smoky Distillery",
        ],
    },
    {
        "query": "best BBQ Nashville",
        "category": "cuisine",
        "city": "nashville",
        "relevant": [
            "Martin's Bar-B-Que Joint",
            "HoneyFire BBQ",
            "Charcoal Cowboys BBQ",
        ],
    },
    {
        "query": "brunch Nashville",
        "category": "mealtime",
        "city": "nashville",
        "relevant": [
            "The Garden Brunch Cafe",
            "Tavern",
            "Another Broken Egg Cafe",
            "Big Bad Breakfast",
        ],
    },
    {
        "query": "best tacos Nashville Tennessee",
        "category": "cuisine",
        "city": "nashville",
        "relevant": [],
    },
    {
        "query": "coffee shops Nashville",
        "category": "type",
        "city": "nashville",
        "relevant": [],
    },
    {
        "query": "best Cuban food Tampa",
        "category": "cuisine",
        "city": "tampa",
        "relevant": ["Cuban Foodies", "Box Of Cubans", "La Teresita Cafe"],
    },
    {
        "query": "seafood restaurants Tampa",
        "category": "cuisine",
        "city": "tampa",
        "relevant": [
            "Shells Seafood Restaurant",
            "Eddie V's Prime Seafood",
            "Heights Seafood",
            "Oystercatchers",
        ],
    },
    {
        "query": "best pizza Tampa",
        "category": "cuisine",
        "city": "tampa",
        "relevant": ["Fabrica Pizza", "Eddie & Sam's NY Pizza", "Due Amici"],
    },
    {
        "query": "sushi Tampa",
        "category": "cuisine",
        "city": "tampa",
        "relevant": ["Sushi Cafe", "Soho Sushi", "Matoi Sushi", "Izakaya Tori"],
    },
    {
        "query": "outdoor dining Tampa waterfront",
        "category": "ambiance",
        "city": "tampa",
        "relevant": [],
    },
    {
        "query": "family friendly restaurants Tampa",
        "category": "occasion",
        "city": "tampa",
        "relevant": [],
    },
    {
        "query": "best gumbo New Orleans",
        "category": "landmark",
        "city": "new_orleans",
        "relevant": ["Gumbo Shop", "Restaurant Rebirth", "Li'l Dizzy's Cafe"],
    },
    {
        "query": "late night food New Orleans",
        "category": "time",
        "city": "new_orleans",
        "relevant": ["Daisy Dukes Express", "Olde Nola Cookery"],
    },
    {
        "query": "best beignets New Orleans",
        "category": "landmark",
        "city": "new_orleans",
        "relevant": ["Café Du Monde", "Cafe Beignet on Royal Street"],
    },
    {
        "query": "romantic dinner New Orleans",
        "category": "occasion",
        "city": "new_orleans",
        "relevant": [
            "Palace Café",
            "Mr. B's Bistro",
            "Coquette",
            "Broussard's",
            "Doris Metropolitan",
        ],
    },
    {
        "query": "best po boy New Orleans",
        "category": "landmark",
        "city": "new_orleans",
        "relevant": [],
    },
    {
        "query": "jazz bars with food NOLA",
        "category": "ambiance+noisy",
        "city": "new_orleans",
        "relevant": [],
    },
    {
        "query": "best brunch Indianapolis",
        "category": "mealtime",
        "city": "indianapolis",
        "relevant": [
            "Yolk - City Way",
            "Mornings Breakfast & Brunch",
            "RIZE - Indianapolis",
            "Gallery Pastry Bar",
        ],
    },
    {
        "query": "romantic dinner Indianapolis",
        "category": "occasion",
        "city": "indianapolis",
        "relevant": ["Mesh Restaurant", "Capri", "Provision", "Ocean Prime"],
    },
    {
        "query": "best tacos Indianapolis",
        "category": "cuisine",
        "city": "indianapolis",
        "relevant": [],
    },
    {
        "query": "craft beer bars Indianapolis Indiana",
        "category": "type",
        "city": "indianapolis",
        "relevant": [
            "Twenty Tap",
            "The Tap",
            "Sun King Brewery",
            "Ale Emporium",
            "St. Joseph Brewery & Public House",
        ],
    },
    {
        "query": "sushi Indianapolis",
        "category": "cuisine",
        "city": "indianapolis",
        "relevant": [],
    },
    {
        "query": "best Mexican food Tucson",
        "category": "cuisine",
        "city": "tucson",
        "relevant": [
            "Taqueria Pico De Gallo",
            "El Guero Canelo",
            "El Antojo Poblano",
            "Taqueria El Pueblito",
        ],
    },
    {
        "query": "brunch Tucson Arizona",
        "category": "mealtime",
        "city": "tucson",
        "relevant": ["Prep & Pastry", "Cup Cafe", "47 Scott"],
    },
    {
        "query": "best BBQ Tucson",
        "category": "cuisine",
        "city": "tucson",
        "relevant": [
            "Kiss Of Smoke BBQ",
            "Kens Hardwood Barbecue",
            "Holy Smokin Butts BBQ",
            "Smokey Mo",
        ],
    },
    {
        "query": "romantic dinner Tucson",
        "category": "occasion",
        "city": "tucson",
        "relevant": [],
    },
    {
        "query": "coffee shops Tucson",
        "category": "type",
        "city": "tucson",
        "relevant": [],
    },
    {
        "query": "best breakfast Reno Nevada",
        "category": "mealtime",
        "city": "reno",
        "relevant": ["Peg's Glorified Ham n Eggs", "Squeeze In"],
    },
    {
        "query": "craft beer bars Reno",
        "category": "type",
        "city": "reno",
        "relevant": ["The Brewer's Cabinet", "Bricks Restaurant & Bar"],
    },
    {
        "query": "romantic dinner Reno",
        "category": "occasion",
        "city": "reno",
        "relevant": ["Beaujolais Bistro"],
    },
    {
        "query": "best tacos Reno NV",
        "category": "cuisine+noisy",
        "city": "reno",
        "relevant": [],
    },
    {
        "query": "best brunch Boise Idaho",
        "category": "mealtime",
        "city": "boise",
        "relevant": ["BACON", "Locavore", "The Chef's Hut", "Moon's Kitchen Cafe"],
    },
    {
        "query": "craft beer Boise",
        "category": "type",
        "city": "boise",
        "relevant": [
            "Sockeye Brewing",
            "EDGE Brewing Co",
            "Highlands Hollow Brewhouse",
            "Bier:Thirty Bottle & Bistro",
        ],
    },
    {
        "query": "romantic dinner Boise",
        "category": "occasion",
        "city": "boise",
        "relevant": [],
    },
    {
        "query": "best pizza Boise",
        "category": "cuisine",
        "city": "boise",
        "relevant": [],
    },
    {
        "query": "romantic dinner Santa Barbara",
        "category": "occasion",
        "city": "santa_barbara",
        "relevant": ["Bouchon Santa Barbara", "The Lark", "Olio e Limone"],
    },
    {
        "query": "best brunch Santa Barbara California",
        "category": "mealtime",
        "city": "santa_barbara",
        "relevant": ["Scarlett Begonia", "Barbareno"],
    },
    {
        "query": "seafood Santa Barbara",
        "category": "cuisine",
        "city": "santa_barbara",
        "relevant": [],
    },
    {
        "query": "wine bars Santa Barbara",
        "category": "type",
        "city": "santa_barbara",
        "relevant": [],
    },
    {
        "query": "fine dining Edmonton Alberta",
        "category": "occasion",
        "city": "edmonton",
        "relevant": [
            "The Workshop Eatery",
            "Harvest Room",
            "XIX Nineteen",
            "Bistro Praha",
            "Otto Food and Drink",
        ],
    },
    {
        "query": "best brunch Edmonton",
        "category": "mealtime",
        "city": "edmonton",
        "relevant": ["Pip", "Little Brick", "OEB Breakfast", "Cafe Blackbird"],
    },
    {
        "query": "best ramen Edmonton",
        "category": "cuisine",
        "city": "edmonton",
        "relevant": [],
    },
    {
        "query": "pho Edmonton Canada",
        "category": "cuisine",
        "city": "edmonton",
        "relevant": [],
    },
    {
        "query": "best BBQ Saint Louis",
        "category": "cuisine",
        "city": "saint_louis",
        "relevant": ["Salt + Smoke", "Pappy's Smokehouse", "Bogart's Smokehouse"],
    },
    {
        "query": "romantic dinner St Louis Missouri",
        "category": "occasion+noisy",
        "city": "saint_louis",
        "relevant": [
            "Bulrush STL",
            "Polite Society",
            "Bar Les Freres",
            "The Piccadilly at Manhattan",
        ],
    },
    {
        "query": "best Vietnamese food Saint Louis",
        "category": "cuisine",
        "city": "saint_louis",
        "relevant": ["Mai Lee"],
    },
    {
        "query": "brunch Saint Louis",
        "category": "mealtime",
        "city": "saint_louis",
        "relevant": [],
    },
    {
        "query": "best restaurants Wilmington Delaware",
        "category": "cuisine",
        "city": "wilmington",
        "relevant": ["Iron Hill Brewery & Restaurant", "Columbus Inn", "Le Shio"],
    },
    {
        "query": "romantic dinner Wilmington DE",
        "category": "occasion+noisy",
        "city": "wilmington",
        "relevant": [
            "La Fia",
            "Green Room",
            "Krazy Kat's Restaurant",
            "Domaine Hudson",
        ],
    },
    {
        "query": "brunch Wilmington",
        "category": "mealtime",
        "city": "wilmington",
        "relevant": [],
    },
    {
        "query": "cozy coffee shop to study",
        "category": "ambiance",
        "city": None,
        "relevant": [
            "The Broad Street Grind",
            "Chapterhouse Café & Gallery",
            "Picasso's Coffee House",
            "Kaffeine Coffee",
        ],
    },
    {
        "query": "family friendly pizza place",
        "category": "occasion+cuisine",
        "city": None,
        "relevant": ["Your Pie", "Little Anthony Pizza", "Uncle Maddios Pizza"],
    },
    {
        "query": "best ramen spots",
        "category": "cuisine",
        "city": None,
        "relevant": [
            "Ramen House",
            "Terakawa Ramen",
            "Ramen Ray",
            "Raijin Ramen",
            "Uncommon Ramen",
        ],
    },
    {
        "query": "spicy food lovers restaurant",
        "category": "cuisine",
        "city": None,
        "relevant": ["Spicy Affair", "Thai Spice", "Spice Indian Cuisine"],
    },
    {
        "query": "outdoor seating restaurants with dogs allowed",
        "category": "ambiance+constraint",
        "city": None,
        "relevant": [],
    },
    {
        "query": "halal restaurants",
        "category": "dietary",
        "city": None,
        "relevant": [],
    },
    {
        "query": "upscale steakhouse",
        "category": "occasion+cuisine",
        "city": None,
        "relevant": [
            "KC Prime",
            "Bern's Steak House",
            "Cerise Craft Steakhouse",
            "Morton's The Steakhouse",
            "Perry's Steakhouse & Grille - Cool Springs",
        ],
    },
]


def mrr_at_k(results: List[Dict], relevant: set, k: int = 5) -> float:
    for i, r in enumerate(results[:k]):
        if in_relevant(r, relevant):
            return 1.0 / (i + 1)
    return 0.0


def hit_at_k(results: List[Dict], relevant: set, k: int) -> float:
    return float(any(in_relevant(r, relevant) for r in results[:k]))


def precision_at_k(results: List[Dict], relevant: set, k: int) -> float:
    if not results:
        return 0.0
    hits = sum(1 for r in results[:k] if in_relevant(r, relevant))
    return hits / k


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


def retrieve(
    url: str, query: str, top_k: int, do_rerank: bool
) -> Tuple[List[Dict], float]:
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{url}/retrieve",
            json={"query": query, "top_k": top_k, "do_rerank": do_rerank},
            timeout=60,
        )
        resp.raise_for_status()
        latency_ms = (time.perf_counter() - t0) * 1000
        data = resp.json()
        results = data if isinstance(data, list) else data.get("results", [])
        return results, latency_ms
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        print(f"   retrieve failed: {e}")
        return [], latency_ms


def location_accuracy(
    results: List[Dict], expected_city: Optional[str]
) -> Optional[float]:

    if expected_city is None or not results:
        return None
    metro_key = expected_city.lower()
    metro_suburbs = set(METRO_AREAS.get(metro_key, [metro_key.replace("_", " ")]))

    def city_matches(result_city: str) -> bool:
        rc = result_city.lower().strip()
        # Direct metro match
        if rc == metro_key.replace("_", " "):
            return True
        # Suburb → metro lookup
        return _SUBURB_TO_METRO.get(rc) == metro_key

    matches = sum(1 for r in results if city_matches(r.get("city") or ""))
    return matches / len(results)


def evaluate(
    url: str,
    top_k: int = 5,
    verbose: bool = False,
    out: Optional[str] = None,
    use_mlflow: bool = False,
    mlflow_experiment: str = "dinerag-retrieval-eval",
):
    strategies = [
        {"name": "Hybrid + Rerank", "do_rerank": True},
        {"name": "Hybrid (no rerank)", "do_rerank": False},
    ]

    labeled = [q for q in TEST_QUERIES if q["relevant"]]
    unlabeled = [q for q in TEST_QUERIES if not q["relevant"]]

    print(f"\n{'='*72}")
    print(f"  DineRAG Comprehensive Retrieval Evaluation")
    print(f"{'='*72}")
    print(f"  Retriever : {url}")
    print(f"  top_k     : {top_k}")

    print("  [warm-up] sending warm-up query...")
    retrieve(url, "best pizza Philadelphia", top_k=top_k, do_rerank=True)
    print("  [warm-up] done\n")

    all_records = []

    for strat in strategies:
        print(f"\n── Strategy: {strat['name']} {'─'*(50-len(strat['name']))}")

        records = []
        for i, test in enumerate(labeled):
            query = test["query"]
            relevant = set(test["relevant"])
            exp_city = test.get("city")
            category = test.get("category", "other")

            results, latency = retrieve(
                url, query, top_k=top_k, do_rerank=strat["do_rerank"]
            )

            mrr = mrr_at_k(results, relevant, k=5)
            h3 = hit_at_k(results, relevant, k=3)
            h5 = hit_at_k(results, relevant, k=5)
            p5 = precision_at_k(results, relevant, k=5)
            loc_acc = location_accuracy(results, exp_city)

            record = {
                "query": query,
                "category": category,
                "city": exp_city,
                "strategy": strat["name"],
                "mrr5": mrr,
                "hit3": h3,
                "hit5": h5,
                "p5": p5,
                "loc_acc": loc_acc,
                "latency": latency,
                "results": len(results),
                "returned": [
                    (r.get("restaurant") or r.get("name") or "?")
                    for r in results[:top_k]
                ],
            }
            records.append(record)

            status = "PASSED" if mrr > 0 else "FAILED"
            if verbose:
                print(f"  {status} [{i+1:02d}] {query}")
                loc_str = f"{loc_acc:.2f}" if loc_acc is not None else "N/A"
                print(
                    f"       MRR@5={mrr:.3f}  Hit@5={h5:.0f}  P@5={p5:.3f}  loc={loc_str}  {latency:.0f}ms"
                )
                if mrr == 0 and results:
                    names = ", ".join(
                        r.get("restaurant") or r.get("name") or "?" for r in results[:3]
                    )
                    print(f"       got: {names}")
            else:
                print(
                    f"  {status} [{i+1:02d}/{len(labeled)}] {query[:55]:<55} MRR={mrr:.2f}  {latency:.0f}ms"
                )

            time.sleep(0.2)

        all_records.extend(records)
        _print_strategy_summary(strat["name"], records, top_k)
        if use_mlflow:
            _log_to_mlflow(mlflow_experiment, strat["name"], records, url, top_k)

    if unlabeled:
        print(f"\nRobustness pass {'─'*20}")
        for test in unlabeled:
            results, latency = retrieve(url, test["query"], top_k=top_k, do_rerank=True)
            city_check = location_accuracy(results, test.get("city"))
            has = bool(results)
            print(
                f"  {'PASSED' if has else 'FAILED'} {test['query']:<55} "
                f"n={len(results)}  loc={f'{city_check:.2f}' if city_check is not None else 'N/A'}  {latency:.0f}ms"
            )
            if results and verbose:
                names = ", ".join(
                    r.get("restaurant") or r.get("name") or "?" for r in results[:3]
                )
                print(f"     → {names}")
            time.sleep(0.2)

    _print_comparison(all_records, strategies)
    _print_per_city(all_records, strategies[0]["name"])
    _print_per_category(all_records, strategies[0]["name"])
    _print_failures(all_records, strategies[0]["name"])

    if out:
        with open(out, "w") as f:
            json.dump(all_records, f, indent=2)
        print(f"\n  Results written to: {out}")


def _agg(records: List[Dict], key: str):
    return [r[key] for r in records if r[key] is not None]


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def _log_to_mlflow(
    experiment: str, strategy: str, records: List[Dict], url: str, top_k: int
) -> None:
    import mlflow

    if "MLFLOW_TRACKING_URI" not in os.environ:
        mlruns_dir = Path(__file__).resolve().parent.parent / "mlruns"
        mlruns_dir.mkdir(exist_ok=True)
        db_path = (mlruns_dir / "mlflow.db").as_posix()
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=strategy):
        mlflow.log_params({
            "strategy": strategy,
            "retriever_url": url,
            "top_k": top_k,
            "rrf_k": config.RRF_K,
            "initial_k": config.INITIAL_K,
            "max_duplicates": config.MAX_DUPLICATES,
            "n_queries": len(records),
        })
        mrr_vals = _agg(records, "mrr5")
        ci = bootstrap_ci(mrr_vals)
        loc_vals = [r["loc_acc"] for r in records if r["loc_acc"] is not None]
        mlflow.log_metrics({
            "mrr5": _mean(mrr_vals),
            "mrr5_ci_low": ci[0],
            "mrr5_ci_high": ci[1],
            "hit3": _mean(_agg(records, "hit3")),
            "hit5": _mean(_agg(records, "hit5")),
            "p5": _mean(_agg(records, "p5")),
            "loc_acc": _mean(loc_vals) if loc_vals else 0.0,
            "avg_latency_ms": _mean(_agg(records, "latency")),
        })


def _print_strategy_summary(name: str, records: List[Dict], top_k: int):
    mrr_vals = _agg(records, "mrr5")
    hit3_vals = _agg(records, "hit3")
    hit5_vals = _agg(records, "hit5")
    p5_vals = _agg(records, "p5")
    lat_vals = _agg(records, "latency")
    loc_vals = [r["loc_acc"] for r in records if r["loc_acc"] is not None]

    mrr_ci = bootstrap_ci(mrr_vals)
    n = len(mrr_vals)

    print(f"\n  Summary ({name}, n={n}):")
    print(
        f"    MRR@5  : {_mean(mrr_vals):.3f}  95% CI [{mrr_ci[0]:.3f}, {mrr_ci[1]:.3f}]"
    )
    print(f"    Hit@3  : {_mean(hit3_vals):.3f}")
    print(f"    Hit@5  : {_mean(hit5_vals):.3f}")
    print(f"    P@5    : {_mean(p5_vals):.3f}  (denominator=k, not len(results))")
    print(f"    Loc acc: {_mean(loc_vals):.3f}  (metro-area aware)")
    if lat_vals:
        p95_idx = int(0.95 * len(lat_vals))
        print(
            f"    Latency: {_mean(lat_vals):.0f}ms avg  |  p95={sorted(lat_vals)[p95_idx]:.0f}ms"
        )


def _print_comparison(records: List[Dict], strategies: List[Dict]):
    print(f"\n\n{'='*72}")
    print("  STRATEGY COMPARISON")
    print(f"{'='*72}\n")
    rows = []
    for strat in strategies:
        sname = strat["name"]
        r = [x for x in records if x["strategy"] == sname]
        if not r:
            continue
        mrr_vals = _agg(r, "mrr5")
        ci = bootstrap_ci(mrr_vals)
        rows.append(
            [
                sname,
                f"{_mean(mrr_vals):.3f}",
                f"[{ci[0]:.3f}, {ci[1]:.3f}]",
                f"{_mean(_agg(r, 'hit3')):.3f}",
                f"{_mean(_agg(r, 'hit5')):.3f}",
                f"{_mean(_agg(r, 'p5')):.3f}",
                f"{_mean(_agg(r, 'latency')):.0f}ms",
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
                "Avg Lat",
                "n",
            ],
            tablefmt="rounded_outline",
        )
    )


def _print_per_city(records: List[Dict], strategy: str):
    print(f"\n\n{'='*72}")
    print(f"  PER-CITY BREAKDOWN  ({strategy})")
    print(f"{'='*72}\n")
    r = [x for x in records if x["strategy"] == strategy]
    cities = sorted(set(x["city"] for x in r if x["city"]))
    rows = []
    for city in cities:
        cr = [x for x in r if x["city"] == city]
        mrr_vals = _agg(cr, "mrr5")
        loc_vals = [x["loc_acc"] for x in cr if x["loc_acc"] is not None]
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
    print(f"  PER-CATEGORY BREAKDOWN  ({strategy})")
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
        print("\n  No zero-MRR queries. PASSED cleanly)")
        return
    print(f"\n\n{'='*72}")
    print(f"  FAILURE ANALYSIS — {len(failures)} zero-MRR queries  ({strategy})")
    print(f"{'='*72}\n")
    for f in failures:
        print(f"  FAILED [{f['city'] or 'no-city'}]  {f['query']}")
        if f["returned"]:
            print(f"     got: {', '.join(f['returned'][:3])}")
        else:
            print(f"     got: (no results)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DineRAG Comprehensive Retrieval Evaluator"
    )
    parser.add_argument(
        "--url",
        default="https://megumind6172--food-rag-retriever-serve.modal.run",
        help="Retriever base URL",
    )
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--mlflow", action="store_true",
        help="Log per-strategy metrics to MLflow (local ./mlruns by default; "
             "set MLFLOW_TRACKING_URI to point elsewhere).",
    )
    parser.add_argument(
        "--mlflow-experiment", type=str, default="dinerag-retrieval-eval",
    )
    args = parser.parse_args()

    evaluate(
        url=args.url,
        top_k=args.top_k,
        verbose=args.verbose,
        out=args.out,
        use_mlflow=args.mlflow,
        mlflow_experiment=args.mlflow_experiment,
    )
