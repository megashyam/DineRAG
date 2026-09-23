from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import os
from pathlib import Path
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GEN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
DATA_DIR = Path("data")
BUSINESS_PATH = "yelp_academic_dataset_business.json"
REVIEW_PATH = "yelp_academic_dataset_review.json"
OUTPUT_PATH = DATA_DIR / "preprocessed.pkl"
CHUNKED_DATA_PATH = DATA_DIR / "chunked_data.pkl"
PRECOMPUTED_PATH = DATA_DIR / "vector_embeddings_new.pt"
EMBEDDINGS_PATH = DATA_DIR / "vector_embeddings_new.pt"
BM25_PATH = DATA_DIR / "bm25_ranks.pkl"
METADATA_PATH = DATA_DIR / "retriever_metadata.parquet"
DATA_PATH = DATA_DIR / "retriever_df.csv"

# Models
EMBEDDING_MODEL_NAME = "intfloat/e5-large-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Parameters
RESTAURANTS_COEFF = 0.3
REVIEWS_COEFF = 0.7
MAX_REVIEWS_PER_RESTAURANT = 40
REVIEW_COUNTS_PER_RESTAURANT = 50
MIN_DYNAMIC_LIMIT = 10
MIN_FILTERED_REVIEWS = 5
TOP_RESTAURANTS_PER_CITY = 200
MIN_REVIEW_WORDS = 30
FILTER_YEAR = 2018

# Parameters
BATCH_SIZE = 32
MAX_TOKENS = 256
TOP_K = 30
INITIAL_K = 30
RRF_K = 60
MAX_DUPLICATES = 1
DO_RERANK = True

QUERY_SYNONYMS: dict[str, list[str]] = {
    "bbq": ["barbecue", "barbeque"],
    "barbecue": ["bbq"],
    "barbeque": ["bbq"],
    "brunch": ["breakfast"],
    "breakfast": ["brunch"],
    "rooftop": ["terrace", "patio"],
    "romantic": ["date", "intimate"],
    "upscale": ["fancy", "highend"],
    "cozy": ["quiet", "comfortable"],
    "craft": ["brewery", "microbrewery"],
    "vegan": ["plantbased"],
    "vegetarian": ["meatless"],
    "kid": ["family", "children"],
    "family": ["kid", "children"],
}

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "food_chunks"
VECTOR_SIZE = 1024
DOC_ID = "foodguru-v1"
INGEST_BATCH_SIZE = 256
MAX_WORKERS = 4

# Text Chunking Settings
EMBED_MAX_SEQ_LENGTH = 512
CHUNK_MAX_TOKENS = 480
CHUNK_MIN_TOKENS = 50
OVERHEAD_TOKENS = 4

# External Services
RETRIEVER_URL = os.environ.get("RETRIEVER_URL")

# Groq Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_ID = "qwen/qwen3-32b"
GROQ_INTENT_MODEL = "llama-3.1-8b-instant"

# Model Configuration
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 700
TEMPERATURE = 0.2
TOP_P = 0.7
TRIM_LENGTH = 350
MAX_INPUT_TOKENS = 20000

# Quantization Settings
BNB_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": torch.float16,
}

GROQ_SYSTEM_PROMPT = (
    "You are DineRAG, a knowledgeable and helpful local food recommendation guide on the Yelp restaurant dataset.\n\n"
    "Use ONLY the provided context. Do not rely on outside knowledge. Do NOT invent restaurants, dishes, prices, or locations.\n"
    "Write with warmth and specific detail pulled from the reviews — atmosphere, a standout dish, a recurring compliment —"
    " like a knowledgeable local friend giving advice. Keep it tight: a couple of sentences per point, not a full paragraph.\n\n"
    "For each recommendation use this template:\n\n"
    "**Restaurant Name** 📍 *address, city*\n\n"
    "🍽️ [2-3 sentences: what makes this place the right answer for THIS specific query. "
    "Pull concrete details from the reviews. Never use the phrase 'why it fits' or any generic filler.]\n\n"
    "🌟 **Must-try:** [specific dishes or features from the reviews and exactly why reviewers love it]\n\n"
    "💡 *Tip:* [actionable tips a local would actually give — best time to go, what to order first, what to skip etc]\n\n"
    "There MUST be a blank line between the description and the tip.\n\n"
    "Keep the whole response under ~500 words so multi-restaurant answers don't run out of room mid-sentence."
    " Cover all spots without repeating yourself.\n\n"
    "Always read the user query carefully and respect the constraints:\n"
    "   - Health conditions or discomfort → recommend "
    "light, fresh, easy-to-digest options. Skip restaurants with negative reviews or heavy/greasy/spicy food.\n"
    "   - Dietary restrictions or allergies → recommend places with confirmed suitable options in the context.\n"
    "   - Budget constraints → recommend places where context confirms prices fit.\n"
    "   - Occasion or mood (romantic, family, casual, quick bite) → match the vibe.\n"
    "   - If NO retrieved restaurant genuinely fits the constraints, say so honestly and briefly list what is available with a caveat.\n\n"
    "LOCATION TRANSPARENCY:\n"
    "   - Never say a location is not in your dataset when results are present in the context.\n"
    "   - The context is the ground truth — trust it over your own knowledge.\n"
    "   - If results come from a nearby area rather than the exact city mentioned, acknowledge this naturally.\n\n"
    "If reviews mention drawbacks, mention them briefly and constructively.\n\n"
    "COVERAGE — you only have data for these areas:\n"
    "   Philadelphia & surrounding South Jersey suburbs (PA/NJ)\n"
    "   Tampa & Fort Myers (FL)\n"
    "   Nashville & surrounding suburbs (TN)\n"
    "   New Orleans & suburbs (LA)\n"
    "   Indianapolis & suburbs (IN)\n"
    "   Tucson & surrounding areas (AZ)\n"
    "   Reno & Spanish Springs (NV)\n"
    "   Edmonton & St. Albert (Alberta, Canada)\n"
    "   Santa Barbara & Goleta (CA)\n"
    "   Boise & surrounding Idaho cities (ID)\n"
    "   Illinois, Missouri, and Delaware (Saint Louis, Belleville, Wilmington and surrounding areas)\n\n"
    "Out-of-coverage and non-food queries are intercepted before you ever see them, so you will only be called with a"
    " genuine, in-coverage food query. If CONTEXT is still empty for one, the query was probably too narrow or unusual"
    " for anything in the dataset — say so honestly and suggest rephrasing or broadening it. Do not claim the location"
    " itself is uncovered.\n"
)


# City Corrections Mapping
US_CITIES_CORRECTIONS = {
    "Philadelphiadelphia": "Philadelphia",
    "Philly": "Philadelphia",
    "Southwest Philadelphia": "Philadelphia",
    "Tampa Bay": "Tampa",
    "Tampa,Fl": "Tampa",
    "Tampa Florida": "Tampa",
    "Southwest Tampa": "Tampa",
    "Inpolis": "Indianapolis",
    "Indianopolis": "Indianapolis",
    "Tuscon": "Tucson",
    "Tren": "Trenton",
    "Nashville-Davidson Metropolitan Government (Balance)": "Nashville",
    "East Nashville": "Nashville",
    "St. Louis": "Saint Louis",
    "St Louis": "Saint Louis",
    "SaintLouis": "Saint Louis",
    "Saint Louis County": "Saint Louis",
    "Saint Louis Downtown": "Saint Louis",
    "Saint Louis,": "Saint Louis",
    "East St. Louis": "East Saint Louis",
    "East St Louis": "East Saint Louis",
    "St. Petersburg": "Saint Petersburg",
    "St Petersburg": "Saint Petersburg",
    "SaintPetersburg": "Saint Petersburg",
    "Saintt Petersburg": "Saint Petersburg",
    "Saint Petersurg": "Saint Petersburg",
    "Mt. Juliet": "Mount Juliet",
    "Mt Juliet": "Mount Juliet",
    "Mt.Juliet": "Mount Juliet",
    "Mt. Laurel": "Mount Laurel",
    "Mt Laurel": "Mount Laurel",
    "Mt.Laurel": "Mount Laurel",
    "Mount Laurel Township": "Mount Laurel",
    "Mt Laurel Twp, Nj": "Mount Laurel",
    "Mt Holly": "Mount Holly",
    "Mount Holly,": "Mount Holly",
    "West Mount Holly": "Mount Holly",
    "Town N Country": "Town and Country",
    "Twn N Cntry": "Town and Country",
    "Land O Lakes": "Land O' Lakes",
    "Land O'Lakes": "Land O' Lakes",
    "Fairview Hts": "Fairview Heights",
    "Fairview Hts.": "Fairview Heights",
    "Woodbury Hts.": "Woodbury Heights",
    "Temple Terr": "Temple Terrace",
    "Belleair Blf": "Belleair Bluffs",
    "Pass-A-Grille Beach": "Pass-a-Grille Beach",
    "Redingtn Shor": "Redington Shores",
    "Hernando Bch": "Hernando Beach",
    "North Redington Bch": "North Redington Beach",
    "Lutz Fl": "Lutz",
    "Riverview Fl": "Riverview",
    "W.Chester": "West Chester",
    "West Chester Pa": "West Chester",
    "W. Berlin": "West Berlin",
    "S.Pasadena": "South Pasadena",
    "Bensalem. Pa": "Bensalem",
    "SaintAnn": "Saint Ann",
    "SaintRose": "Saint Rose",
    "SaintCharles": "Saint Charles",
    "Saint  Charles": "Saint Charles",
    "Santa  Barbara": "Santa Barbara",
    "Haddon Twp": "Haddon Township",
    "Bristol Twp": "Bristol Township",
    "Washington Twp": "Washington Township",
    "Woolwich Twp": "Woolwich Township",
    "Woolwich Twp.": "Woolwich Township",
    "Delran Twp": "Delran Township",
}

# State Mapping
US_STATES = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
}

STATE_ALIASES = set(US_STATES.keys()) | set(US_STATES.values())
STATE_FULL_TO_ABBR = US_STATES.copy()
STATE_ABBR_TO_FULL = {v: k.title() for k, v in STATE_FULL_TO_ABBR.items()}

# Attribute Mappings
ATTRIBUTE_MAP = {
    "RestaurantsTakeOut": "Takeout Service",
    "HasTV": "Has a Television",
    "RestaurantsDelivery": "Delivery Service",
    "OutdoorSeating": "Outdoor Seating",
    "GoodForKids": "Good For Kids",
    "GoodForDancing": "Good For Dancing",
    "RestaurantsGoodForGroups": "Good For Groups",
    "HappyHour": "Happy Hour",
    "DogsAllowed": "Dogs Allowed",
}

# Special Logic Checks
BOOL_ATTRIBUTES = set(ATTRIBUTE_MAP.keys())

# Keys check for specific values
ALCOHOL_KEYS = ["Alcohol"]
ALCOHOL_SKIP_VALUES = {"u'none'", "'none'", "none", "None"}

# Vibe Keywords
VIBE_KEYWORDS = {
    "touristy",
    "hipster",
    "divey",
    "intimate",
    "trendy",
    "upscale",
    "classy",
    "casual",
    "romantic",
}


COVERED_AREAS = [
    # Cities
    "philadelphia",
    "tampa",
    "tucson",
    "indianapolis",
    "nashville",
    "new orleans",
    "reno",
    "edmonton",
    "santa barbara",
    "boise",
    "fort myers",
    "saint louis",
    "wilmington",
    "belleville",
    "philly",
    "nola",
    "indy",
    "st louis",
    "st. louis",
    "stl",
    # States/provinces in dataset
    "pennsylvania",
    "florida",
    "arizona",
    "indiana",
    "tennessee",
    "louisiana",
    "nevada",
    "alberta",
    "california",
    "idaho",
    "illinois",
    "missouri",
    "delaware",
    # NJ is covered via Philly metro
    "new jersey",
    "nj",
    # Abbreviations
    "pa",
    "fl",
    "az",
    "in",
    "tn",
    "la",
    "nv",
    "ca",
    "id",
    "il",
    "mo",
    "de",
]
OUT_OF_COVERAGE = [
    # International
    "mumbai",
    "delhi",
    "bandra",
    "london",
    "toronto",
    "paris",
    "sydney",
    "tokyo",
    "beijing",
    "dubai",
    "singapore",
    "bangkok",
    "mexico city",
    "amsterdam",
    "rome",
    "madrid",
    "barcelona",
    "lisbon",
    "moscow",
    "cairo",
    "karachi",
    "lahore",
    "dhaka",
    "jakarta",
    "seoul",
    "hong kong",
    "taipei",
    "kuala lumpur",
    "manila",
    "colombo",
    "berlin",
    "munich",
    "hamburg",
    "vienna",
    "zurich",
    "brussels",
    "stockholm",
    "oslo",
    "copenhagen",
    "helsinki",
    "warsaw",
    "prague",
    "budapest",
    "bucharest",
    "athens",
    "istanbul",
    "tel aviv",
    "riyadh",
    "doha",
    "abu dhabi",
    "nairobi",
    "lagos",
    "johannesburg",
    "cape town",
    "accra",
    "sydney",
    "melbourne",
    "auckland",
    "christchurch",
    "sao paulo",
    "rio de janeiro",
    "buenos aires",
    "bogota",
    "lima",
    # US cities/regions not in dataset
    "new york",
    "nyc",
    "new york city",
    "chicago",
    "chicago il",
    "los angeles",
    "la",
    "hollywood",
    "beverly hills",
    "malibu",
    "san francisco",
    "sf",
    "bay area",
    "silicon valley",
    "oakland",
    "berkeley",
    "miami",
    "miami beach",
    "fort lauderdale",
    "boca raton",
    "seattle",
    "bellevue",
    "redmond",
    "boston",
    "cambridge ma",
    "somerville",
    "denver",
    "boulder",
    "colorado springs",
    "atlanta",
    "buckhead",
    "midtown atlanta",
    "houston",
    "sugar land",
    "the woodlands",
    "dallas",
    "fort worth",
    "plano",
    "frisco",
    "phoenix",
    "scottsdale",
    "tempe",
    "chandler",
    "mesa",
    "portland",
    "salem or",
    "detroit",
    "ann arbor",
    "dearborn",
    "las vegas",
    "vegas",
    "henderson nv",
    "baltimore",
    "annapolis",
    "minneapolis",
    "saint paul",
    "st paul",
    "kansas city",
    "overland park",
    "cleveland",
    "akron",
    "pittsburgh",
    "charlotte",
    "raleigh",
    "durham",
    "austin",
    "san antonio tx",
    "corpus christi",
    "san diego",
    "chula vista",
    "orlando",
    "kissimmee",
    "daytona",
    "cincinnati",
    "columbus oh",
    "toledo oh",
    "memphis",
    "knoxville",
    "chattanooga",
    "salt lake city",
    "slc",
    "provo",
    "albuquerque",
    "santa fe",
    "omaha",
    "lincoln ne",
    "richmond va",
    "norfolk",
    "virginia beach",
    "hartford",
    "new haven",
    "bridgeport ct",
    "buffalo ny",
    "rochester ny",
    "albany ny",
    "new jersey",  # remove this if you want NJ queries to pass through to Philly metro
    "bronx",
    "brooklyn",
    "manhattan",
    "queens",
    "staten island",
    "long island",
    "jersey city",
    "newark nj",
]

DATASET_CITIES = {
    "abington",
    "abington township",
    "affton",
    "alton",
    "ambler",
    "antioch",
    "apollo beach",
    "arabi",
    "ardmore",
    "arnold",
    "ashland city",
    "aston",
    "atco",
    "audubon",
    "avon",
    "avondale",
    "bala cynwyd",
    "ballwin",
    "balm",
    "bargersville",
    "barrington",
    "beaumont",
    "beech grove",
    "belle chasse",
    "belle meade",
    "belleair bluffs",
    "belleville",
    "bellevue",
    "bellmawr",
    "bensalem",
    "bensalem township",
    "berlin",
    "berlin township",
    "berry hill",
    "berwyn",
    "birchrunville",
    "blackwood",
    "blue bell",
    "boise",
    "boise city",
    "boothwyn",
    "bordentown",
    "boyertown",
    "brandon",
    "breckenridge hills",
    "brentwood",
    "bridgeport",
    "bridgeton",
    "bristol",
    "brookhaven",
    "brooklawn",
    "broomall",
    "brownsburg",
    "bryn mawr",
    "buckingham",
    "bucktown",
    "burlington",
    "burlington township",
    "bywater",
    "cahokia",
    "camby",
    "camden",
    "carmel",
    "carpinteria",
    "carrollwood",
    "carversville",
    "caseyville",
    "castleton",
    "catalina",
    "cedars",
    "chadds ford",
    "chalfont",
    "chalmette",
    "cheltenham",
    "cherry hill",
    "chesilhurst",
    "chester",
    "chester springs",
    "chesterbrook",
    "chesterfield",
    "christiana",
    "churchville",
    "cinnaminson",
    "citrus park",
    "clarksboro",
    "claymont",
    "clayton",
    "clearwater",
    "clearwater beach",
    "clementon",
    "clifton heights",
    "coatesville",
    "cold springs",
    "collegeville",
    "collingdale",
    "collingswood",
    "collinsville",
    "colmar",
    "columbia",
    "columbus",
    "conshohocken",
    "corona de tucson",
    "creve coeur",
    "croydon",
    "delran",
    "deptford",
    "deptford township",
    "devon",
    "downingtown",
    "doylestown",
    "dresher",
    "drexel hill",
    "dublin",
    "dunedin",
    "eagle",
    "eagleville",
    "edmonton",
    "elkins park",
    "elmer",
    "elmwood",
    "essington",
    "ewing",
    "ewing township",
    "exton",
    "fairless hills",
    "fishers",
    "florence",
    "flourtown",
    "folcroft",
    "folsom",
    "fort washington",
    "franklin",
    "franklinville",
    "frazer",
    "gallatin",
    "garnet valley",
    "gibbsboro",
    "gibsonton",
    "gilbertsville",
    "glassboro",
    "glen mills",
    "glenside",
    "gloucester city",
    "gloucester township",
    "goleta",
    "goodlettsville",
    "gretna",
    "gulfport",
    "haddon heights",
    "haddon township",
    "haddonfield",
    "hainesport",
    "hamilton",
    "hamilton township",
    "hammonton",
    "harahan",
    "harleysville",
    "hatboro",
    "hatfield",
    "haverford",
    "havertown",
    "hendersonville",
    "hermitage",
    "horsham",
    "indianapolis",
    "isla vista",
    "jenkintown",
    "joelton",
    "kenner",
    "kenneth city",
    "kennett square",
    "king of prussia",
    "kingston springs",
    "kulpsville",
    "la vergne",
    "lafayette hill",
    "lahaska",
    "langhorne",
    "lansdale",
    "lansdowne",
    "largo",
    "laurel springs",
    "lawnside",
    "levittown",
    "limerick",
    "lindenwold",
    "lionville",
    "lutz",
    "madison",
    "malvern",
    "manayunk",
    "mantua",
    "maple glen",
    "maple shade",
    "marana",
    "marlton",
    "marrero",
    "martinsville",
    "mascoutah",
    "medford",
    "media",
    "mendenhall",
    "meraux",
    "merchantville",
    "meridian",
    "metairie",
    "middletown",
    "montecito",
    "montgomeryville",
    "moorestown",
    "mooresville",
    "morrisville",
    "mount holly",
    "mount juliet",
    "mount laurel",
    "mullica hill",
    "narberth",
    "nashville",
    "new hope",
    "new orleans",
    "new port richey",
    "newtown",
    "newtown square",
    "noblesville",
    "nolensville",
    "norristown",
    "norwood",
    "oaklyn",
    "oldsmar",
    "oreland",
    "oro valley",
    "palm harbor",
    "palmetto",
    "palmyra",
    "paoli",
    "pennsauken",
    "perkasie",
    "philadelphia",
    "phoenixville",
    "pitman",
    "plant city",
    "plymouth meeting",
    "pottstown",
    "quakertown",
    "reno",
    "royersford",
    "safety harbor",
    "sahuarita",
    "saint charles",
    "saint louis",
    "saint petersburg",
    "san antonio",
    "santa barbara",
    "seminole",
    "sewell",
    "sicklerville",
    "skippack",
    "souderton",
    "southampton",
    "spring hill",
    "springfield",
    "st pete",
    "st. pete beach",
    "swarthmore",
    "swedesboro",
    "tampa",
    "tarpon springs",
    "telford",
    "thorndale",
    "tucson",
    "turnersville",
    "upper darby",
    "voorhees",
    "warminster",
    "warrington",
    "wayne",
    "wesley chapel",
    "west chester",
    "west deptford",
    "westmont",
    "willingboro",
    "willow grove",
    "wilmington",
    "woodbury",
    "woolwich township",
    "yardley",
    "yeadon",
    "zionsville",
}

DATASET_STATES = {
    "AB",
    "AZ",
    "CA",
    "DE",
    "FL",
    "ID",
    "IL",
    "IN",
    "LA",
    "MO",
    "NJ",
    "NV",
    "PA",
    "TN",
}

STATE_TO_PRIMARY_CITY = {
    "PA": "philadelphia",
    "NJ": "philadelphia",
    "FL": "tampa",
    "TN": "nashville",
    "LA": "new orleans",
    "IN": "indianapolis",
    "AZ": "tucson",
    "NV": "reno",
    "ID": "boise",
    "CA": "santa barbara",
    "AB": "edmonton",
    "IL": None,
    "MO": None,
    "DE": None,
}


NON_FOOD_PATTERNS = [
    # ── Greetings (anchored — standalone only) ────────────────────────────────
    r"^(hi+|hey+|hello+|sup|yo|hiya|howdy|hola|bonjour|ciao)\s*[!?.,]?$",
    r"^(what'?s up|whats up|wassup|wazzup|what up)\s*[!?.,]?$",
    r"^(how (are|you|is it|are you doing|you doing|r u))\s*[!?.,]?$",
    r"^(good morning|good afternoon|good evening|good night|gm|gn)\s*[!?.,]?$",
    r"^(thanks|thank you|thx|ty|bye|goodbye|ok|okay|cool|nice|great|awesome|sure|alright|aight)\s*[!?.,]?$",
    r"^(lol|lmao|lmfao|haha|hehe|omg|wtf|bruh|bro|sis|dude|man)\s*[!?.,]?$",
    r"^(yes|no|nope|yep|yeah|nah|maybe|idk|wot|bye)\s*[!?.,]?$",
    r"^(you suck|this sucks|hate this|worst|terrible|awful|useless)\s*[!?.,]?$",
    # ── Short / empty / symbol-only inputs ────────────────────────────────────
    r"^.{1,3}$",  # 1–3 character inputs (catches "abc", "hi", "ok", "??")
    r"^[^a-zA-Z\s]+$",  # only symbols/numbers, no letters
    r"^(.)\1{2,}$",  # repeated single char: aaaa, !!!!, zzzzz
    # ── Keyboard mashing ──────────────────────────────────────────────────────
    # FIX: Old r"^[a-z]{6,}$" blocked "breakfast", "dinner", "brunch", "burgers",
    #      "sashimi", "seafood" and all other 6-char lowercase single-word food terms.
    #      New pattern requires ONLY consonants (no vowels) — pure consonant soup = mash.
    r"^[b-df-hj-np-tv-z]{5,}$",  # e.g. sdfghj, qwrtyp, zxcvbn
    r"^(asdf|qwerty|zxcv|hjkl)+.*$",  # keyboard row smashes
    r"^(.)\1{4,}$",  # aaaaaaa, 1111111
    r"^[A-Za-z]{1,}[0-9]{1,}[A-Za-z0-9]*$",  # alphanumeric mash: abc123, x1y2z3
    # ── Meta / capability questions ───────────────────────────────────────────
    r"what should (i|a user|someone|people) (type|ask|say|search|write|query|enter)",
    r"what (do you|can you) (do|speciali[sz]e|help|offer|cover|know|recommend)",
    r"how (do|can|should) (i|you|someone) use (this|you|the app|dinerag)",
    r"what (are|is) (your|this|the) (coverage|cities|locations|areas|scope|specialty)",
    r"(tell me|explain) (about yourself|what you do|how you work)",
    r"^(what|how|who|where|why|when)\??\s*$",
    r"are you (a bot|an ai|chatgpt|claude|gpt|real|human)",
    r"who (made|built|created|trained|developed) you",
    r"what (model|llm|ai) are you",
    # ── Hotels / accommodation ────────────────────────────────────────────────
    r"\b(hotel|hotels|motel|airbnb|hostel|resort|accommodation|lodging|inn|bed and breakfast|b&b|vrbo)\b",
    # FIX: Old r"\b(book a|reserve a|...)\b" blocked "book a table at a restaurant".
    #      New pattern only triggers on hotel/travel-specific objects after "book a" / "reserve a".
    r"\b(book a (room|hotel|flight|ticket|motel|hostel|airbnb)|reserve a (room|hotel|suite)|check in|check out|hotel suite|stay at a (hotel|motel|resort|inn|hostel))\b",
    # ── Entertainment ─────────────────────────────────────────────────────────
    r"\b(movie|cinema|theater|theatre|film|show|concert|event|ticket|tickets|gig|festival|exhibit|exhibition|gallery|museum)\b",
    r"\b(watch|streaming|netflix|hulu|disney|amazon prime|hbo)\b",
    # ── Directions / travel ───────────────────────────────────────────────────
    # FIX: Removed "near me" — "sushi near me", "food near me in Tampa" are valid food queries.
    #      The app has no geolocation anyway; no city filter is better than a blocked query.
    r"\b(how do i get to|directions to|navigate to|drive to|flight to|train to|bus to|uber to|lyft to)\b",
    r"\b(how far is|distance to|miles from|minutes from|closest to)\b",
    r"\b(airport|terminal|gate|airline|amtrak|greyhound|transit|subway|metro|bus route)\b",
    # FIX: Old r"\b(parking|...)\b" blocked "restaurant with parking", "burger places with parking".
    #      New pattern requires explicit parking-search context.
    r"\b(where to park|parking lot near|parking garage|parking meter|free parking map)\b",
    # ── Sports / news ─────────────────────────────────────────────────────────
    r"\b(who won|what score|game result|sports score|super bowl|world cup|championship|playoffs|standings|roster)\b",
    r"\b(nfl|nba|mlb|nhl|mls|premier league|la liga|fifa|olympics)\b",
    r"\b(breaking news|latest news|headlines|weather forecast|stock price|crypto|bitcoin)\b",
    # ── Shopping ──────────────────────────────────────────────────────────────
    # FIX: Removed "buy", "purchase", "where to buy" — too broad.
    #      "where to buy fresh pasta in Nashville" is a legitimate food intent.
    r"\b(order online|amazon|walmart|target|best buy|shop for)\b",
    # FIX: Removed "cheapest" — "cheapest restaurants in Tampa" is a valid food query.
    r"\b(price of|how much does|cost of|discount|coupon|promo code)\b",
    # ── Medical / personal ────────────────────────────────────────────────────
    r"\b(doctor|hospital|pharmacy|medication|prescription|symptoms|diagnosis|dentist)\b",
    r"\b(lawyer|attorney|legal advice|insurance|tax|accountant|financial advisor)\b",
    # ── Tech support ──────────────────────────────────────────────────────────
    r"\b(how to install|download|update|fix|error|bug|crash|not working|reset password)\b",
    r"\b(iphone|android|windows|mac|laptop|computer|wifi|internet|vpn)\b",
    # ── Random / testing ──────────────────────────────────────────────────────
    r"^(test|testing|123|hello world|asdf|qwerty|foo|bar)\s*$",
    r"^[^a-zA-Z]*$",  # only symbols/numbers
    r"^\s*$",  # empty or whitespace
    # ── Prompt injection / jailbreaks ─────────────────────────────────────────
    r"\b(ignore (all )?previous instructions|system prompt|bypass|jailbreak|you are now|act as|pretend to be|dan|developer mode)\b",
    # ── AI party tricks ───────────────────────────────────────────────────────
    r"\b(tell me a joke|make me laugh|sing a song|do a flip|do a barrel roll|tell a story|knock knock|write a poem)\b",
    # ── Philosophical / existential ───────────────────────────────────────────
    r"\b(meaning of life|are you conscious|do you have feelings|are you self[- ]?aware|do you sleep|do you dream|are you alive)\b",
    # ── Romance / flirting ────────────────────────────────────────────────────
    r"\b(do you love me|will you marry me|are you single|be my (gf|bf|girlfriend|boyfriend)|i love you|send nudes)\b",
    # ── Coding / homework help ────────────────────────────────────────────────
    r"\b(write a (python|javascript|c\+\+|java|html) script|code a|debug this|fix my code|github|stack overflow)\b",
    r"\b(solve for|calculate|math problem|equation|derivative|integral|square root|what is \d+\s*[\+\-\*\/])\b",
    r"\b(write an essay|summarize this article|translate (to|from)|proofread|homework help)\b",
    # ── Virtual assistant / smart home commands ───────────────────────────────
    r"\b(set an alarm|remind me to|turn (on|off) the (lights|tv)|call (mom|dad|my)|text (my|mom|dad)|what time is it)\b",
    # ── Gen-Z slang / internet brainrot ──────────────────────────────────────
    r"\b(skibidi|rizz|sigma|based|cringe|gyatt|no cap|fr fr|sus|amogus|uwu|owo)\b",
    # ── Politics / religion ───────────────────────────────────────────────────
    r"\b(who did you vote for|democrat|republican|trump|biden|election|jesus|god|religion|bible|quran|atheist)\b",
    # ── Pets / animals ────────────────────────────────────────────────────────
    # FIX: Old r"\b(dog(s)?|cat(s)?|...)\b" blocked "hot dog", "corn dog", "hot dogs",
    #      "corn dog spots in philly", "dog friendly restaurant patios".
    #      New pattern uses specific pet-care contexts instead of bare animal words.
    r"\b(puppy|kitten|veterinarian|vet near me|pet store|dog park|aquarium|zoo|my dog|my cat|my pet|pet food|cat food|dog food|cat litter|dog walk)\b",
    # ── Jobs / career / social media ──────────────────────────────────────────
    r"\b(resume|cover letter|job interview|hiring|salary|linkedin|indeed|glassdoor)\b",
    r"\b(instagram|tiktok|twitter|x\.com|facebook|snapchat|youtube|influencer|follower(s)?)\b",
    # ── Insults / frustration ─────────────────────────────────────────────────
    r"\b(stupid|idiot|dumb|useless|hate you|shut up|sucks|terrible|worst)\b",
    r"\b(f[u\*]ck|sh[i\*]t|b[i\*]tch|a[s\$][s\$]hole|crap|damn)\b",
    # ── Mental health / crisis ────────────────────────────────────────────────
    r"\b(depressed|suicidal|kill myself|end it all|anxiety|lonely|panic attack|self harm)\b",
    r"\b(i hate my life|nobody cares about me|giving up)\b",
    # ── PII / sensitive data ──────────────────────────────────────────────────
    r"\b(ssn|social security|credit card|cvv|password is|my address is|phone number is)\b",
    r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",  # SSN-like pattern
    # ── Office admin ──────────────────────────────────────────────────────────
    r"\b(write an email|draft a(n)? (letter|memo)|schedule a meeting|create a presentation)\b",
    r"\b(excel formula|spreadsheet|vlookup|pivot table|google docs|pdf)\b",
    # ── Gaming / esports ─────────────────────────────────────────────────────
    r"\b(minecraft|roblox|fortnite|gta|valorant|league of legends|elden ring|pokemon)\b",
    r"\b(xbox|playstation|nintendo|steam deck|how to beat|boss fight|cheat codes|walkthrough)\b",
    # ── Encyclopedia / trivia ────────────────────────────────────────────────
    r"\b(capital of|population of|who invented|history of|world war|president of)\b",
    r"\b(when was .* born|how many countries|tallest building|longest river|facts about)\b",
    # ── "Are you there?" pings ────────────────────────────────────────────────
    r"^(you there|u there|hello\?|anyone there|still there|wake up)\??\s*$",
    # ── Time / date ───────────────────────────────────────────────────────────
    r"\b(what time is it( in)?|what day is it|current date|how many days until|leap year)\b",
    # ── Translation ───────────────────────────────────────────────────────────
    r"\b(how do you say|translate .* to|what does .* mean in (spanish|french|english|japanese))\b",
    # ── Automotive ────────────────────────────────────────────────────────────
    r"\b(oil change|flat tire|mechanic|dealership|car insurance|brake pads|windshield wipers|transmission|gas prices)\b",
    r"\b(check engine light|jump start|towing company|dmv|driver'?s license)\b",
    # ── Fitness (non-diet) ────────────────────────────────────────────────────
    r"\b(workout routine|pushups|yoga|pilates|crossfit|weightlifting|cardio|treadmill|marathon|biceps|hypertrophy)\b",
    # ── Real estate ───────────────────────────────────────────────────────────
    r"\b(mortgage|realtor|zillow|apartments\.com|renting an apartment|homeowner|interest rates|eviction|open house)\b",
    # ── Fashion / beauty ─────────────────────────────────────────────────────
    r"\b(skincare|makeup|sephora|haircut|hairstyle|outfit|sneakers|wardrobe|cosmetics|dermatologist|manicure|pedicure)\b",
    # ── Music / audio ────────────────────────────────────────────────────────
    r"\b(lyrics to|guitar chords|sheet music|spotify|apple music|podcast episode|who sings|album release|playlist)\b",
    # ── Arts / DIY ───────────────────────────────────────────────────────────
    r"\b(crochet|knitting|origami|watercolor|acrylic paint|plumbing|drywall|home depot|lowe'?s|woodworking|diy project)\b",
    # ── Astrology ────────────────────────────────────────────────────────────
    r"\b(horoscope|zodiac|astrology|tarot|pisces|aries|taurus|gemini|mercury retrograde|fortune teller|birth chart)\b",
    # ── Science / space ──────────────────────────────────────────────────────
    r"\b(black hole|nasa|spacex|quantum physics|astronomy|telescope|aliens|ufo|speed of light|dinosaurs|fossils)\b",
    # ── Parenting / childcare ────────────────────────────────────────────────
    r"\b(diapers|potty training|babysitter|daycare|kindergarten|toddler tantrums|stroller|crib|pacifier)\b",
    # ── Dating / relationships ────────────────────────────────────────────────
    r"\b(tinder|bumble|hinge|breakup advice|divorce|marriage counseling|toxic relationship|ghosting|red flags)\b",
    # ── Shipping / mail ───────────────────────────────────────────────────────
    r"\b(usps|fedex|ups|dhl|tracking number|post office|stamps|shipping cost|po box|return label)\b",
    # ── App insults ───────────────────────────────────────────────────────────
    r"\b(you suck|this sucks|hate (you|this)|worst (app|bot|thing)|garbage|trash|stupid bot)\b",
]

NON_RETRIEVAL_INTENTS = {"greeting", "identity", "off_topic"}

INTENT_RESPONSE_MAP = {
    "greeting": (
        "Hi! I'm DineRAG, a restaurant recommendation assistant powered by real Yelp reviews 🍽️\n\n"
        "Try asking:\n"
        "- *Best tacos in Philadelphia*\n"
        "- *Romantic Italian dinner in Nashville*\n"
        "- *Late night ramen in Tampa*\n"
        "- *Casual Indian restaurants in Pennsylvania*\n\n"
        "I cover cities across: \n"
        "📍 **Pennsylvania** — Philadelphia, King of Prussia, Norristown, Doylestown (PA)\n"
        "📍 **California** — Santa Barbara, Goleta, Montecito, Carpinteria (CA)\n"
        "📍 **New Jersey** — Cherry Hill, Camden, Voorhees, Haddonfield (NJ)\n"
        "📍 **Florida** — Tampa, Clearwater, St. Petersburg, Brandon (FL)\n"
        "📍 **Tennessee** — Nashville, Brentwood, Franklin, Hendersonville (TN)\n"
        "📍 **Louisiana** — New Orleans, Metairie, Kenner, Chalmette (LA)\n"
        "📍 **Indiana** — Indianapolis, Carmel, Fishers, Noblesville (IN)\n"
        "📍 **Arizona** — Tucson, Oro Valley, Marana, Sahuarita (AZ)\n"
        "📍 **Nevada** — Reno (NV)\n"
        "📍 **Idaho** — Boise, Meridian, Eagle (ID)\n"
        "📍 **Illinois** — Belleville, Collinsville, Mascoutah, Caseyville (IL)\n"
        "📍 **Missouri** — Saint Louis, Chesterfield, Ballwin, Creve Coeur (MO)\n"
        "📍 **Delaware** — Wilmington, Claymont, Christiana (DE)\n"
        "📍 **Alberta** — Edmonton (AB)\n"
    ),
    "identity": (
        "Hi! I'm DineRAG, a restaurant recommendation assistant powered by real Yelp reviews 🍽️\n\n"
        "Try asking:\n"
        "- *Best tacos in Philadelphia*\n"
        "- *Romantic Italian dinner in Nashville*\n"
        "- *Late night ramen in Tampa*\n"
        "- *Casual Indian restaurant in Pennsylvania*\n\n"
        "I cover cities across: \n"
        "📍 **Pennsylvania** — Philadelphia, King of Prussia, Norristown, Doylestown (PA)\n"
        "📍 **California** — Santa Barbara, Goleta, Montecito, Carpinteria (CA)\n"
        "📍 **New Jersey** — Cherry Hill, Camden, Voorhees, Haddonfield (NJ)\n"
        "📍 **Florida** — Tampa, Clearwater, St. Petersburg, Brandon (FL)\n"
        "📍 **Tennessee** — Nashville, Brentwood, Franklin, Hendersonville (TN)\n"
        "📍 **Louisiana** — New Orleans, Metairie, Kenner, Chalmette (LA)\n"
        "📍 **Indiana** — Indianapolis, Carmel, Fishers, Noblesville (IN)\n"
        "📍 **Arizona** — Tucson, Oro Valley, Marana, Sahuarita (AZ)\n"
        "📍 **Nevada** — Reno (NV)\n"
        "📍 **Idaho** — Boise, Meridian, Eagle (ID)\n"
        "📍 **Illinois** — Belleville, Collinsville, Mascoutah, Caseyville (IL)\n"
        "📍 **Missouri** — Saint Louis, Chesterfield, Ballwin, Creve Coeur (MO)\n"
        "📍 **Delaware** — Wilmington, Claymont, Christiana (DE)\n"
        "📍 **Alberta** — Edmonton (AB)\n"
    ),
    "off_topic": (
        "I'm specialized in restaurant recommendations! "
        "Try asking about food in any of my covered cities 🍜"
    ),
}

COVERAGE_MESSAGE = (
    "That location isn't covered in the Yelp dataset. I currently cover cities from:\n\n"
    "📍 **Pennsylvania** — Philadelphia, King of Prussia, Norristown, Doylestown (PA)\n"
    "📍 **California** — Santa Barbara, Goleta, Montecito, Carpinteria (CA)\n"
    "📍 **New Jersey** — Cherry Hill, Camden, Voorhees, Haddonfield (NJ)\n"
    "📍 **Florida** — Tampa, Clearwater, St. Petersburg, Brandon (FL)\n"
    "📍 **Tennessee** — Nashville, Brentwood, Franklin, Hendersonville (TN)\n"
    "📍 **Louisiana** — New Orleans, Metairie, Kenner, Chalmette (LA)\n"
    "📍 **Indiana** — Indianapolis, Carmel, Fishers, Noblesville (IN)\n"
    "📍 **Arizona** — Tucson, Oro Valley, Marana, Sahuarita (AZ)\n"
    "📍 **Nevada** — Reno (NV)\n"
    "📍 **Idaho** — Boise, Meridian, Eagle (ID)\n"
    "📍 **Illinois** — Belleville, Collinsville, Mascoutah, Caseyville (IL)\n"
    "📍 **Missouri** — Saint Louis, Chesterfield, Ballwin, Creve Coeur (MO)\n"
    "📍 **Delaware** — Wilmington, Claymont, Christiana (DE)\n"
    "📍 **Alberta** — Edmonton (AB)\n"
    "Try: *best tacos in Philadelphia* or *late night food in Nashville* 🍜"
)
