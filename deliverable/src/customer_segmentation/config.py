"""Project-wide paths and constants. Every analytical assumption lives here, in one place."""

from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # original CSVs; customers.csv (PII) is read only by privacy.py
PROCESSED_DIR = DATA_DIR / "processed"  # pseudonymized data, safe for analysis
REPORTS_DIR = DATA_DIR / "reports"  # aggregated data-quality reports

# Snapshot date for every "as of" calculation (age, recency).
# Last order in the dataset is 2026-09-30, so we analyse as of that day.
REFERENCE_DATE = date(2026, 9, 30)

# Email domain that belongs to the brand itself: likely staff or test accounts.
INTERNAL_EMAIL_DOMAINS = frozenset({"marcamoda.mx"})

# Plausible customer age range; outside it the birthdate is treated as invalid.
MIN_VALID_AGE = 16
MAX_VALID_AGE = 100

AGE_BANDS = [
    (16, 24, "16-24"),
    (25, 34, "25-34"),
    (35, 44, "35-44"),
    (45, 54, "45-54"),
    (55, 64, "55-64"),
    (65, 100, "65+"),
]

# Cleaning rules.
NET_SALE_STATUS = "completada"  # only completed orders count as net sales
BASICS_COLLECTION = "BASICOS"  # timeless products: no launch date by design
UNKNOWN_COLLECTION = "UNKNOWN"  # SKUs sold but missing from the catalogue

# A retail customer never buys more than 3 units of one SKU in this data. Lines above this
# threshold are bulk purchases and their buyer is excluded from customer-level analysis.
MAX_UNITS_PER_LINE = 10

# Customer features.
# A collection sells about 34% of its lifetime net sales in the first 30 days after launch,
# about 3x a normal month (11%). Buying inside that window is the signal of interest in new
# collections.
LAUNCH_WINDOW_DAYS = 30

# "Recent" activity window for current-value features.
RECENT_MONTHS = 12

# Business question criteria. Each threshold sits in a natural gap of the data
# (see the sensitivity table in the notebook).
# Q1 new collections: bought within the launch window in at least half of the launches the
# customer could have bought (participation is bimodal: most customers are at 0% or 100%).
NEW_COLLECTION_MIN_PARTICIPATION = 0.5

# Q2 recurrence: buys every two months or more often, and is still active. Median gaps cluster
# at 30-60 and 90-270 days with a valley at 60-90; recency of 90 days = 1.5x the rhythm.
RECURRENT_MIN_PURCHASE_DAYS = 3  # needs at least two gaps to speak of a rhythm
RECURRENT_MAX_MEDIAN_GAP_DAYS = 60
RECURRENT_MAX_RECENCY_DAYS = 90

# Q3 high value: net sales in the last 12 months. Almost no customer spends 20k-30k MXN,
# which separates a distinct high-spend group above it.
HIGH_VALUE_MIN_SALES_12M = 25_000
