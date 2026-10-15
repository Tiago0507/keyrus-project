"""Privacy layer: the only module that reads raw personal data.

It turns data/raw/customers.csv into data/processed/customers_pseudo.parquet:
- email and phone are normalized, then replaced by a keyed hash (HMAC-SHA256),
  so duplicates can still be detected downstream without exposing the values;
- birthdate is reduced to an age band;
- the free-text column notas_crm is dropped (only a has_crm_note flag is kept);
- data-quality flags are computed here, because raw values are gone afterwards.

It also writes an aggregated quality report (counts only, no personal data).

Run:  uv run python -m customer_segmentation.privacy
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from datetime import date

import pandas as pd
from dotenv import load_dotenv

from customer_segmentation import config

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
HASH_LENGTH = 16  # hex chars (64 bits): collision-free in practice for thousands of rows

# Columns removed from the analytical dataset (direct identifiers and free text).
PII_COLUMNS = ["email", "telefono", "fecha_nacimiento", "notas_crm"]


def normalize_email(raw: str | None) -> str | None:
    """Trim and lowercase an email. Return None if it is not a valid address."""
    if raw is None or pd.isna(raw):
        return None
    email = str(raw).strip().lower()
    return email if EMAIL_RE.match(email) else None


def normalize_phone(raw: str | None) -> str | None:
    """Reduce a Mexican phone number to its 10 national digits. None if not valid.

    Accepts '+52 55 1234 5678', '5512345678', '(55) 1234-5678', '525512345678'
    and the legacy mobile prefix '521...'.
    """
    if raw is None or pd.isna(raw):
        return None
    digits = re.sub(r"\D", "", str(raw))
    if len(digits) == 13 and digits.startswith("521"):
        digits = digits[3:]
    elif len(digits) == 12 and digits.startswith("52"):
        digits = digits[2:]
    return digits if len(digits) == 10 else None


def pseudonymize(value: str | None, key: bytes) -> str | None:
    """Keyed hash of an already-normalized value: same input and key, same token."""
    if value is None or pd.isna(value):
        return None
    digest = hmac.new(key, value.encode("utf-8"), hashlib.sha256).hexdigest()
    return digest[:HASH_LENGTH]


def age_at(birthdate: date, reference: date) -> int:
    return (
        reference.year
        - birthdate.year
        - ((reference.month, reference.day) < (birthdate.month, birthdate.day))
    )


def birthdate_to_band(raw: str | None, reference: date) -> tuple[str | None, str]:
    """Return (age_band, status). Status is one of: ok, missing, unparseable, out_of_range."""
    if raw is None or pd.isna(raw) or not str(raw).strip():
        return None, "missing"
    parsed = pd.to_datetime(str(raw).strip(), format="%Y-%m-%d", errors="coerce")
    if pd.isna(parsed):
        return None, "unparseable"
    age = age_at(parsed.date(), reference)
    if not config.MIN_VALID_AGE <= age <= config.MAX_VALID_AGE:
        return None, "out_of_range"
    band = next(label for low, high, label in config.AGE_BANDS if low <= age <= high)
    return band, "ok"


def load_key() -> bytes:
    load_dotenv(config.PROJECT_ROOT / ".env")
    key = os.getenv("PSEUDONYMIZATION_KEY", "").strip()
    if len(key) < 32:
        raise SystemExit(
            "PSEUDONYMIZATION_KEY is missing or too short. Copy .env.example to .env and set it."
        )
    return key.encode("utf-8")


def pseudonymize_customers(raw: pd.DataFrame, key: bytes, reference: date) -> pd.DataFrame:
    """Build the analytical customer table. No direct identifier survives."""
    out = raw.drop(columns=PII_COLUMNS).copy()

    emails = raw["email"].map(normalize_email)
    phones = raw["telefono"].map(normalize_phone)
    out["email_hash"] = emails.map(lambda e: pseudonymize(e, key))
    out["phone_hash"] = phones.map(lambda p: pseudonymize(p, key))
    out["email_valid"] = emails.notna()
    out["phone_valid"] = phones.notna()
    out["is_internal_email"] = emails.map(
        lambda e: isinstance(e, str) and e.split("@")[1] in config.INTERNAL_EMAIL_DOMAINS
    )

    bands = raw["fecha_nacimiento"].map(lambda b: birthdate_to_band(b, reference))
    out["age_band"] = bands.str[0]
    out["birthdate_status"] = bands.str[1]

    out["has_crm_note"] = raw["notas_crm"].fillna("").str.strip().ne("")
    return out


def quality_report(raw: pd.DataFrame, pseudo: pd.DataFrame) -> dict:
    """Aggregated counts only: safe to share, print or send to an AI assistant."""

    def dup_rows(col: str) -> int:
        s = pseudo[col].dropna()
        return int(s.duplicated(keep=False).sum())

    return {
        "rows": len(raw),
        "unique_customer_id": int(raw["customer_id"].nunique()),
        "email_invalid": int((~pseudo["email_valid"]).sum()),
        "email_needed_normalization": int(
            (raw["email"].fillna("") != raw["email"].fillna("").str.strip().str.lower()).sum()
        ),
        "email_internal_domain": int(pseudo["is_internal_email"].sum()),
        "phone_invalid": int((~pseudo["phone_valid"]).sum()),
        "rows_sharing_email_hash": dup_rows("email_hash"),
        "rows_sharing_phone_hash": dup_rows("phone_hash"),
        "birthdate_status": pseudo["birthdate_status"].value_counts().to_dict(),
        "age_band": pseudo["age_band"].value_counts(dropna=False).sort_index().to_dict(),
        "crm_notes_dropped": int(pseudo["has_crm_note"].sum()),
    }


def main() -> None:
    key = load_key()
    raw = pd.read_csv(
        config.RAW_DIR / "customers.csv", dtype=str, keep_default_na=False, na_values=[""]
    )
    pseudo = pseudonymize_customers(raw, key, config.REFERENCE_DATE)

    leaked = set(PII_COLUMNS) & set(pseudo.columns)
    if leaked:  # explicit check: `assert` would be skipped under `python -O`
        raise RuntimeError(f"PII columns leaked into output: {leaked}")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pseudo.to_parquet(config.PROCESSED_DIR / "customers_pseudo.parquet", index=False)

    report = quality_report(raw, pseudo)
    report_path = config.REPORTS_DIR / "privacy_quality_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
