"""Unit tests for the privacy layer. They use synthetic values only, never real data."""

from datetime import date

import pandas as pd
import pytest

from customer_segmentation.privacy import (
    PII_COLUMNS,
    birthdate_to_band,
    normalize_email,
    normalize_phone,
    pseudonymize,
    pseudonymize_customers,
)

KEY = b"x" * 64
REF = date(2026, 9, 30)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ana@example.com", "ana@example.com"),
        ("  Ana@Example.COM ", "ana@example.com"),
        ("ana@@example.com", None),
        ("ana@example", None),
        ("ana@", None),
        ("", None),
        (None, None),
    ],
)
def test_normalize_email(raw, expected):
    assert normalize_email(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("+52 55 1234 5678", "5512345678"),
        ("5512345678", "5512345678"),
        ("(55) 1234-5678", "5512345678"),
        ("525512345678", "5512345678"),
        ("5215512345678", "5512345678"),
        ("551 234 5678", "5512345678"),
        ("+52 55", None),
        ("N/A", None),
        ("12", None),
        (None, None),
    ],
)
def test_normalize_phone(raw, expected):
    assert normalize_phone(raw) == expected


def test_pseudonymize_is_deterministic_and_key_dependent():
    assert pseudonymize("a@b.com", KEY) == pseudonymize("a@b.com", KEY)
    assert pseudonymize("a@b.com", KEY) != pseudonymize("a@b.com", b"y" * 64)
    assert pseudonymize(None, KEY) is None
    assert pseudonymize(float("nan"), KEY) is None


def test_case_variants_of_same_email_get_same_token():
    tokens = {pseudonymize(normalize_email(e), KEY) for e in ["a@b.com", " A@B.com", "a@B.COM"]}
    assert len(tokens) == 1


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1990-10-01", ("35-44", "ok")),  # turns 36 the day after REF
        ("1990-09-30", ("35-44", "ok")),
        ("2008-06-07", ("16-24", "ok")),
        ("2020-01-01", (None, "out_of_range")),
        ("1900-01-01", (None, "out_of_range")),
        ("07/06/1990", (None, "unparseable")),
        ("", (None, "missing")),
        (None, (None, "missing")),
    ],
)
def test_birthdate_to_band(raw, expected):
    assert birthdate_to_band(raw, REF) == expected


def test_output_contains_no_pii_columns():
    raw = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "email": ["ana@marcamoda.mx"],
            "telefono": ["5512345678"],
            "fecha_nacimiento": ["1990-01-01"],
            "ciudad": ["León"],
            "fecha_registro": ["2024-01-01"],
            "canal_registro": ["online"],
            "opt_in_email": ["true"],
            "opt_in_whatsapp": ["false"],
            "notas_crm": ["free text"],
        }
    )
    out = pseudonymize_customers(raw, KEY, REF)
    assert not set(PII_COLUMNS) & set(out.columns)
    row = out.iloc[0]
    assert row["is_internal_email"]
    assert row["has_crm_note"]
    assert row["age_band"] == "35-44"
