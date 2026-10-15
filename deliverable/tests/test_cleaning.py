"""Tests for the cleaning rules and the impact of cleaning.

Each row of the synthetic data below is one test case:
- C1 and C9 are the same person (same email token), C2 is a staff account,
  C3 is a bulk buyer and C4 registered but never bought.
- O1 has an exact duplicate line, O2 has the same SKU with two discounts,
  O3 is a guest order and O4 a returned order with an unknown customer ID.
"""

import pandas as pd
import pytest

from customer_segmentation.cleaning import (
    IssueLog,
    clean_items,
    clean_orders,
    clean_products,
    compare,
    flag_bulk_buyers,
    normalize_collection,
    resolve_customers,
    run,
    sales_bridge,
)


def make_customers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C9", "C2", "C3", "C4"],
            "ciudad": ["León", "León", "Puebla", "Mérida", "León"],
            "fecha_registro": [
                "2024-01-01",
                "2024-01-01",
                "2024-02-01",
                "2024-03-01",
                "2024-04-01",
            ],
            "canal_registro": ["online", "tienda", "tienda", "online", "tienda"],
            "opt_in_email": ["true", "true", "true", "false", "true"],
            "opt_in_whatsapp": ["true", "false", "true", "false", "true"],
            "email_hash": ["aaa", "aaa", "bbb", None, None],
            "phone_hash": ["p1", "p1", "p2", "p3", "p4"],
            "email_valid": [True, True, True, False, False],
            "phone_valid": [True, True, True, True, True],
            "is_internal_email": [False, False, True, False, False],
            "age_band": ["25-34", None, "35-44", None, "16-24"],
            "birthdate_status": ["ok", "missing", "ok", "missing", "ok"],
            "has_crm_note": [False, True, False, False, False],
        }
    )


def make_products() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sku": ["S1", "S2", "S3"],
            "categoria": ["Vestidos", "Blusas", "Playeras básicas"],
            "coleccion": ["OI-26", "PV26", "BASICOS"],
            "fecha_lanzamiento": ["2026-08-18", "2026-02-12", None],
            "precio_lista": ["1000", "500", "200"],
        }
    )


def make_orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["O1", "O2", "O3", "O4", "O5"],
            "customer_id": ["C1", "C9", None, "C7", "C3"],
            "fecha": ["2026-09-01", "2026-01-01", "2026-03-01", "2026-03-01", "2026-04-01"],
            "canal": ["online", "tienda", "tienda", "online", "tienda"],
            "tienda_id": [None, "T01", "T02", None, "T01"],
            "estado": ["completada", "completada", "completada", "devuelta", "completada"],
        }
    )


def make_items() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["O1", "O1", "O2", "O2", "O3", "O5"],
            "sku": ["S1", "S1", "S1", "S1", "S999", "S3"],
            "cantidad": ["1", "1", "1", "1", "1", "50"],
            "precio_unitario": ["1000", "1000", "1000", "1000", "300", "200"],
            "descuento_pct": ["0", "0", "20", "0", "0", "0"],
        }
    )


def test_normalize_collection():
    labels = pd.Series(["OI-26", "oi26", "OI 26", "Oi26", "OI26 ", "BASICOS"])
    assert normalize_collection(labels).tolist() == ["OI26"] * 5 + ["BASICOS"]


def test_clean_products_flags_basics_as_not_seasonal():
    p = clean_products(make_products(), IssueLog())
    assert p.set_index("sku")["is_seasonal"].to_dict() == {"S1": True, "S2": True, "S3": False}


def test_resolve_customers_merges_duplicates_into_original_id():
    customers, id_map = resolve_customers(make_customers(), IssueLog())
    assert id_map["C9"] == "C1"
    assert set(customers["customer_id"]) == {"C1", "C2", "C3", "C4"}
    c1 = customers.set_index("customer_id").loc["C1"]
    assert c1["n_source_records"] == 2
    assert c1["canal_registro"] == "both"
    assert c1["age_band"] == "25-34"
    assert c1["has_crm_note"]


def test_merged_consent_is_most_restrictive():
    customers, _ = resolve_customers(make_customers(), IssueLog())
    c1 = customers.set_index("customer_id").loc["C1"]
    assert c1["opt_in_email"]  # both records said yes
    assert not c1["opt_in_whatsapp"]  # one record said no


def test_customer_without_email_is_not_merged_with_others():
    customers, id_map = resolve_customers(make_customers(), IssueLog())
    assert id_map["C3"] == "C3"


def test_internal_accounts_are_excluded():
    customers, _ = resolve_customers(make_customers(), IssueLog())
    reasons = customers.set_index("customer_id")["exclusion_reason"]
    assert reasons["C2"] == "internal_account"
    assert pd.isna(reasons["C1"])


def test_clean_orders_statuses_and_reassignment():
    customers, id_map = resolve_customers(make_customers(), IssueLog())
    o = clean_orders(make_orders(), customers, id_map, IssueLog()).set_index("order_id")
    assert o.loc["O2", "customer_id"] == "C1"  # reassigned from duplicate C9
    assert o.loc["O2", "source_customer_id"] == "C9"
    assert o.loc["O3", "customer_status"] == "guest"
    assert o.loc["O4", "customer_status"] == "unknown_id"
    assert not o.loc["O4", "is_net_sale"]  # returned


def test_orders_before_registration_are_logged_not_changed():
    customers, id_map = resolve_customers(make_customers(), IssueLog())
    raw = make_orders()
    raw.loc[raw["order_id"] == "O2", "fecha"] = "2023-12-01"  # C1 registered on 2024-01-01
    log = IssueLog()
    o = clean_orders(raw, customers, id_map, log)
    issues = log.to_frame().set_index("code")
    assert issues.loc["O3", "rows_affected"] == 1
    assert len(o) == len(raw)
    assert o.set_index("order_id").loc["O2", "fecha"] == pd.Timestamp("2023-12-01")


def test_clean_items_rules():
    products = clean_products(make_products(), IssueLog())
    customers, id_map = resolve_customers(make_customers(), IssueLog())
    orders = clean_orders(make_orders(), customers, id_map, IssueLog())
    i = clean_items(make_items(), products, orders, IssueLog())

    assert (i["order_id"] == "O1").sum() == 1  # exact duplicate dropped
    assert (i["order_id"] == "O2").sum() == 2  # same SKU, different discount: kept
    assert sorted(i.loc[i["order_id"] == "O2", "net_amount"]) == [800.0, 1000.0]
    assert i.loc[i["sku"] == "S999", "coleccion"].item() == "UNKNOWN"
    assert i.loc[i["sku"] == "S1", "coleccion"].unique().tolist() == ["OI26"]
    # O2 (2026-01-01) is before S1's launch (2026-08-18); O1 (2026-09-01) is after it
    assert i.loc[i["order_id"] == "O2", "pre_launch"].all()
    assert not i.loc[i["order_id"] == "O1", "pre_launch"].any()
    assert not i.loc[i["sku"] == "S3", "pre_launch"].any()  # basics never pre-launch


def test_bulk_buyer_is_flagged_by_rule():
    log = IssueLog()
    products = clean_products(make_products(), log)
    customers, id_map = resolve_customers(make_customers(), log)
    orders = clean_orders(make_orders(), customers, id_map, log)
    items = clean_items(make_items(), products, orders, log)
    customers = flag_bulk_buyers(customers, orders, items, log)
    assert customers.set_index("customer_id").loc["C3", "exclusion_reason"] == "bulk_buyer"


def test_run_defines_customer_scope():
    result = run(make_customers(), make_orders(), make_items(), make_products())
    scope = result["orders"].set_index("order_id")["in_customer_scope"]
    assert scope.to_dict() == {"O1": True, "O2": True, "O3": False, "O4": False, "O5": False}
    assert {
        "C1",
        "C1b",
        "C1c",
        "C4",
        "O1",
        "O2",
        "O3",
        "O4",
        "I1",
        "I2",
        "I3",
        "I4",
        "P1",
        "P2",
        "P3",
    } == set(result["issue_log"]["code"])


@pytest.fixture
def result():
    return run(make_customers(), make_orders(), make_items(), make_products())


def test_sales_bridge_reconciles(result):
    bridge = sales_bridge(make_items(), result).set_index("step")["amount_mxn"]
    deltas = ["Duplicated lines", "Returned and cancelled orders", "Discounts"]
    assert bridge["Raw gross sales (as loaded)"] + bridge[deltas].sum() == pytest.approx(
        bridge["= Brand net sales"]
    )
    removed = ["Guest purchases (no customer)", "Unknown customer IDs", "Internal accounts",
               "Bulk buyer"]  # fmt: skip
    assert bridge["= Brand net sales"] + bridge[removed].sum() == pytest.approx(
        bridge["= Net sales in customer analysis"]
    )


def test_sales_bridge_values(result):
    bridge = sales_bridge(make_items(), result).set_index("step")["amount_mxn"]
    # raw: O1 2x1000, O2 2x1000, O3 300, O5 50x200 = 14,300
    assert bridge["Raw gross sales (as loaded)"] == 14_300
    assert bridge["Duplicated lines"] == -1_000  # second O1 line
    assert bridge["Discounts"] == -200  # 20% on one O2 line
    assert bridge["Bulk buyer"] == -10_000  # C3's 50 units
    assert bridge["= Net sales in customer analysis"] == 2_800  # C1: O1 + O2


def test_compare_detects_merged_duplicate_and_bulk_buyer(result):
    table = compare(make_orders(), make_items(), make_products(), result).set_index("metric")
    # naive sees C1 and C9 (same person) plus C3 (bulk); clean sees only C1
    assert table.loc["Clientes con compras", "naive"] == 3
    assert table.loc["Clientes con compras", "clean"] == 1
    # account types instead of customer IDs, so the table stays aggregate-only
    assert table.loc["Mejor cliente", "naive"] == "comprador por volumen"
    assert table.loc["Mejor cliente", "clean"] == "cliente minorista"


def test_compare_isolates_each_oi26_rule(result):
    table = compare(make_orders(), make_items(), make_products(), result).set_index("metric")
    # S1 is labelled 'OI-26', so the naive view misses both raw IDs that bought it
    missed = "Compradores de OI26 que no aparecen por la etiqueta mal escrita"
    assert table.loc[missed, "naive"] == 2
    # C1 bought OI26 before (O2) and after (O1) launch, so it is not a pre-launch-only buyer
    pre_launch = "Compradores de OI26 que solo compraron antes del lanzamiento"
    assert table.loc[pre_launch, "naive"] == 0
