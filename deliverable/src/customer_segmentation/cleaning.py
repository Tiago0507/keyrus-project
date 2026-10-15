"""Data-quality rules and their impact on the conclusions.

Every rule records how many rows it touches in an issue log. Rows are fixed, flagged or excluded
with a stated reason, never dropped silently. `compare` and `sales_bridge` show what the
conclusions would have been without cleaning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from customer_segmentation import config


@dataclass
class IssueLog:
    """Collects one record per cleaning rule: what was found and what was done."""

    records: list[dict] = field(default_factory=list)

    def add(self, code: str, table: str, issue: str, rows: int, treatment: str) -> None:
        self.records.append(
            {
                "code": code,
                "table": table,
                "issue": issue,
                "rows_affected": int(rows),
                "treatment": treatment,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)


# Products


def normalize_collection(labels: pd.Series) -> pd.Series:
    """Uppercase and keep only letters and digits, so 'OI-26', 'oi26' and 'OI 26 ' become 'OI26'."""
    return labels.str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)


def clean_products(raw: pd.DataFrame, log: IssueLog) -> pd.DataFrame:
    p = raw.copy()

    normalized = normalize_collection(p["coleccion"])
    log.add(
        "P1",
        "productos",
        "Etiqueta de colección escrita de formas no estándar (OI-26, oi26, 'OI 26')",
        (normalized != p["coleccion"]).sum(),
        "Normalizada a la etiqueta canónica (mayúsculas, sin espacios ni guiones)",
    )
    p["coleccion"] = normalized

    p["fecha_lanzamiento"] = pd.to_datetime(p["fecha_lanzamiento"], format="%Y-%m-%d")
    p["precio_lista"] = pd.to_numeric(p["precio_lista"])
    p["is_seasonal"] = p["coleccion"] != config.BASICS_COLLECTION
    log.add(
        "P2",
        "productos",
        "Los básicos no tienen fecha de lanzamiento",
        p["fecha_lanzamiento"].isna().sum(),
        "No es un error: son atemporales. Se marcan como no de temporada",
    )
    return p


# Customers


def resolve_customers(pseudo: pd.DataFrame, log: IssueLog) -> tuple[pd.DataFrame, pd.Series]:
    """Merge records that belong to the same person (same email token).

    Returns the deduplicated customer table and a map from each source customer_id to its master ID.
    Master = earliest registration, ties broken by the lowest ID (the original record).
    """
    c = pseudo.copy()
    c["fecha_registro"] = pd.to_datetime(c["fecha_registro"], format="%Y-%m-%d")
    for col in ["opt_in_email", "opt_in_whatsapp"]:
        c[col] = c[col].map({"true": True, "false": False})

    c = c.sort_values(["fecha_registro", "customer_id"])
    # Email is the login identifier in the brand's identity system, so it is the match key.
    # A customer without a valid email can only match itself.
    identity_key = c["email_hash"].fillna(c["customer_id"])
    c["master_id"] = c.groupby(identity_key)["customer_id"].transform("first")
    id_map = c.set_index("customer_id")["master_id"]

    is_secondary = c["customer_id"] != c["master_id"]
    log.add(
        "C1",
        "clientes",
        "Misma persona registrada con dos IDs de cliente (mismo token de email)",
        is_secondary.sum(),
        "Fusionada en el ID original; sus órdenes se reasignan a ese ID",
    )

    groups = c.groupby("master_id")
    whatsapp_conflicts = (groups["opt_in_whatsapp"].nunique() > 1).sum()
    log.add(
        "C1b",
        "clientes",
        "Los registros duplicados no coinciden en el consentimiento de WhatsApp",
        whatsapp_conflicts,
        "Gana el más restrictivo: hay consentimiento solo si todos los registros lo dieron",
    )

    merged = groups.agg(
        fecha_registro=("fecha_registro", "min"),
        canal_registro=(
            "canal_registro",
            lambda s: s.iloc[0] if s.nunique() == 1 else "both",
        ),
        ciudad=("ciudad", "first"),
        opt_in_email=("opt_in_email", "all"),
        opt_in_whatsapp=("opt_in_whatsapp", "all"),
        email_hash=("email_hash", "first"),
        phone_hash=("phone_hash", "first"),
        email_valid=("email_valid", "any"),
        phone_valid=("phone_valid", "any"),
        is_internal_email=("is_internal_email", "any"),
        age_band=("age_band", "first"),
        has_crm_note=("has_crm_note", "any"),
        n_source_records=("customer_id", "size"),
    )
    merged = merged.rename_axis("customer_id").reset_index()

    merged["exclusion_reason"] = pd.Series(pd.NA, index=merged.index, dtype="string")
    internal = merged["is_internal_email"]
    merged.loc[internal, "exclusion_reason"] = "internal_account"
    log.add(
        "C4",
        "clientes",
        "Cuentas del dominio de email de la marca (personal, tiendas o pruebas)",
        internal.sum(),
        "Fuera del análisis por cliente; sus ventas siguen en el total de la marca",
    )
    return merged, id_map


# Orders


def clean_orders(
    raw: pd.DataFrame, customers: pd.DataFrame, id_map: pd.Series, log: IssueLog
) -> pd.DataFrame:
    o = raw.copy()
    o["fecha"] = pd.to_datetime(o["fecha"], format="%Y-%m-%d")
    o["source_customer_id"] = o["customer_id"]

    missing = o["customer_id"].isna()
    unknown = o["customer_id"].notna() & ~o["customer_id"].isin(id_map.index)
    log.add(
        "O1",
        "órdenes",
        "Orden sin customer_id (compra de invitado en tienda)",
        missing.sum(),
        "Se mantiene en la venta de la marca; fuera del análisis por cliente",
    )
    log.add(
        "O2",
        "órdenes",
        "customer_id que no existe en la tabla de clientes",
        unknown.sum(),
        "Se mantiene en la venta de la marca; fuera del análisis por cliente",
    )

    o["customer_id"] = o["customer_id"].map(id_map)
    reassigned = o["customer_id"].notna() & (o["customer_id"] != o["source_customer_id"])
    log.add(
        "C1c",
        "órdenes",
        "Órdenes hechas con un ID de cliente duplicado",
        reassigned.sum(),
        "Reasignadas al ID original del cliente",
    )

    registered_on = o["customer_id"].map(customers.set_index("customer_id")["fecha_registro"])
    log.add(
        "O3",
        "órdenes",
        "Orden anterior al registro del cliente",
        (o["fecha"] < registered_on).sum(),
        "No es un error: compra en tienda y registro posterior. La historia del cliente se mide "
        "desde su primera compra, no desde el registro",
    )

    o["customer_status"] = np.select([missing, unknown], ["guest", "unknown_id"], "registered")
    o["is_net_sale"] = o["estado"] == config.NET_SALE_STATUS
    log.add(
        "O4",
        "órdenes",
        "Órdenes devueltas o canceladas",
        (~o["is_net_sale"]).sum(),
        "Fuera de la venta neta",
    )
    return o


# Items


def clean_items(
    raw: pd.DataFrame, products: pd.DataFrame, orders: pd.DataFrame, log: IssueLog
) -> pd.DataFrame:
    i = raw.copy()
    for col in ["cantidad", "precio_unitario", "descuento_pct"]:
        i[col] = pd.to_numeric(i[col])

    exact_dups = i.duplicated()
    log.add(
        "I1",
        "líneas de orden",
        "Línea duplicada (idéntica en todas sus columnas)",
        exact_dups.sum(),
        "Eliminada: se trata como doble carga de la misma línea",
    )
    i = i[~exact_dups].copy()

    same_sku = i.duplicated(["order_id", "sku"], keep=False)
    log.add(
        "I2",
        "líneas de orden",
        "Mismo SKU dos veces en una orden con distinto descuento",
        same_sku.sum(),
        "Se conserva: dos unidades reales, una con promoción y otra sin ella",
    )

    orphan_sku = ~i["sku"].isin(products["sku"])
    log.add(
        "I4",
        "líneas de orden",
        "SKU que no está en el catálogo de productos",
        orphan_sku.sum(),
        f"Se mantiene en la venta con colección {config.UNKNOWN_COLLECTION}",
    )

    i = i.merge(
        products[["sku", "categoria", "coleccion", "fecha_lanzamiento", "is_seasonal"]],
        on="sku",
        how="left",
    )
    i["coleccion"] = i["coleccion"].fillna(config.UNKNOWN_COLLECTION)
    i["is_seasonal"] = i["is_seasonal"].fillna(False).astype(bool)

    i["gross_amount"] = i["cantidad"] * i["precio_unitario"]
    i["net_amount"] = i["gross_amount"] * (1 - i["descuento_pct"] / 100)

    order_dates = i["order_id"].map(orders.set_index("order_id")["fecha"])
    i["pre_launch"] = order_dates < i["fecha_lanzamiento"]  # NaT (basics, unknown) gives False
    log.add(
        "P3",
        "líneas de orden",
        "Línea vendida antes de la fecha de lanzamiento de su producto",
        i["pre_launch"].sum(),
        "Cuenta como venta, pero nunca como compra de colección nueva",
    )
    return i


def flag_bulk_buyers(
    customers: pd.DataFrame, orders: pd.DataFrame, items: pd.DataFrame, log: IssueLog
) -> pd.DataFrame:
    """Exclude customers with lines far above retail quantities (rule-based, not by ID)."""
    bulk_orders = items.loc[items["cantidad"] > config.MAX_UNITS_PER_LINE, "order_id"]
    bulk_ids = orders.loc[orders["order_id"].isin(bulk_orders), "customer_id"].dropna().unique()

    c = customers.copy()
    is_bulk = c["customer_id"].isin(bulk_ids) & c["exclusion_reason"].isna()
    c.loc[is_bulk, "exclusion_reason"] = "bulk_buyer"
    log.add(
        "I3",
        "líneas de orden",
        f"Líneas con más de {config.MAX_UNITS_PER_LINE} unidades (compra por volumen)",
        len(bulk_orders),
        f"Su comprador ({is_bulk.sum()} cliente) queda fuera del análisis por cliente",
    )
    return c


def run(
    customers_pseudo: pd.DataFrame,
    orders_raw: pd.DataFrame,
    items_raw: pd.DataFrame,
    products_raw: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    log = IssueLog()
    products = clean_products(products_raw, log)
    customers, id_map = resolve_customers(customers_pseudo, log)
    orders = clean_orders(orders_raw, customers, id_map, log)
    items = clean_items(items_raw, products, orders, log)
    customers = flag_bulk_buyers(customers, orders, items, log)

    excluded = customers.loc[customers["exclusion_reason"].notna(), "customer_id"]
    orders["in_customer_scope"] = (orders["customer_status"] == "registered") & ~orders[
        "customer_id"
    ].isin(excluded)

    return {
        "customers": customers,
        "orders": orders,
        "order_items": items,
        "products": products,
        "issue_log": log.to_frame(),
    }


def read_raw(name: str) -> pd.DataFrame:
    return pd.read_csv(
        config.RAW_DIR / f"{name}.csv", dtype=str, keep_default_na=False, na_values=[""]
    )


# Impact of cleaning

NEWEST_COLLECTION = "OI26"
ACCOUNT_LABELS = {"internal_account": "cuenta interna", "bulk_buyer": "comprador por volumen"}


def naive_lines(
    orders_raw: pd.DataFrame, items_raw: pd.DataFrame, products_raw: pd.DataFrame
) -> pd.DataFrame:
    """One row per raw item line, joined without any fix. Discount is applied."""
    i = items_raw.copy()
    for col in ["cantidad", "precio_unitario", "descuento_pct"]:
        i[col] = pd.to_numeric(i[col])
    i["amount"] = i["cantidad"] * i["precio_unitario"] * (1 - i["descuento_pct"] / 100)
    i = i.merge(orders_raw[["order_id", "customer_id"]], on="order_id", how="left")
    return i.merge(products_raw[["sku", "coleccion"]], on="sku", how="left")


def clean_lines(result: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Net-sale lines of in-scope customers, from the cleaning output."""
    orders = result["orders"]
    in_scope = orders.loc[orders["in_customer_scope"] & orders["is_net_sale"]]
    i = result["order_items"].merge(in_scope[["order_id", "customer_id"]], on="order_id")
    return i.rename(columns={"net_amount": "amount"})


def customer_metrics(lines: pd.DataFrame) -> dict:
    """Customer-level metrics that the three business questions depend on."""
    per_customer = (
        lines.dropna(subset=["customer_id"])
        .groupby("customer_id")
        .agg(sales=("amount", "sum"), orders=("order_id", "nunique"))
        .sort_values("sales", ascending=False)
    )
    top_decile = per_customer.head(math.ceil(len(per_customer) * 0.10))
    return {
        "customers": len(per_customer),
        "top_customer": per_customer.index[0],
        "top_customer_sales": per_customer["sales"].iloc[0],
        "top_customer_vs_second": (
            per_customer["sales"].iloc[0] / per_customer["sales"].iloc[1]
            if len(per_customer) > 1
            else float("nan")
        ),
        "top10pct_sales_share": top_decile["sales"].sum() / per_customer["sales"].sum(),
        "repeat_customers": int((per_customer["orders"] >= 2).sum()),
        "max_orders": int(per_customer["orders"].max()),
        "max_orders_customer": per_customer["orders"].idxmax(),
        "top10_ids": per_customer.index[:10].tolist(),
    }


def collection_metrics(lines: pd.DataFrame, label_matches: pd.Series) -> dict:
    newest = lines.loc[label_matches]
    return {
        "newest_buyers": newest["customer_id"].nunique(),
        "newest_sales": newest["amount"].sum(),
    }


def compare(
    orders_raw: pd.DataFrame,
    items_raw: pd.DataFrame,
    products_raw: pd.DataFrame,
    result: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Side-by-side table: metric | naive | clean | issue codes that explain the gap."""
    naive = naive_lines(orders_raw, items_raw, products_raw)
    clean = clean_lines(result)
    n, c = customer_metrics(naive), customer_metrics(clean)

    n_new = collection_metrics(naive, naive["coleccion"] == NEWEST_COLLECTION)
    c_new = collection_metrics(
        clean, (clean["coleccion"] == NEWEST_COLLECTION) & ~clean["pre_launch"]
    )

    customers = result["customers"]
    excluded = set(customers.loc[customers["exclusion_reason"].notna(), "customer_id"])
    naive_top10_not_retail = sum(cid in excluded for cid in n["top10_ids"])

    # The table shows what kind of account it is, never a customer ID: outputs stay aggregate.
    orders = result["orders"].drop_duplicates("source_customer_id")
    master = orders.set_index("source_customer_id")["customer_id"]
    reason = customers.set_index("customer_id")["exclusion_reason"]

    def describe(customer_id: str) -> str:
        if pd.isna(master.get(customer_id)):
            return "ID de cliente inexistente"
        return ACCOUNT_LABELS.get(reason[master[customer_id]], "cliente minorista")

    # Effect of each OI26 rule on its own: buyers the naive view misses (P1) or wrongly counts (P3).
    label = naive["coleccion"].fillna("")
    exact = label == NEWEST_COLLECTION
    misspelled = (normalize_collection(label) == NEWEST_COLLECTION) & ~exact
    missed = set(naive.loc[misspelled, "customer_id"].dropna()) - set(
        naive.loc[exact, "customer_id"].dropna()
    )
    newest = clean[clean["coleccion"] == NEWEST_COLLECTION]
    pre_launch_only = set(newest["customer_id"]) - set(
        newest.loc[~newest["pre_launch"], "customer_id"]
    )

    new = NEWEST_COLLECTION
    rows = [
        ("Clientes con compras", n["customers"], c["customers"], "C1, C4, I3, O2"),
        ("Mejor cliente", describe(n["top_customer"]), describe(c["top_customer"]), "I3"),
        ("Venta del mejor cliente (MXN)", n["top_customer_sales"], c["top_customer_sales"], "I3"),
        ("Mejor cliente frente al segundo (veces)", n["top_customer_vs_second"],
         c["top_customer_vs_second"], "I3"),
        ("Cuentas no minoristas en el top 10", naive_top10_not_retail, 0, "C4, I3"),
        ("% de la venta del 10 % de clientes que más compra", n["top10pct_sales_share"] * 100,
         c["top10pct_sales_share"] * 100, "I3, C4, O4"),
        ("Clientes con 2 o más órdenes", n["repeat_customers"], c["repeat_customers"], "C1, O4"),
        ("Más órdenes de un solo cliente", n["max_orders"], c["max_orders"], "C4"),
        ("Cliente con más órdenes", describe(n["max_orders_customer"]),
         describe(c["max_orders_customer"]), "C4"),
        (f"SKU de {new} en el catálogo", int((products_raw["coleccion"] == new).sum()),
         int((result["products"]["coleccion"] == new).sum()), "P1"),
        (f"Compradores de {new}", n_new["newest_buyers"], c_new["newest_buyers"], "P1, P3, O4"),
        (f"Compradores de {new} que no aparecen por la etiqueta mal escrita", len(missed), 0,
         "P1"),
        (f"Compradores de {new} que solo compraron antes del lanzamiento", len(pre_launch_only),
         0, "P3"),
        (f"Venta de {new} (MXN)", n_new["newest_sales"], c_new["newest_sales"], "P1, P3, O4"),
    ]  # fmt: skip
    return pd.DataFrame(rows, columns=["metric", "naive", "clean", "explained_by"])


def sales_bridge(items_raw: pd.DataFrame, result: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Waterfall from raw gross sales to the net sales used in the customer analysis.

    Every step is a difference between two sums, so the bridge always reconciles.
    """
    raw = items_raw.assign(
        gross=pd.to_numeric(items_raw["cantidad"]) * pd.to_numeric(items_raw["precio_unitario"])
    )
    items = result["order_items"]
    customers = result["customers"].set_index("customer_id")["exclusion_reason"]
    lines = items.merge(
        result["orders"][["order_id", "is_net_sale", "customer_status", "customer_id"]],
        on="order_id",
    )
    completed = lines[lines["is_net_sale"]]
    reason = completed["customer_id"].map(customers)

    def net(mask: pd.Series) -> float:
        return completed.loc[mask, "net_amount"].sum()

    steps = [
        ("Raw gross sales (as loaded)", raw["gross"].sum()),
        ("Duplicated lines", items["gross_amount"].sum() - raw["gross"].sum()),
        ("Returned and cancelled orders", -lines.loc[~lines["is_net_sale"], "gross_amount"].sum()),
        ("Discounts", completed["net_amount"].sum() - completed["gross_amount"].sum()),
        ("= Brand net sales", completed["net_amount"].sum()),
        ("Guest purchases (no customer)", -net(completed["customer_status"] == "guest")),
        ("Unknown customer IDs", -net(completed["customer_status"] == "unknown_id")),
        ("Internal accounts", -net(reason == "internal_account")),
        ("Bulk buyer", -net(reason == "bulk_buyer")),
        ("= Net sales in customer analysis",
         net((completed["customer_status"] == "registered") & reason.isna())),
    ]  # fmt: skip
    bridge = pd.DataFrame(steps, columns=["step", "amount_mxn"])
    brand_net = completed["net_amount"].sum()
    bridge["share_of_brand_net"] = bridge["amount_mxn"] / brand_net
    return bridge
