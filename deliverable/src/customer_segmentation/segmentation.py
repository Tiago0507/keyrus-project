"""Customer features, the three business questions and the behavioural archetypes.

- `build_features`: one row per in-scope customer, built only from net sales.
- `segment_flags`: one transparent rule per business question, thresholds in config.
- `assign_archetypes`: K-Means on seven behaviour features (k = 5), named by profile rules,
  plus a depth-3 decision tree that translates the clusters into readable segment rules.
"""

from __future__ import annotations

import re
from datetime import date

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from customer_segmentation import config

# Customer features

FEATURE_DESCRIPTIONS = {
    # identity and profile
    "customer_id": "ID maestro del cliente (duplicados ya fusionados)",
    "ciudad": "Ciudad declarada en el registro",
    "canal_registro": "Canal de registro: online, tienda o both (duplicados fusionados)",
    "age_band": "Rango de edad a la fecha de corte (vacío si la fecha falta o no es válida)",
    "fecha_registro": "Fecha de registro más antigua",
    # contactability (consent AND a valid address)
    "contactable_email": "Aceptó email y tiene un email válido",
    "contactable_whatsapp": "Aceptó WhatsApp y tiene un teléfono válido",
    "contactable_any": "Se le puede contactar por al menos un canal",
    # RFM
    "has_purchase": "Tiene al menos una orden completada",
    "n_orders": "Órdenes completadas (frecuencia)",
    "net_sales": "Venta neta en MXN, con descuento, solo órdenes completadas (valor)",
    "avg_order_value": "Venta neta dividida entre órdenes completadas (ticket promedio)",
    "first_order_date": "Fecha de la primera orden completada",
    "last_order_date": "Fecha de la última orden completada",
    "recency_days": "Días desde la última orden completada hasta la fecha de corte (recencia)",
    "net_sales_12m": "Venta neta de los últimos 12 meses",
    # recurrence
    "n_purchase_days": "Días distintos con una orden completada (el mismo día cuenta una vez)",
    "median_days_between_purchases": "Mediana de días entre días de compra (requiere 2 o más días)",
    # interest in new collections
    "seasonal_sales_share": "Proporción de la venta neta en colecciones de temporada (no BASICOS)",
    "launches_bought": "Colecciones distintas compradas dentro de su ventana de lanzamiento",
    "launches_eligible": "Lanzamientos cuya ventana terminó después de su primera orden",
    "launch_participation_rate": "launches_bought dividido entre launches_eligible",
    # channel and price sensitivity
    "online_order_share": "Proporción de órdenes completadas hechas online",
    "avg_discount_pct": "Descuento promedio recibido, ponderado por monto bruto (%)",
}

COUNT_COLUMNS = ["n_orders", "net_sales", "net_sales_12m", "n_purchase_days", "launches_bought",
                 "seasonal_sales"]  # fmt: skip


def _rfm(orders: pd.DataFrame, order_net: pd.Series, ref: pd.Timestamp) -> pd.DataFrame:
    o = orders.assign(net=orders["order_id"].map(order_net).fillna(0.0))
    g = o.groupby("customer_id")
    out = pd.DataFrame(
        {
            "n_orders": g["order_id"].nunique(),
            "net_sales": g["net"].sum(),
            "first_order_date": g["fecha"].min(),
            "last_order_date": g["fecha"].max(),
        }
    )
    out["avg_order_value"] = out["net_sales"] / out["n_orders"]
    out["recency_days"] = (ref - out["last_order_date"]).dt.days

    recent = o[o["fecha"] > ref - pd.DateOffset(months=config.RECENT_MONTHS)]
    out["net_sales_12m"] = recent.groupby("customer_id")["net"].sum()
    return out


def _recurrence(orders: pd.DataFrame) -> pd.DataFrame:
    days = orders[["customer_id", "fecha"]].drop_duplicates().sort_values(["customer_id", "fecha"])
    days["gap"] = days.groupby("customer_id")["fecha"].diff().dt.days
    return days.groupby("customer_id").agg(
        n_purchase_days=("fecha", "size"),
        median_days_between_purchases=("gap", "median"),
    )


def _new_collections(
    lines: pd.DataFrame, products: pd.DataFrame, first_order: pd.Series
) -> pd.DataFrame:
    launches = products.loc[products["is_seasonal"]].groupby("coleccion")["fecha_lanzamiento"].min()
    window = pd.Timedelta(days=config.LAUNCH_WINDOW_DAYS)

    seasonal = lines[lines["is_seasonal"]]
    days_since_launch = seasonal["fecha"] - seasonal["fecha_lanzamiento"]
    in_window = seasonal[
        ~seasonal["pre_launch"] & (days_since_launch <= window)
    ]  # pre_launch already excludes negative values

    out = pd.DataFrame(
        {
            "seasonal_sales": seasonal.groupby("customer_id")["net_amount"].sum(),
            "launches_bought": in_window.groupby("customer_id")["coleccion"].nunique(),
        }
    )
    # A launch counts as an opportunity only if the customer already existed before it closed.
    out = out.reindex(first_order.index)
    out["launches_eligible"] = sum(
        (first_order <= launch + window).astype(int) for launch in launches
    )
    return out


def build_features(
    result: dict[str, pd.DataFrame], reference: date = config.REFERENCE_DATE
) -> pd.DataFrame:
    ref = pd.Timestamp(reference)
    customers = result["customers"]
    orders = result["orders"]

    in_scope = customers.loc[customers["exclusion_reason"].isna()].set_index("customer_id")
    net_orders = orders.loc[orders["in_customer_scope"] & orders["is_net_sale"]]
    lines = result["order_items"].merge(
        net_orders[["order_id", "customer_id", "fecha"]], on="order_id"
    )
    order_net = lines.groupby("order_id")["net_amount"].sum()

    f = in_scope[["ciudad", "canal_registro", "age_band", "fecha_registro"]].copy()
    f["contactable_email"] = in_scope["opt_in_email"] & in_scope["email_valid"]
    f["contactable_whatsapp"] = in_scope["opt_in_whatsapp"] & in_scope["phone_valid"]
    f["contactable_any"] = f["contactable_email"] | f["contactable_whatsapp"]

    rfm = _rfm(net_orders, order_net, ref)
    f = f.join(rfm).join(_recurrence(net_orders))
    f = f.join(_new_collections(lines, result["products"], rfm["first_order_date"]))
    f["online_order_share"] = (
        (net_orders["canal"] == "online").groupby(net_orders["customer_id"]).mean()
    )
    g = lines.groupby("customer_id")
    f["avg_discount_pct"] = (1 - g["net_amount"].sum() / g["gross_amount"].sum()) * 100

    f[COUNT_COLUMNS] = f[COUNT_COLUMNS].fillna(0)
    f["n_orders"] = f["n_orders"].astype(int)
    f["has_purchase"] = f["n_orders"] > 0
    f["seasonal_sales_share"] = f.pop("seasonal_sales") / f["net_sales"].where(f["has_purchase"])
    f["launch_participation_rate"] = f["launches_bought"] / f["launches_eligible"].where(
        f["launches_eligible"] > 0
    )

    # Reconciliation: customer-level sales must add up to the line-level net sales.
    if abs(f["net_sales"].sum() - lines["net_amount"].sum()) > 0.01:
        raise RuntimeError("Customer net sales do not reconcile with line-level net sales")

    f = f.rename_axis("customer_id").reset_index()
    return f[list(FEATURE_DESCRIPTIONS)]


def launch_curve(result: dict[str, pd.DataFrame], bucket_days: int = 30) -> pd.DataFrame:
    """Share of seasonal net sales by days since launch: the evidence for the 30-day window.

    Uses net sales of in-scope customers, post-launch only, in buckets of `bucket_days`.
    """
    orders = result["orders"]
    net_orders = orders.loc[orders["in_customer_scope"] & orders["is_net_sale"]]
    lines = result["order_items"].merge(net_orders[["order_id", "fecha"]], on="order_id")
    seasonal = lines.loc[lines["is_seasonal"] & ~lines["pre_launch"]]
    days = (seasonal["fecha"] - seasonal["fecha_lanzamiento"]).dt.days
    bucket = (days // bucket_days).clip(upper=12) * bucket_days  # last bucket = 360+ days
    all_buckets = range(0, 13 * bucket_days, bucket_days)
    curve = seasonal.groupby(bucket)["net_amount"].sum().reindex(all_buckets, fill_value=0)
    return pd.DataFrame(
        {"days_from": curve.index, "share_of_sales": (curve / curve.sum()).to_numpy()}
    )


def feature_dictionary() -> pd.DataFrame:
    return pd.DataFrame(
        {"feature": list(FEATURE_DESCRIPTIONS), "description": list(FEATURE_DESCRIPTIONS.values())}
    )


# Business questions

QUESTIONS = {
    "new_collection_fan": "P1. ¿Quiénes se interesan por las nuevas colecciones?",
    "recurrent": "P2. ¿Quiénes compran de forma recurrente?",
    "high_value": "P3. ¿Quiénes son los clientes de mayor valor?",
}


def _mxn(amount: int) -> str:
    return f"{amount:,}".replace(",", ".")


CRITERIA = {
    "new_collection_fan": (
        f"Compró en los primeros {config.LAUNCH_WINDOW_DAYS} días del lanzamiento en al menos el "
        f"{config.NEW_COLLECTION_MIN_PARTICIPATION * 100:.0f} % de los lanzamientos que tuvo "
        "disponibles"
    ),
    "recurrent": (
        f"{config.RECURRENT_MIN_PURCHASE_DAYS} o más días de compra, mediana de "
        f"{config.RECURRENT_MAX_MEDIAN_GAP_DAYS} días o menos entre compras y última compra hace "
        f"{config.RECURRENT_MAX_RECENCY_DAYS} días o menos"
    ),
    "high_value": (
        f"{_mxn(config.HIGH_VALUE_MIN_SALES_12M)} MXN o más de venta neta en los últimos 12 meses"
    ),
}


def is_new_collection_fan(f: pd.DataFrame, min_participation: float) -> pd.Series:
    return f["launch_participation_rate"].fillna(0) >= min_participation


def is_recurrent(
    f: pd.DataFrame, min_days: int, max_median_gap: float, max_recency: float
) -> pd.Series:
    return (
        (f["n_purchase_days"] >= min_days)
        & (f["median_days_between_purchases"] <= max_median_gap)
        & (f["recency_days"] <= max_recency)
    )


def is_high_value(f: pd.DataFrame, min_sales_12m: float) -> pd.Series:
    return f["net_sales_12m"] >= min_sales_12m


def segment_flags(f: pd.DataFrame) -> pd.DataFrame:
    """One boolean column per question, using the thresholds in config."""
    return pd.DataFrame(
        {
            "customer_id": f["customer_id"],
            "new_collection_fan": is_new_collection_fan(f, config.NEW_COLLECTION_MIN_PARTICIPATION),
            "recurrent": is_recurrent(
                f,
                config.RECURRENT_MIN_PURCHASE_DAYS,
                config.RECURRENT_MAX_MEDIAN_GAP_DAYS,
                config.RECURRENT_MAX_RECENCY_DAYS,
            ),
            "high_value": is_high_value(f, config.HIGH_VALUE_MIN_SALES_12M),
        }
    )


def brand_net_sales(result: dict[str, pd.DataFrame], since: pd.Timestamp | None = None) -> float:
    """Net sales of the whole brand (every completed order, in scope or not)."""
    orders = result["orders"]
    completed = orders.loc[orders["is_net_sale"]]
    if since is not None:
        completed = completed.loc[completed["fecha"] > since]
    items = result["order_items"]
    return items.loc[items["order_id"].isin(completed["order_id"]), "net_amount"].sum()


def answer_table(
    f: pd.DataFrame, flags: pd.DataFrame, brand_total: float, brand_total_12m: float
) -> pd.DataFrame:
    buyers = f["has_purchase"].sum()
    rows = []
    for key, label in QUESTIONS.items():
        group = f.loc[flags[key].to_numpy()]
        rows.append(
            {
                "question": label,
                "criterion": CRITERIA[key],
                "customers": len(group),
                "share_of_buyers": len(group) / buyers,
                "share_of_brand_sales_3y": group["net_sales"].sum() / brand_total,
                "share_of_brand_sales_12m": group["net_sales_12m"].sum() / brand_total_12m,
                "contactable_share": group["contactable_any"].mean(),
                "basics_only_share": (group["seasonal_sales_share"] == 0).mean(),
            }
        )
    return pd.DataFrame(rows)


def overlap_table(f: pd.DataFrame, flags: pd.DataFrame, brand_total: float) -> pd.DataFrame:
    """Customers and sales for every combination of the three flags (buyers only)."""
    keys = list(QUESTIONS)
    data = flags.assign(net_sales=f["net_sales"].to_numpy())[f["has_purchase"].to_numpy()]
    table = (
        data.groupby(keys)
        .agg(customers=("customer_id", "size"), net_sales=("net_sales", "sum"))
        .reset_index()
    )
    table["share_of_brand_sales_3y"] = table["net_sales"] / brand_total
    table["n_groups"] = table[keys].sum(axis=1)
    return table.sort_values(["n_groups", "customers"], ascending=[False, False])


def sensitivity_table(f: pd.DataFrame, brand_total_12m: float) -> pd.DataFrame:
    """How many customers each answer captures if its main threshold moves."""
    scenarios = []
    for t in [0.34, 0.5, 0.67, 1.0]:
        scenarios.append(
            (
                "new_collection_fan",
                f"participación de al menos el {t * 100:.0f} %",
                is_new_collection_fan(f, t),
            )
        )
    # Q2 moves one condition at a time around the chosen values (60-day gap, 90-day recency).
    gap, recency = config.RECURRENT_MAX_MEDIAN_GAP_DAYS, config.RECURRENT_MAX_RECENCY_DAYS
    q2 = [(g, recency) for g in [45, gap, 90]] + [(gap, r) for r in [60, 180]]
    for gap, recency in q2:
        scenarios.append(
            (
                "recurrent",
                f"brecha hasta {gap} días, última compra hace {recency} días o menos",
                is_recurrent(f, config.RECURRENT_MIN_PURCHASE_DAYS, gap, recency),
            )
        )
    for amount in [15_000, 20_000, 25_000, 30_000, 40_000]:
        scenarios.append(
            ("high_value", f"venta en 12 meses de {_mxn(amount)} MXN o más",
             is_high_value(f, amount))
        )  # fmt: skip
    # Conventional alternative for Q3: the top 10% by 12-month sales.
    top_decile = f["net_sales_12m"].rank(ascending=False, method="first") <= round(
        f["has_purchase"].sum() * 0.10
    )
    scenarios.append(("high_value", "el 10 % que más compra en 12 meses (convención)", top_decile))

    rows = [
        {
            "question": QUESTIONS[key],
            "scenario": name,
            "customers": int(mask.sum()),
            "share_of_brand_sales_12m": f.loc[mask, "net_sales_12m"].sum() / brand_total_12m,
        }
        for key, name, mask in scenarios
    ]  # fmt: skip
    return pd.DataFrame(rows)


# Archetypes


N_ARCHETYPES = 5  # chosen in the notebook (section 4)
RANDOM_STATE = 0
NO_PURCHASE = "registered_no_purchase"

ARCHETYPE_NAMES = {
    "fashion_loyalists": "Fieles a la moda",
    "basics_replenishers": "Reponedores de básicos",
    "sale_hunters": "Cazadores de rebajas",
    "store_one_timers": "Compradores únicos de tienda",
    "lapsed_online": "Ocasionales online dormidos",
    NO_PURCHASE: "Registrados sin compra",
}

# Interpretable features used by the surrogate decision tree (no transformations), with the
# label shown in the printed rules.
RULE_FEATURES = {
    "recency_days": "días desde la última compra",
    "n_purchase_days": "días de compra",
    "avg_order_value": "ticket promedio (MXN)",
    "launch_participation_rate": "participación en lanzamientos",
    "seasonal_sales_share": "proporción de venta de temporada",
    "avg_discount_pct": "descuento promedio (%)",
    "online_order_share": "proporción de órdenes online",
}


def clustering_matrix(buyers: pd.DataFrame) -> pd.DataFrame:
    """Behaviour features for clustering. Log reduces the long right tail of counts and amounts."""
    return pd.DataFrame(
        {
            "log_recency": np.log1p(buyers["recency_days"]),
            "log_purchase_days": np.log1p(buyers["n_purchase_days"]),
            "log_avg_order_value": np.log(buyers["avg_order_value"]),
            "launch_participation": buyers["launch_participation_rate"],
            "seasonal_share": buyers["seasonal_sales_share"],
            "avg_discount_pct": buyers["avg_discount_pct"],
            "online_share": buyers["online_order_share"],
        },
        index=buyers.index,
    )


def standardize(x: pd.DataFrame) -> np.ndarray:
    return StandardScaler().fit_transform(x)


def evaluate_k(
    z: np.ndarray, ks: range = range(2, 11), n_resamples: int = 10, sample_frac: float = 0.8
) -> pd.DataFrame:
    """Quality and stability of K-Means for each k."""
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for k in ks:
        labels = KMeans(k, n_init=20, random_state=RANDOM_STATE).fit_predict(z)
        agreement = []
        for seed in range(n_resamples):
            idx = rng.choice(len(z), int(sample_frac * len(z)), replace=False)
            sub = KMeans(k, n_init=10, random_state=seed + 1).fit_predict(z[idx])
            agreement.append(adjusted_rand_score(labels[idx], sub))
        rows.append(
            {
                "k": k,
                "silhouette": silhouette_score(z, labels),  # higher = better separated
                "stability_min_ari": float(np.min(agreement)),  # 1 = identical groups
                "smallest_cluster": int(np.bincount(labels).min()),
            }
        )
    return pd.DataFrame(rows)


def name_clusters(profile: pd.DataFrame) -> dict[int, str]:
    """Map K-Means labels to archetype keys using their mean behaviour.

    Fails loudly if the profile does not match the expected structure, instead of silently
    giving a wrong name to a group.
    """
    remaining = set(profile.index)

    def pick(key: str, candidates: pd.Series) -> None:
        if len(candidates) != 1:
            raise ValueError(f"Cannot identify '{key}': {len(candidates)} candidate clusters")
        names[candidates.index[0]] = key
        remaining.discard(candidates.index[0])

    names: dict[int, str] = {}
    pick("basics_replenishers", profile.loc[profile["seasonal_share"] < 0.5])
    pick("fashion_loyalists", profile.loc[profile["launch_participation"] > 0.5])
    pick("sale_hunters", profile.loc[profile["avg_discount_pct"] > 25])
    rest = profile.loc[sorted(remaining)].sort_values("online_share")
    if len(rest) != 2:
        raise ValueError(f"Expected 2 lapsed clusters, found {len(rest)}")
    names[rest.index[0]] = "store_one_timers"
    names[rest.index[1]] = "lapsed_online"
    return names


def assign_archetypes(features: pd.DataFrame, k: int = N_ARCHETYPES) -> pd.Series:
    """Archetype key per customer_id. Non-buyers get NO_PURCHASE without clustering."""
    f = features.set_index("customer_id")
    buyers = f.loc[f["has_purchase"]]
    x = clustering_matrix(buyers)
    labels = KMeans(k, n_init=20, random_state=RANDOM_STATE).fit_predict(standardize(x))
    names = name_clusters(x.groupby(labels).mean())

    archetype = pd.Series(NO_PURCHASE, index=f.index, name="archetype")
    archetype.loc[buyers.index] = [names[label] for label in labels]
    return archetype


def profile_archetypes(
    features: pd.DataFrame, flags: pd.DataFrame, archetype: pd.Series, brand_total: float
) -> pd.DataFrame:
    data = features.merge(flags, on="customer_id").assign(
        archetype=lambda d: d["customer_id"].map(archetype)
    )
    g = data.groupby("archetype")
    profile = g.agg(
        customers=("customer_id", "size"),
        net_sales_3y=("net_sales", "sum"),
        median_recency_days=("recency_days", "median"),
        median_orders=("n_orders", "median"),
        median_days_between=("median_days_between_purchases", "median"),
        median_avg_order_value=("avg_order_value", "median"),
        launch_participation=("launch_participation_rate", "mean"),
        seasonal_share=("seasonal_sales_share", "mean"),
        avg_discount_pct=("avg_discount_pct", "mean"),
        online_share=("online_order_share", "mean"),
        q1_new_collection_fan=("new_collection_fan", "mean"),
        q2_recurrent=("recurrent", "mean"),
        q3_high_value=("high_value", "mean"),
        contactable=("contactable_any", "mean"),
    )
    profile.insert(1, "share_of_customers", profile["customers"] / profile["customers"].sum())
    profile.insert(3, "share_of_brand_sales_3y", profile["net_sales_3y"] / brand_total)
    profile.insert(0, "name", profile.index.map(ARCHETYPE_NAMES))
    order = list(ARCHETYPE_NAMES)
    return profile.reindex([key for key in order if key in profile.index])


def purchase_months(result: dict[str, pd.DataFrame], archetype: pd.Series) -> pd.DataFrame:
    """Share of each archetype's completed orders by calendar month (rows: archetype)."""
    orders = result["orders"]
    net_orders = orders.loc[orders["in_customer_scope"] & orders["is_net_sale"]]
    data = net_orders.assign(
        archetype=net_orders["customer_id"].map(archetype), month=net_orders["fecha"].dt.month
    )
    counts = data.groupby(["archetype", "month"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=range(1, 13), fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0)


def surrogate_rules(features: pd.DataFrame, archetype: pd.Series) -> tuple[float, str]:
    """Depth-3 decision tree that reproduces the clusters with simple, readable conditions."""
    buyers = features.loc[features["has_purchase"]].set_index("customer_id")
    x = buyers[list(RULE_FEATURES)]
    y = archetype.loc[buyers.index]
    tree = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
    accuracy = cross_val_score(tree, x, y, cv=5).mean()
    tree.fit(x, y)
    rules = export_text(
        tree,
        feature_names=list(RULE_FEATURES.values()),
        class_names=[ARCHETYPE_NAMES[c] for c in tree.classes_],
        decimals=1,
    )
    # export_text writes "class:" and decimal points; the notebook shows Spanish.
    rules = re.sub(r"(\d)\.(\d)", r"\1,\2", rules.replace("class: ", "arquetipo: "))
    return float(accuracy), rules
