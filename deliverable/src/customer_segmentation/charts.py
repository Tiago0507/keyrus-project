"""Charts and tables for the notebook and the executive summary. Every chart function takes
aggregated data and returns a matplotlib Figure, so the same charts can be reused in slides.
`table` shows any result table with Spanish headers; the code keeps English column names.

Design rules (validated colour-blind safe palette):
- one series uses slot 1 (blue), a second series slot 2 (orange), context marks neutral gray;
- emphasis over decoration: the bar the story is about is colored, the rest is gray;
- thin marks, solid hairline grid, text always in ink colors (never in the series color);
- selective direct labels, one y-axis per chart (never dual axes).
"""

from __future__ import annotations

import numbers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from customer_segmentation import config

# Validated categorical slots (light mode) and chart chrome.
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
CONTEXT = "#c3c2b7"  # neutral gray for de-emphasized marks
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#1c5cab", "#f0efec", "#e34948"])


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.edgecolor": CONTEXT,
            "axes.linewidth": 1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 1,
            "grid.linestyle": "-",
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "legend.frameon": False,
            "figure.dpi": 110,
        }
    )


def number(value: float, decimals: int = 0) -> str:
    """Spanish number format, as in the narrative: 1.827 and 38,6."""
    return f"{value:,.{decimals}f}".translate(str.maketrans(",.", ".,"))


def _millions(value: float) -> str:
    return f"{number(value / 1e6, 1)} M"


def _percent_axis(ax: plt.Axes, axis: str = "y") -> None:
    fmt = plt.FuncFormatter(lambda v, _: f"{v:.0%}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _subtitle(ax: plt.Axes, text: str) -> None:
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK_SECONDARY, fontsize=9, va="bottom")


# Quality


def sales_bridge_chart(bridge: pd.DataFrame) -> Figure:
    """Waterfall from raw gross sales to the net sales used in the customer analysis."""
    labels = {
        "Raw gross sales (as loaded)": "Venta bruta\ncargada",
        "Duplicated lines": "Líneas\nduplicadas",
        "Returned and cancelled orders": "Devueltas y\ncanceladas",
        "Discounts": "Descuentos",
        "= Brand net sales": "Venta neta\nde la marca",
        "Guest purchases (no customer)": "Invitados",
        "Unknown customer IDs": "IDs\ninexistentes",
        "Internal accounts": "Cuentas\ninternas",
        "Bulk buyer": "Compra por\nvolumen",
        "= Net sales in customer analysis": "Venta del\nanálisis",
    }
    steps = bridge["step"].tolist()
    amounts = bridge["amount_mxn"].to_numpy()
    is_total = [i == 0 or s.startswith("=") for i, s in enumerate(steps)]

    bottoms, running = [], 0.0
    for amount, total in zip(amounts, is_total, strict=True):
        if total:
            bottoms.append(0.0)
            running = amount
        else:
            running += amount
            bottoms.append(running)
    heights = [a if t else -a for a, t in zip(amounts, is_total, strict=True)]

    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(steps))
    colors = [SERIES_1 if t else CONTEXT for t in is_total]
    ax.bar(x, heights, bottom=bottoms, width=0.6, color=colors)
    for xi, amount, bottom, height, total in zip(
        x, amounts, bottoms, heights, is_total, strict=True
    ):
        text = _millions(amount) if total else f"-{_millions(-amount)}"
        ax.text(
            xi,
            bottom + height,
            text,
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK if total else INK_SECONDARY,
            fontweight="bold" if total else None,
        )
    ax.set_xticks(x, [labels.get(s, s) for s in steps], fontsize=8.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1e6:.0f} M"))
    ax.set_ylabel("MXN")
    ax.grid(axis="x", visible=False)
    inflation = amounts[0] / amounts[steps.index("= Brand net sales")] - 1
    title = f"La venta cargada era un {number(inflation * 100)} % mayor que la venta neta real"
    ax.set_title(title, pad=22)
    _subtitle(ax, "Enero de 2024 a septiembre de 2026. Totales en azul, ajustes en gris")
    fig.tight_layout()
    return fig


# Features


def launch_curve_chart(curve: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(9, 3.8))
    labels = [str(d) if d < 360 else "360+" for d in curve["days_from"]]
    colors = [SERIES_1 if d == 0 else CONTEXT for d in curve["days_from"]]
    ax.bar(labels, curve["share_of_sales"], width=0.6, color=colors)
    first = curve["share_of_sales"].iloc[0]
    ax.text(0, first, f"{first:.0%}", ha="center", va="bottom", fontweight="bold")
    _percent_axis(ax)
    ax.set_xlabel("Días desde el lanzamiento de la colección (inicio de cada tramo de 30 días)")
    ax.grid(axis="x", visible=False)
    ax.set_title("El primer mes de una colección vende unas 3 veces lo de un mes normal", pad=22)
    _subtitle(
        ax,
        f"% de la venta de temporada según los días desde el lanzamiento. Ventana de "
        f"lanzamiento de {config.LAUNCH_WINDOW_DAYS} días",
    )
    fig.tight_layout()
    return fig


# Questions


def _threshold(ax: plt.Axes, x: float, label: str) -> None:
    ax.axvline(x, color=INK, linewidth=1.5)
    ax.text(x, 1.0, f" {label}", transform=ax.get_xaxis_transform(), ha="left", va="top",
            color=INK, fontsize=9, fontweight="bold")  # fmt: skip


def threshold_charts(features: pd.DataFrame) -> Figure:
    """Each criterion's threshold drawn on the distribution it cuts: it sits in a natural gap."""
    buyers = features.loc[features["has_purchase"]]
    recurrent_candidates = buyers["n_purchase_days"] >= config.RECURRENT_MIN_PURCHASE_DAYS
    high_value = config.HIGH_VALUE_MIN_SALES_12M / 1000
    # (values, bins, threshold, label, the group is below the threshold, title, x label)
    panels = [
        (buyers["launch_participation_rate"], np.linspace(0, 1, 11),
         config.NEW_COLLECTION_MIN_PARTICIPATION, "50 % o más", False,
         "P1. Colecciones nuevas", "Participación en lanzamientos"),
        (buyers.loc[recurrent_candidates, "median_days_between_purchases"].clip(upper=360),
         np.arange(0, 375, 15), config.RECURRENT_MAX_MEDIAN_GAP_DAYS, "60 días o menos", True,
         "P2. Recurrencia", "Mediana de días entre compras (3+ compras)"),
        ((buyers.loc[buyers["net_sales_12m"] > 0, "net_sales_12m"] / 1000).clip(upper=84),
         np.arange(0, 85, 5), high_value, "25 mil o más", False,
         "P3. Mayor valor", "Venta neta últimos 12 meses (miles de MXN)"),
    ]  # fmt: skip

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (values, bins, threshold, label, below, title, xlabel) in zip(
        axes, panels, strict=True
    ):
        counts, edges = np.histogram(values, bins=bins)
        in_group = edges[:-1] < threshold if below else edges[:-1] >= threshold
        width = 0.8 * (edges[1] - edges[0])
        ax.bar(edges[:-1], counts, width=width, align="edge",
               color=[SERIES_1 if k else CONTEXT for k in in_group])  # fmt: skip
        _threshold(ax, threshold, label)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    _percent_axis(axes[0], "x")
    axes[0].set_ylabel("Clientes")
    fig.suptitle(
        "Cada umbral cae en un hueco natural de los datos",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=13,
    )
    fig.tight_layout()
    return fig


OVERLAP_LABELS = {
    "new_collection_fan": "lanzamientos",
    "recurrent": "recurrente",
    "high_value": "alto valor",
}


def overlap_chart(overlaps: pd.DataFrame) -> Figure:
    """Share of brand sales for every combination of the three groups."""

    def name(row: pd.Series) -> str:
        parts = [label for key, label in OVERLAP_LABELS.items() if row[key]]
        if len(parts) <= 1:
            return f"Solo {parts[0]}" if parts else "Ninguno de los tres"
        return (", ".join(parts[:-1]) + " y " + parts[-1]).capitalize()

    data = overlaps.assign(label=overlaps.apply(name, axis=1)).sort_values(
        "share_of_brand_sales_3y"
    )
    fig, ax = plt.subplots(figsize=(10, 3.8))
    colors = [SERIES_1 if n == 3 else CONTEXT for n in data["n_groups"]]
    ax.barh(data["label"], data["share_of_brand_sales_3y"], height=0.6, color=colors)
    for y, (share, customers) in enumerate(
        zip(data["share_of_brand_sales_3y"], data["customers"], strict=True)
    ):
        ax.text(
            share,
            y,
            f"  {number(share * 100, 1)} % ({number(customers)} clientes)",
            va="center",
            fontsize=9,
            color=INK_SECONDARY,
        )
    _percent_axis(ax, "x")
    ax.set_xlim(0, data["share_of_brand_sales_3y"].max() * 1.45)
    ax.grid(axis="y", visible=False)
    core = data.loc[data["n_groups"] == 3].iloc[0]
    ax.set_title(
        f"El núcleo de {core['customers']} clientes genera el "
        f"{number(core['share_of_brand_sales_3y'] * 100)} % de la venta",
        pad=22,
    )
    _subtitle(ax, "% de la venta neta de la marca (3 años) por combinación de grupos")
    fig.tight_layout()
    return fig


# Archetypes


def k_selection_chart(k_table: pd.DataFrame, chosen_k: int) -> Figure:
    """Two measures on different scales, so two panels instead of a dual axis."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharex=True)
    # Label offsets chosen per panel so the annotation never sits on the line.
    for ax, col, title, offset, align in [
        (axes[0], "silhouette", "Separación (silueta, más alto es mejor)", (-10, 8), "right"),
        (
            axes[1],
            "stability_min_ari",
            "Estabilidad en el peor caso (ARI, más alto es mejor)",
            (10, 2),
            "left",
        ),
    ]:
        ax.plot(k_table["k"], k_table[col], color=CONTEXT, linewidth=2, marker="o", markersize=5)
        row = k_table.loc[k_table["k"] == chosen_k]
        ax.plot(
            row["k"],
            row[col],
            "o",
            color=SERIES_1,
            markersize=9,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )
        ax.annotate(
            f"k = {chosen_k}: {number(row[col].item(), 3)}",
            (chosen_k, row[col].item()),
            xytext=offset,
            ha=align,
            textcoords="offset points",
            fontweight="bold",
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Número de grupos (k)")
    fig.suptitle(
        f"k = {chosen_k}: casi la mejor separación y una estabilidad muy alta",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=13,
    )
    fig.tight_layout()
    return fig


def archetype_share_chart(profile: pd.DataFrame) -> Figure:
    """Share of customers vs share of sales per archetype (two series, one axis)."""
    data = profile.iloc[::-1]
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.barh(y + 0.17, data["share_of_customers"], height=0.32, color=CONTEXT, label="% de clientes")
    ax.barh(
        y - 0.17,
        data["share_of_brand_sales_3y"],
        height=0.32,
        color=SERIES_1,
        label="% de la venta (3 años)",
    )
    for yi, share in zip(y, data["share_of_brand_sales_3y"], strict=True):
        ax.text(share, yi - 0.17, f"  {share:.0%}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, data["name"])
    _percent_axis(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("Dos arquetipos (36 % de los clientes) generan el 75 % de la venta", pad=22)
    _subtitle(ax, "Peso de cada arquetipo en clientes y en venta neta de la marca")
    fig.tight_layout()
    return fig


def archetype_heatmap(profile: pd.DataFrame) -> Figure:
    """What characterizes each archetype: each feature compared with the average archetype."""
    columns = {  # column: (label, number format)
        "median_recency_days": ("Días desde\nla última compra", "{:.0f}"),
        "median_orders": ("Órdenes", "{:.0f}"),
        "median_avg_order_value": ("Ticket\npromedio", "{:,.0f}"),
        "launch_participation": ("Participación en\nlanzamientos", "{:.0%}"),
        "seasonal_share": ("% temporada\n(vs básicos)", "{:.0%}"),
        "avg_discount_pct": ("Descuento\npromedio", "{:.0f} %"),
        "online_share": ("% online", "{:.0%}"),
    }
    data = profile.dropna(subset=["median_recency_days"])[list(columns)].astype(float)
    z = (data - data.mean()) / data.std(ddof=0)

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.imshow(z.to_numpy(), cmap=DIVERGING, vmin=-2, vmax=2, aspect="auto")
    for i in range(z.shape[0]):
        for j, (_, fmt) in enumerate(columns.values()):
            ink = "white" if abs(z.iloc[i, j]) > 1.2 else INK
            ax.text(j, i, fmt.format(data.iloc[i, j]), ha="center", va="center", fontsize=9,
                    color=ink)  # fmt: skip
    ax.set_xticks(range(len(columns)), [label for label, _ in columns.values()], fontsize=8.5)
    ax.set_yticks(range(len(data)), profile.loc[data.index, "name"])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Qué caracteriza a cada arquetipo", pad=22)
    _subtitle(
        ax,
        "Valores típicos (medianas o promedios). En azul, por debajo del promedio de "
        "los arquetipos; en rojo, por encima",
    )
    fig.tight_layout()
    return fig


def purchase_month_chart(months: pd.DataFrame) -> Figure:
    """Sale hunters buy only in January, July and November (Buen Fin)."""
    month_names = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                   "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]  # fmt: skip
    hunters = months.loc["sale_hunters"]
    others = months.drop(index=["sale_hunters"]).mean()
    x = np.arange(12)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(x - 0.18, others, width=0.34, color=CONTEXT, label="Resto de arquetipos (promedio)")
    ax.bar(x + 0.18, hunters, width=0.34, color=SERIES_1, label="Cazadores de rebajas")
    for m in [0, 6, 10]:
        ax.text(
            m + 0.18,
            hunters.iloc[m],
            f"{hunters.iloc[m]:.0%}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xticks(x, month_names)
    _percent_axis(ax)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, hunters.max() * 1.35)
    ax.legend(loc="upper right")
    ax.set_title("Los cazadores de rebajas solo compran en enero, julio y el Buen Fin", pad=22)
    _subtitle(ax, "% de las órdenes de cada grupo según el mes de compra (2024-2026)")
    fig.tight_layout()
    return fig


# Tables

COLUMN_LABELS = {
    # issue log and comparison
    "code": "Código",
    "table": "Tabla",
    "issue": "Problema",
    "rows_affected": "Filas afectadas",
    "treatment": "Tratamiento",
    "metric": "Métrica",
    "naive": "Sin limpiar",
    "clean": "Limpio",
    "explained_by": "Reglas que lo explican",
    # data dictionary
    "feature": "Variable",
    "description": "Descripción",
    # questions and sensitivity
    "question": "Pregunta",
    "criterion": "Criterio",
    "scenario": "Escenario",
    "customers": "Clientes",
    "share_of_buyers": "% de los compradores",
    "share_of_brand_sales_3y": "% de la venta (3 años)",
    "share_of_brand_sales_12m": "% de la venta (12 meses)",
    "contactable_share": "% contactable",
    "basics_only_share": "% que solo compra básicos",
    # choice of k
    "silhouette": "Separación (silueta)",
    "stability_min_ari": "Estabilidad en el peor caso (ARI)",
    "smallest_cluster": "Grupo más pequeño",
    # archetype profile
    "name": "Arquetipo",
    "share_of_customers": "% de los clientes",
    "net_sales_3y": "Venta neta 3 años (MXN)",
    "median_recency_days": "Días desde la última compra (mediana)",
    "median_orders": "Órdenes (mediana)",
    "median_days_between": "Días entre compras (mediana)",
    "median_avg_order_value": "Ticket promedio (mediana, MXN)",
    "launch_participation": "Participación en lanzamientos",
    "seasonal_share": "% de venta de temporada",
    "avg_discount_pct": "Descuento promedio (%)",
    "online_share": "% de órdenes online",
    "q1_new_collection_fan": "% en P1",
    "q2_recurrent": "% en P2",
    "q3_high_value": "% en P3",
    "contactable": "% contactable",
}

# Proportions between 0 and 1, shown as percentages.
SHARE_COLUMNS = {
    "share_of_buyers", "share_of_brand_sales_3y", "share_of_brand_sales_12m",
    "contactable_share", "basics_only_share", "share_of_customers", "launch_participation",
    "seasonal_share", "online_share", "q1_new_collection_fan", "q2_recurrent", "q3_high_value",
    "contactable",
}  # fmt: skip


# Measures compared at three decimals in the narrative (0,472 against 0,480).
THREE_DECIMALS = {"silhouette", "stability_min_ari"}


def _cell(value: object, decimals: int) -> object:
    if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Real):
        return value
    if pd.isna(value):
        return ""
    return number(value, 0 if float(value).is_integer() else decimals)


def table(df: pd.DataFrame) -> pd.DataFrame:
    """The same table for display: Spanish headers, Spanish numbers, proportions as %."""
    shown = df.copy()
    for col in shown.columns:
        if col in SHARE_COLUMNS:
            shown[col] = shown[col].map(lambda v: "" if pd.isna(v) else f"{number(v * 100, 1)} %")
        else:
            decimals = 3 if col in THREE_DECIMALS else 2
            shown[col] = shown[col].map(lambda v, d=decimals: _cell(v, d))
    index_name = COLUMN_LABELS.get(shown.index.name, shown.index.name)
    return shown.rename(columns=COLUMN_LABELS).rename_axis(index=index_name)
