"""Sprint 5: Exploration mode — interactive journal discovery dashboard."""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

PARQUET = Path("journals_scored.parquet")

# Plotly country name normalisations (DOAJ names -> Plotly/ISO names)
_COUNTRY_NORM = {
    "Iran, Islamic Republic of": "Iran",
    "Korea, Republic of": "South Korea",
    "Korea, Democratic People's Republic of": "North Korea",
    "Russian Federation": "Russia",
    "Türkiye": "Turkey",
    "Viet Nam": "Vietnam",
    "Bolivia, Plurinational State of": "Bolivia",
    "Moldova, Republic of": "Moldova",
    "Tanzania, United Republic of": "Tanzania",
    "Venezuela, Bolivarian Republic of": "Venezuela",
    "Syrian Arab Republic": "Syria",
    "Congo, Democratic Republic of the": "DR Congo",
    "Lao People's Democratic Republic": "Laos",
    "Brunei Darussalam": "Brunei",
    "Palestine, State of": "Palestine",
    "Macedonia, the Former Yugoslav Republic of": "North Macedonia",
    "Trinidad and Tobago": "Trinidad and Tobago",
    "Czechia": "Czech Republic",
}

APC_TIERS = {
    "Free ($0)":        (0, 0),
    "Budget ($1-500)":  (1, 500),
    "Mid ($501-1500)":  (501, 1500),
    "Premium (>$1500)": (1501, 99_999),
}


# ---------------------------------------------------------------------------
# Data loading (cached — runs once per session)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    df["country_plot"] = df["country"].map(_COUNTRY_NORM).fillna(df["country"])
    df["main_subject"] = df["subjects"].apply(_main_subject)
    df["main_language"] = df["languages"].apply(
        lambda v: str(v).split(",")[0].strip() if pd.notna(v) else "Unknown"
    )
    # Ensure nullable booleans behave predictably in boolean masks
    for col in ("is_predatory", "is_core", "has_doi", "has_waiver", "plagiarism_check"):
        df[col] = df[col].fillna(False).astype(bool)
    return df


def _main_subject(val) -> str:
    """Extract the top-level subject category (before first colon/pipe)."""
    if pd.isna(val):
        return "Other"
    first = str(val).split("|")[0].split(";")[0].strip()
    return first.split(":")[0].strip() if ":" in first else first


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Powered Journal Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📚 AI Powered Journal Recommender")
st.caption("Explore 22,890+ open-access journals from the DOAJ — Exploration Mode")

df = load_data()

# ---------------------------------------------------------------------------
# Sidebar — filters
# ---------------------------------------------------------------------------

st.sidebar.header("Filters")

topic_q = st.sidebar.text_input(
    "Keyword search", placeholder="title, subject, publisher…"
)

all_countries = sorted(df["country"].dropna().unique())
sel_countries = st.sidebar.multiselect("Country", all_countries)

sel_quartiles = st.sidebar.multiselect(
    "SCImago quartile", ["Q1", "Q2", "Q3", "Q4", "Unranked"]
)

sel_apc_tiers = st.sidebar.multiselect("APC tier", list(APC_TIERS))

weeks_max = int(df["weeks_to_pub"].max())
weeks_range = st.sidebar.slider("Max weeks to publication", 1, weeks_max, weeks_max)

all_languages = sorted({
    lang.strip()
    for v in df["languages"].dropna()
    for lang in str(v).split(",")
    if lang.strip()
})
sel_languages = st.sidebar.multiselect("Language", all_languages)

sel_clusters = st.sidebar.multiselect(
    "Cluster", sorted(df["cluster_label"].dropna().unique())
)

st.sidebar.subheader("Quality")
excl_predatory = st.sidebar.checkbox("Exclude predatory journals", value=True)
only_core      = st.sidebar.checkbox("Scopus / WoS indexed (is_core)")
only_doi       = st.sidebar.checkbox("Has DOI")
only_waiver    = st.sidebar.checkbox("Has APC waiver")
only_plagiarism = st.sidebar.checkbox("Plagiarism screening")

# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------

mask = pd.Series(True, index=df.index)

if topic_q:
    q = topic_q.lower()
    mask &= (
        df["title"].str.lower().str.contains(q, na=False)
        | df["subjects"].str.lower().str.contains(q, na=False)
        | df["keywords"].str.lower().str.contains(q, na=False)
        | df["publisher"].str.lower().str.contains(q, na=False)
    )
if sel_countries:
    mask &= df["country"].isin(sel_countries)
if sel_quartiles:
    qm = pd.Series(False, index=df.index)
    if "Unranked" in sel_quartiles:
        qm |= df["sjr_quartile"].isna()
    rest = [q for q in sel_quartiles if q != "Unranked"]
    if rest:
        qm |= df["sjr_quartile"].isin(rest)
    mask &= qm
if sel_apc_tiers:
    am = pd.Series(False, index=df.index)
    for tier in sel_apc_tiers:
        lo, hi = APC_TIERS[tier]
        am |= df["apc_usd"].between(lo, hi)
    mask &= am

mask &= df["weeks_to_pub"] <= weeks_range

if sel_languages:
    lm = pd.Series(False, index=df.index)
    for lang in sel_languages:
        lm |= df["languages"].str.contains(lang, na=False, case=False)
    mask &= lm
if sel_clusters:
    mask &= df["cluster_label"].isin(sel_clusters)
if excl_predatory:
    mask &= ~df["is_predatory"]
if only_core:
    mask &= df["is_core"]
if only_doi:
    mask &= df["has_doi"]
if only_waiver:
    mask &= df["has_waiver"]
if only_plagiarism:
    mask &= df["plagiarism_check"]

fdf = df[mask].copy()

# ---------------------------------------------------------------------------
# KPI metrics
# ---------------------------------------------------------------------------

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Journals",     f"{len(fdf):,}")
c2.metric("Countries",    f"{fdf['country'].nunique():,}")
c3.metric("Avg score",    f"{fdf['final_score'].mean():.3f}" if len(fdf) else "—")
c4.metric("Median APC",   f"${fdf['apc_usd'].median():.0f}" if len(fdf) else "—")
c5.metric("Q1 journals",  f"{(fdf['sjr_quartile'] == 'Q1').sum():,}")

if len(fdf) == 0:
    st.warning("No journals match the current filters — try relaxing some constraints.")
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# Chart row 1: World map + APC vs Score scatter
# ---------------------------------------------------------------------------

col_map, col_scatter = st.columns([3, 2], gap="large")

with col_map:
    st.subheader("Geographic distribution")
    country_agg = (
        fdf.groupby("country_plot", as_index=False)
        .agg(
            Journals=("title", "count"),
            avg_score=("final_score", "mean"),
            avg_apc=("apc_usd", "mean"),
        )
        .rename(columns={"country_plot": "Country"})
    )
    fig_map = px.scatter_geo(
        country_agg,
        locations="Country",
        locationmode="country names",
        size="Journals",
        color="avg_score",
        color_continuous_scale="Viridis",
        size_max=55,
        hover_name="Country",
        hover_data={
            "Journals": True,
            "avg_score": ":.3f",
            "avg_apc": ":$.0f",
            "Country": False,
        },
        labels={"avg_score": "Avg score", "avg_apc": "Avg APC"},
        projection="natural earth",
    )
    fig_map.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(title="Avg score", thickness=12, len=0.6),
        geo=dict(showframe=False, showcoastlines=True, coastlinecolor="lightgrey"),
        height=360,
    )
    st.plotly_chart(fig_map, use_container_width=True)

with col_scatter:
    st.subheader("APC vs score")
    # Cap at 4,000 points for browser performance
    plot_df = fdf.nlargest(4_000, "final_score") if len(fdf) > 4_000 else fdf
    fig_scatter = px.scatter(
        plot_df,
        x="apc_usd",
        y="final_score",
        color="cluster_label",
        hover_name="title",
        hover_data={
            "publisher": True,
            "country": True,
            "sjr_quartile": True,
            "apc_usd": ":$,.0f",
            "final_score": ":.3f",
            "cluster_label": False,
        },
        labels={
            "apc_usd": "APC (USD)",
            "final_score": "Score",
            "cluster_label": "Cluster",
        },
        opacity=0.55,
        category_orders={"cluster_label": sorted(fdf["cluster_label"].dropna().unique())},
    )
    fig_scatter.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=360,
        legend=dict(
            title="Cluster",
            orientation="v",
            yanchor="top", y=1,
            xanchor="right", x=1,
            font=dict(size=11),
        ),
        xaxis=dict(title="APC (USD)"),
        yaxis=dict(title="Score"),
    )
    fig_scatter.update_traces(marker=dict(size=5))
    st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Chart row 2: Subject × Country heatmap
# ---------------------------------------------------------------------------

st.subheader("Subject area × Country heatmap")

N_SUBJ = 12
N_CNTRY = 15

top_subjects  = fdf["main_subject"].value_counts().head(N_SUBJ).index.tolist()
top_countries = fdf["country"].value_counts().head(N_CNTRY).index.tolist()

heat_src = fdf[
    fdf["main_subject"].isin(top_subjects) & fdf["country"].isin(top_countries)
]

if len(heat_src) == 0:
    st.info("Not enough data to render heatmap with current filters.")
else:
    heat_pivot = (
        heat_src.groupby(["main_subject", "country"])
        .size()
        .reset_index(name="count")
        .pivot(index="main_subject", columns="country", values="count")
        .reindex(index=top_subjects, columns=top_countries)
        .fillna(0)
        .astype(int)
    )
    fig_heat = px.imshow(
        heat_pivot,
        color_continuous_scale="Blues",
        aspect="auto",
        text_auto=True,
        labels=dict(x="Country", y="Subject area", color="Journals"),
    )
    fig_heat.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=420,
        xaxis_tickangle=-35,
        coloraxis_showscale=False,
        font=dict(size=12),
    )
    fig_heat.update_traces(textfont_size=10)
    st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Table + CSV export
# ---------------------------------------------------------------------------

TABLE_COLS = {
    "title":         "Title",
    "publisher":     "Publisher",
    "country":       "Country",
    "sjr_quartile":  "Quartile",
    "apc_usd":       "APC ($)",
    "weeks_to_pub":  "Weeks",
    "final_score":   "Score",
    "cluster_label": "Cluster",
    "license":       "License",
    "is_core":       "Scopus/WoS",
    "url":           "Journal website",
    "authors_url":   "Submit / Author guide",
    "doaj_url":      "DOAJ link",
}
present_cols = {k: v for k, v in TABLE_COLS.items() if k in fdf.columns}

t_ctrl, t_export = st.columns([3, 1])
with t_ctrl:
    st.subheader(f"Journal table — {len(fdf):,} results")
    sort_col = st.selectbox(
        "Sort by",
        ["Score", "APC ($)", "Weeks", "Quartile"],
        index=0,
    )
with t_export:
    st.write("")
    st.write("")
    csv_bytes = fdf[list(present_cols)].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name="journals_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

# Build display dataframe
display = fdf[list(present_cols)].rename(columns=present_cols).copy()
display["Score"]   = display["Score"].round(3)
display["APC ($)"] = display["APC ($)"].fillna(0).astype(int)
display["Weeks"]   = display["Weeks"].fillna(0).astype(int)

_sort_map = {"Score": ("Score", False), "APC ($)": ("APC ($)", True),
             "Weeks": ("Weeks", True), "Quartile": ("Quartile", True)}
sort_field, sort_asc = _sort_map[sort_col]
display = display.sort_values(sort_field, ascending=sort_asc)

st.dataframe(
    display,
    use_container_width=True,
    height=450,
    column_config={
        "DOAJ link":           st.column_config.LinkColumn("DOAJ link",           display_text="DOAJ"),
        "Journal website":     st.column_config.LinkColumn("Journal website",     display_text="Website"),
        "Submit / Author guide": st.column_config.LinkColumn("Submit / Author guide", display_text="Submit"),
        "Scopus/WoS": st.column_config.CheckboxColumn("Scopus/WoS"),
        "Score": st.column_config.NumberColumn("Score", format="%.3f"),
        "APC ($)": st.column_config.NumberColumn("APC ($)", format="$%d"),
    },
    hide_index=True,
)
