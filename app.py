"""AI Powered Journal Recommender — unified single-app entry point.

Two modes selectable from the sidebar:
  🗺  Exploration   — interactive dashboard, no article needed
  🔬  Consultation  — personalised top-10 ranking for a specific article

Run with:
    streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import nlp as nlp_engine
import llm as llm_engine
from api_client import APIClient, APIError

PARQUET = Path("journals_scored.parquet")

CONFIDENCE_BADGE = {
    "High":   "🟢 High",
    "Medium": "🟡 Medium",
    "Low":    "🔴 Low",
}

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
    "Czechia": "Czech Republic",
}

APC_TIERS = {
    "Free ($0)":        (0, 0),
    "Budget ($1-500)":  (1, 500),
    "Mid ($501-1500)":  (501, 1500),
    "Premium (>$1500)": (1501, 99_999),
}

GUEST_SESSION_LIMIT = 3   # free queries per browser session before login is required


# ---------------------------------------------------------------------------
# Shared data + model loading (cached once per session)
# ---------------------------------------------------------------------------

def _main_subject(val) -> str:
    if pd.isna(val):
        return "Other"
    first = str(val).split("|")[0].split(";")[0].strip()
    return first.split(":")[0].strip() if ":" in first else first


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    for col in ("is_predatory", "is_core", "has_doi", "has_waiver", "plagiarism_check"):
        df[col] = df[col].fillna(False).astype(bool)
    df["country_plot"]  = df["country"].map(_COUNTRY_NORM).fillna(df["country"])
    df["main_subject"]  = df["subjects"].apply(_main_subject)
    df["main_language"] = df["languages"].apply(
        lambda v: str(v).split(",")[0].strip() if pd.notna(v) else "Unknown"
    )
    return df


@st.cache_resource
def load_nlp_model():
    return nlp_engine.load_model()


# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Powered Journal Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar — mode selector (always visible at top)
# ---------------------------------------------------------------------------

st.sidebar.image("https://img.icons8.com/color/96/open-book.png", width=60)
st.sidebar.title("AI Powered Journal Recommender")
st.sidebar.markdown("*Open-access journal finder for researchers*")
st.sidebar.divider()

mode = st.sidebar.radio(
    "Select mode",
    options=["🗺  Exploration", "🔬  Consultation"],
    label_visibility="collapsed",
)
st.sidebar.divider()

if "prev_mode" not in st.session_state:
    st.session_state.prev_mode = mode
if st.session_state.prev_mode != mode:
    label = "Exploration" if "Exploration" in mode else "Consultation"
    st.toast(f"Switching to {label} mode…", icon="🔄")
    st.session_state.prev_mode = mode

df = load_data()

# ===========================================================================
# EXPLORATION MODE
# ===========================================================================

if "Exploration" in mode:

    st.title("🗺  Exploration Mode")
    st.caption("Browse and filter 22,890+ open-access journals from the DOAJ")

    # --- Sidebar filters ---
    st.sidebar.header("Filters")

    topic_q = st.sidebar.text_input("Keyword search", placeholder="title, subject, publisher…")

    all_countries = sorted(df["country"].dropna().unique())
    sel_countries = st.sidebar.multiselect("Country", all_countries)

    sel_quartiles = st.sidebar.multiselect("SCImago quartile", ["Q1", "Q2", "Q3", "Q4", "Unranked"])
    sel_apc_tiers = st.sidebar.multiselect("APC tier", list(APC_TIERS))

    weeks_max   = int(df["weeks_to_pub"].max())
    weeks_range = st.sidebar.slider("Max weeks to publication", 1, weeks_max, weeks_max)

    all_languages = sorted({
        lang.strip()
        for v in df["languages"].dropna()
        for lang in str(v).split(",")
        if lang.strip()
    })
    sel_languages = st.sidebar.multiselect("Language", all_languages)
    sel_clusters  = st.sidebar.multiselect("Cluster", sorted(df["cluster_label"].dropna().unique()))

    st.sidebar.subheader("Quality")
    excl_pred_e  = st.sidebar.checkbox("Exclude predatory journals", value=True, key="e_pred")
    only_core_e  = st.sidebar.checkbox("Scopus / WoS indexed",        key="e_core")
    only_doi_e   = st.sidebar.checkbox("Has DOI",                      key="e_doi")
    only_waiver_e = st.sidebar.checkbox("Has APC waiver",              key="e_waiver")
    only_plag_e  = st.sidebar.checkbox("Plagiarism screening",         key="e_plag")

    # --- Apply filters ---
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
    if excl_pred_e:  mask &= ~df["is_predatory"]
    if only_core_e:  mask &= df["is_core"]
    if only_doi_e:   mask &= df["has_doi"]
    if only_waiver_e: mask &= df["has_waiver"]
    if only_plag_e:  mask &= df["plagiarism_check"]

    fdf = df[mask].copy()

    # --- KPIs ---
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Journals",    f"{len(fdf):,}")
    c2.metric("Countries",   f"{fdf['country'].nunique():,}")
    c3.metric("Avg score",   f"{fdf['final_score'].mean():.3f}" if len(fdf) else "—")
    c4.metric("Median APC",  f"${fdf['apc_usd'].median():.0f}"  if len(fdf) else "—")
    c5.metric("Q1 journals", f"{(fdf['sjr_quartile'] == 'Q1').sum():,}")

    if len(fdf) == 0:
        st.warning("No journals match the current filters — try relaxing some constraints.")
        st.stop()

    st.divider()

    # --- World map + Scatter ---
    col_map, col_scatter = st.columns([3, 2], gap="large")

    with col_map:
        st.subheader("Geographic distribution")
        country_agg = (
            fdf.groupby("country_plot", as_index=False)
            .agg(Journals=("title", "count"), avg_score=("final_score", "mean"), avg_apc=("apc_usd", "mean"))
            .rename(columns={"country_plot": "Country"})
        )
        fig_map = px.scatter_geo(
            country_agg, locations="Country", locationmode="country names",
            size="Journals", color="avg_score", color_continuous_scale="Viridis",
            size_max=55, hover_name="Country",
            hover_data={"Journals": True, "avg_score": ":.3f", "avg_apc": ":$.0f", "Country": False},
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
        plot_df = fdf.nlargest(4_000, "final_score") if len(fdf) > 4_000 else fdf
        fig_scatter = px.scatter(
            plot_df, x="apc_usd", y="final_score", color="cluster_label",
            hover_name="title",
            hover_data={"publisher": True, "country": True, "sjr_quartile": True,
                        "apc_usd": ":$,.0f", "final_score": ":.3f", "cluster_label": False},
            labels={"apc_usd": "APC (USD)", "final_score": "Score", "cluster_label": "Cluster"},
            opacity=0.55,
            category_orders={"cluster_label": sorted(fdf["cluster_label"].dropna().unique())},
        )
        fig_scatter.update_layout(
            margin=dict(l=0, r=0, t=0, b=0), height=360,
            legend=dict(title="Cluster", orientation="v", yanchor="top", y=1,
                        xanchor="right", x=1, font=dict(size=11)),
        )
        fig_scatter.update_traces(marker=dict(size=5))
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()

    # --- Heatmap ---
    st.subheader("Subject area × Country heatmap")
    top_subjects  = fdf["main_subject"].value_counts().head(12).index.tolist()
    top_countries = fdf["country"].value_counts().head(15).index.tolist()
    heat_src = fdf[fdf["main_subject"].isin(top_subjects) & fdf["country"].isin(top_countries)]

    if len(heat_src) == 0:
        st.info("Not enough data to render heatmap with current filters.")
    else:
        heat_pivot = (
            heat_src.groupby(["main_subject", "country"]).size()
            .reset_index(name="count")
            .pivot(index="main_subject", columns="country", values="count")
            .reindex(index=top_subjects, columns=top_countries).fillna(0).astype(int)
        )
        fig_heat = px.imshow(heat_pivot, color_continuous_scale="Blues", aspect="auto",
                             text_auto=True, labels=dict(x="Country", y="Subject area", color="Journals"))
        fig_heat.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=420,
                               xaxis_tickangle=-35, coloraxis_showscale=False, font=dict(size=12))
        fig_heat.update_traces(textfont_size=10)
        st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()

    # --- Table ---
    TABLE_COLS = {
        "title": "Title", "publisher": "Publisher", "country": "Country",
        "sjr_quartile": "Quartile", "apc_usd": "APC ($)", "weeks_to_pub": "Weeks",
        "final_score": "Score", "cluster_label": "Cluster", "license": "License",
        "is_core": "Scopus/WoS", "url": "Journal website",
        "authors_url": "Submit / Author guide", "doaj_url": "DOAJ link",
    }
    present_cols = {k: v for k, v in TABLE_COLS.items() if k in fdf.columns}

    t_ctrl, t_export = st.columns([3, 1])
    with t_ctrl:
        st.subheader(f"Journal table — {len(fdf):,} results")
        sort_col = st.selectbox("Sort by", ["Score", "APC ($)", "Weeks", "Quartile"], index=0)
    with t_export:
        st.write(""); st.write("")
        st.download_button("Download CSV",
                           data=fdf[list(present_cols)].to_csv(index=False).encode("utf-8"),
                           file_name="journals_filtered.csv", mime="text/csv",
                           use_container_width=True)

    display = fdf[list(present_cols)].rename(columns=present_cols).copy()
    display["Score"]   = display["Score"].round(3)
    display["APC ($)"] = display["APC ($)"].fillna(0).astype(int)
    display["Weeks"]   = display["Weeks"].fillna(0).astype(int)
    _sort_map = {"Score": ("Score", False), "APC ($)": ("APC ($)", True),
                 "Weeks": ("Weeks", True), "Quartile": ("Quartile", True)}
    sf, sa = _sort_map[sort_col]
    display = display.sort_values(sf, ascending=sa)

    st.dataframe(display, use_container_width=True, height=450,
                 column_config={
                     "DOAJ link":             st.column_config.LinkColumn("DOAJ link",    display_text="DOAJ"),
                     "Journal website":       st.column_config.LinkColumn("Journal website", display_text="Website"),
                     "Submit / Author guide": st.column_config.LinkColumn("Submit / Author guide", display_text="Submit"),
                     "Scopus/WoS": st.column_config.CheckboxColumn("Scopus/WoS"),
                     "Score":  st.column_config.NumberColumn("Score",  format="%.3f"),
                     "APC ($)": st.column_config.NumberColumn("APC ($)", format="$%d"),
                 }, hide_index=True)


# ===========================================================================
# CONSULTATION MODE
# ===========================================================================

else:

    st.title("🔬  Consultation Mode")
    st.caption("Enter your article details and receive a personalised top-10 open-access journal ranking")

    # --- Sidebar: API login ---
    _client    = APIClient()
    _api_online = _client.health()

    st.sidebar.header("Account")
    if _api_online:
        if "api_token" not in st.session_state:
            st.session_state.api_token = None
            st.session_state.api_user  = None

        if st.session_state.api_token is None:
            with st.sidebar.expander("Log in", expanded=True):
                _email = st.text_input("Email",    key="login_email")
                _pwd   = st.text_input("Password", type="password", key="login_pwd")
                c1, c2 = st.columns(2)
                if c1.button("Login", use_container_width=True):
                    try:
                        st.session_state.api_token = _client.login(_email, _pwd)
                        st.session_state.api_user  = _client.me(st.session_state.api_token)
                        st.session_state.guest_queries = 0  # reset on login
                        st.rerun()
                    except APIError as e:
                        st.error(e.detail)
                if c2.button("Register", use_container_width=True):
                    try:
                        _client.register(_email, _pwd)
                        st.success("Registered! Please log in.")
                    except APIError as e:
                        st.error(e.detail)
            # Guest query counter
            _g_used = st.session_state.get("guest_queries", 0)
            _g_left = GUEST_SESSION_LIMIT - _g_used
            if _g_left > 0:
                st.sidebar.info(
                    f"**{_g_left}** guest quer{'y' if _g_left == 1 else 'ies'} left this session.  \n"
                    f"Log in for **50/month**.")
            else:
                st.sidebar.warning("Guest limit reached.  \nLog in for **50 queries/month**.")
        else:
            _user = st.session_state.api_user or {}
            st.sidebar.success(f"**{_user.get('email','')}**  \nRole: `{_user.get('role','')}`")
            if st.sidebar.button("Logout"):
                try:
                    _client._post("/auth/logout", token=st.session_state.api_token)
                except Exception:
                    pass
                st.session_state.api_token = None
                st.session_state.api_user  = None
                st.rerun()

            # --- Search history panel ---
            st.sidebar.divider()
            st.sidebar.subheader("Search history")
            _h_col1, _h_col2 = st.sidebar.columns([3, 1])
            with _h_col2:
                if st.button("↺", help="Refresh history"):
                    st.session_state.pop("hist_items", None)
            if "hist_items" not in st.session_state:
                try:
                    st.session_state.hist_items = _client.get_history(
                        st.session_state.api_token, limit=10)
                except Exception:
                    st.session_state.hist_items = []
            _hist_items = st.session_state.get("hist_items", [])
            if not _hist_items:
                st.sidebar.caption("No past searches yet.")
            else:
                for _hi in _hist_items:
                    _t = _hi["title"]
                    _t_short = (_t[:42] + "…") if len(_t) > 42 else _t
                    _area_str = f"  ·  {_hi['area'][:20]}" if _hi.get("area") else ""
                    _date = _hi["created_at"][:10]
                    st.sidebar.markdown(
                        f"<small>**{_t_short}**{_area_str}<br>{_date} · {_hi['result_count']} results</small>",
                        unsafe_allow_html=True)
                    if st.sidebar.button("Re-run ↩", key=f"hist_{_hi['id']}",
                                         use_container_width=True):
                        _load_ok = False
                        try:
                            _saved = _client.get_history_results(
                                st.session_state.api_token, _hi["id"])
                            st.session_state.history_display = {
                                "results": _saved,
                                "meta": {
                                    "title": _hi["title"],
                                    "area":  _hi.get("area", ""),
                                    "date":  _date,
                                },
                            }
                            st.session_state.history_prefill = {
                                "title":    _hi["title"],
                                "area":     _hi.get("area", ""),
                                "abstract": _hi.get("abstract", ""),
                            }
                            _load_ok = True
                        except Exception as _he:
                            st.sidebar.error(f"Could not load: {_he}")
                        if _load_ok:
                            st.rerun()
    else:
        st.sidebar.caption("API offline — direct NLP mode")

    st.sidebar.divider()

    # --- Sidebar: priority sliders ---
    st.sidebar.header("Your priorities")
    st.sidebar.caption("Adjust to reflect what matters most. Auto-normalised.")

    speed_raw    = st.sidebar.slider("⚡ Speed (fast publication)",    0, 100, 33)
    prestige_raw = st.sidebar.slider("🏆 Prestige (impact / indexing)", 0, 100, 34)
    cost_raw     = st.sidebar.slider("💰 Cost (low APC)",              0, 100, 33)

    _t = speed_raw + prestige_raw + cost_raw or 1
    speed_w    = speed_raw    / _t
    prestige_w = prestige_raw / _t
    cost_w     = cost_raw     / _t

    st.sidebar.markdown(
        f"**Effective:** Speed {speed_w:.0%} · Prestige {prestige_w:.0%} · Cost {cost_w:.0%}"
    )
    st.sidebar.divider()

    st.sidebar.subheader("Options")
    top_k_candidates = st.sidebar.select_slider(
        "NLP candidate pool", options=[50, 100, 200, 500], value=200)
    excl_pred_c = st.sidebar.checkbox("Exclude predatory journals", value=True, key="c_pred")
    only_core_c = st.sidebar.checkbox("Scopus / WoS only",          key="c_core")

    # --- Consume one-shot session state ---
    _history_prefill  = st.session_state.pop("history_prefill", None)
    _history_display  = st.session_state.get("history_display", None)

    # --- Input form ---
    with st.form("query_form"):
        col_title, col_area = st.columns([3, 2])
        with col_title:
            title_input = st.text_input(
                "Article title *",
                value=_history_prefill["title"] if _history_prefill else "",
                placeholder="e.g. Deep learning for early cancer detection in CT scans")
        with col_area:
            area_input = st.text_input(
                "Topic area",
                value=_history_prefill.get("area", "") if _history_prefill else "",
                placeholder="e.g. machine learning, oncology")
        abstract_input = st.text_area(
            "Abstract (optional — improves matching)",
            value=_history_prefill.get("abstract", "") if _history_prefill else "",
            placeholder="Paste your abstract here…", height=120)
        submitted = st.form_submit_button(
            "Find matching journals", type="primary", use_container_width=True)

    # A new submission clears any loaded history display
    if submitted:
        st.session_state.pop("history_display", None)
        _history_display = None

    if not submitted and not _history_display:
        st.stop()

    # --- History display fast-path (no NLP re-run) ---
    if _history_display:
        _hmeta = _history_display["meta"]
        st.info(
            f"📋 Showing saved results — *{_hmeta['title'][:80]}*  \n"
            f"Originally run on **{_hmeta['date']}**")
        if st.button("✕ Clear — run a new query", type="secondary"):
            st.session_state.pop("history_display", None)
            st.rerun()
        top10 = pd.DataFrame(_history_display["results"])
        top10["consultation_score"] = top10["publication_score"]
        title_input   = _hmeta["title"]
        area_input    = _hmeta.get("area", "")
        explanations  = [""] * len(top10)
        speed_w = prestige_w = cost_w = 1 / 3

    else:
        # --- Live query path ---
        if not title_input.strip():
            st.error("Please enter an article title.")
            st.stop()

        level = 1 + bool(area_input.strip()) + bool(abstract_input.strip())
        st.info(f"Input level **{level}/3** — "
                f"{'title only' if level==1 else 'title + area' if level==2 else 'title + area + abstract'}")

    token        = st.session_state.get("api_token")
    api_response = None

    if not _history_display:
        # Path A: API mode
        if token and _api_online:
            with st.spinner("Matching via API…"):
                try:
                    api_response = _client.match(
                        token=token, title=title_input.strip(),
                        area=area_input.strip() or None,
                        abstract=abstract_input.strip() or None,
                        top_k=10, speed=speed_raw, prestige=prestige_raw, cost=cost_raw)
                except APIError as e:
                    if e.status_code == 429:
                        st.error(e.detail); st.stop()
                    st.warning(f"API error {e.status_code}: {e.detail} — using direct mode")
                    api_response = None

        if api_response:
            used      = api_response.get("queries_used", 0)
            remaining = api_response.get("queries_remaining")
            st.sidebar.info(
                f"Queries this month: **{used} / 50**"
                if remaining is not None else
                f"Queries this month: **{used}** (unlimited)")
            top10 = pd.DataFrame(api_response["results"])
            top10["consultation_score"] = top10["publication_score"]

        # Path B: direct NLP mode
        else:
            # Guest cap check
            if not token:
                _g_used = st.session_state.get("guest_queries", 0)
                if _g_used >= GUEST_SESSION_LIMIT:
                    st.error(
                        f"You've used all **{GUEST_SESSION_LIMIT}** guest queries for this session.  \n"
                        f"Log in or create a free account for **50 queries/month**.")
                    st.stop()

            with st.spinner("Loading NLP model…"):
                load_nlp_model()
            pre_df = df[~df["is_predatory"]].copy() if excl_pred_c else df.copy()
            if only_core_c:
                pre_df = pre_df[pre_df["is_core"]]
            with st.spinner(f"Matching across {len(pre_df):,} journals…"):
                try:
                    nlp_results = nlp_engine.match(
                        title=title_input.strip(),
                        area=area_input.strip() or None,
                        abstract=abstract_input.strip() or None,
                        df=pre_df, top_k=top_k_candidates)
                except FileNotFoundError as exc:
                    st.error(str(exc)); st.stop()
            if len(nlp_results) == 0:
                st.warning("No results — try relaxing filters."); st.stop()

            ADJUSTABLE = 0.35
            nlp_results["publication_score"] = (
                nlp_results["nlp_score"]                    * 0.40
                + nlp_results["b3_score"].fillna(0.0)        * 0.25
                + nlp_results["weeks_score"].fillna(0.5)     * speed_w    * ADJUSTABLE
                + nlp_results["b4_score"].fillna(0.0)        * prestige_w * ADJUSTABLE
                + nlp_results["b5_score"].fillna(0.5)        * cost_w     * ADJUSTABLE
            ).clip(0, 1)
            nlp_results["consultation_score"] = nlp_results["publication_score"]
            top10 = nlp_results.nlargest(10, "consultation_score").reset_index(drop=True)

            # Increment guest counter after a successful direct query
            if not token:
                st.session_state["guest_queries"] = st.session_state.get("guest_queries", 0) + 1

    if len(top10) == 0:
        st.warning("No results found."); st.stop()

    # AI explanations — skip for history display (results are already cached in DB)
    if not _history_display:
        _ck = f"llm_exp_{hash((title_input, area_input or '', (abstract_input or '')[:100]))}"
        if llm_engine.is_available():
            if _ck not in st.session_state:
                jlist = [{"title": str(r.get("title") or ""),
                          "subjects": str(r.get("subjects") or r.get("cluster_label") or ""),
                          "sjr_quartile": r.get("sjr_quartile"),
                          "apc_usd": float(r.get("apc_usd") or 0),
                          "is_core": bool(r.get("is_core") or False),
                          "cluster_label": r.get("cluster_label")}
                         for _, r in top10.iterrows()]
                with st.spinner("Generating AI explanations…"):
                    st.session_state[_ck] = llm_engine.explain_recommendations(
                        title_input.strip(), area_input.strip() or None,
                        abstract_input.strip() or None, jlist)
            explanations = st.session_state[_ck]
        else:
            explanations = [""] * len(top10)

    # Results header
    st.divider()
    st.subheader(f"Top {len(top10)} journals for: *{title_input[:80]}*")
    st.caption(
        f"B2 NLP 40% · B3 Editorial 25% · "
        f"Speed {speed_w*35:.0f}% · Prestige {prestige_w*35:.0f}% · Cost {cost_w*35:.0f}%")

    # Result cards
    for idx, row in top10.iterrows():
        rank         = idx + 1
        badge        = CONFIDENCE_BADGE.get(str(row.get("confidence", "")), "⚪ —")
        quartile     = row.get("sjr_quartile")
        quartile_str = str(quartile) if pd.notna(quartile) else "Unranked"
        apc          = float(row.get("apc_usd", 0) or 0)
        apc_str      = "Free ($0)" if apc == 0 else f"${apc:,.0f}"
        weeks        = int(row.get("weeks_to_pub") or 0)
        doaj_url     = str(row.get("doaj_url") or "#")

        with st.container(border=True):
            h1, h2, h3 = st.columns([7, 1.5, 1.5])
            with h1:
                st.markdown(f"**#{rank} &nbsp; [{row['title']}]({doaj_url})**")
                st.caption(
                    f"{row.get('publisher','—')} &nbsp;·&nbsp; "
                    f"{row.get('country','—')} &nbsp;·&nbsp; "
                    f"{row.get('license','—')}")
            with h2:
                st.metric("Match", badge)
            with h3:
                st.metric("Quartile", quartile_str)

            bar_col, detail_col = st.columns([3, 2])

            with bar_col:
                scores = {
                    "Overall score": float(row["consultation_score"]),
                    "NLP relevance": float(row["nlp_score"]),
                    "Prestige (B4)": float(row.get("b4_score", 0) or 0),
                    "Speed (weeks)": float(row.get("weeks_score", 0) or 0),
                    "Cost (APC)":    float(row.get("b5_score", 0) or 0),
                }
                colors = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#1abc9c"]
                fig = go.Figure()
                for (label, val), color in zip(reversed(list(scores.items())), reversed(colors)):
                    fig.add_trace(go.Bar(
                        x=[val], y=[label], orientation="h", marker_color=color,
                        text=[f"{val:.3f}"], textposition="inside",
                        insidetextanchor="start", showlegend=False, width=0.6))
                fig.update_layout(
                    xaxis=dict(range=[0, 1], showticklabels=False, showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False),
                    margin=dict(l=0, r=0, t=0, b=0), height=175,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            with detail_col:
                d1, d2 = st.columns(2)
                d1.metric("APC", apc_str)
                d2.metric("Pub. time", f"{weeks} wks" if weeks else "—")
                flags = []
                if row.get("is_core"):      flags.append("✅ Scopus/WoS")
                if row.get("has_waiver"):   flags.append("💰 Waiver")
                if row.get("has_doi"):      flags.append("🔗 DOI")
                if row.get("plagiarism_check"): flags.append("🛡 Plagiarism check")
                if flags: st.markdown("  \n".join(flags))
                st.link_button("Open in DOAJ ↗", doaj_url, use_container_width=True)

            # Access links
            access_links = []
            def _safe_url(key):
                v = row.get(key)
                return str(v) if v and not pd.isna(v) else None
            if _safe_url("url"):          access_links.append(("🌐 Journal website",     _safe_url("url")))
            if _safe_url("authors_url"):  access_links.append(("📝 Submit / Author guide", _safe_url("authors_url")))
            if _safe_url("aims_url"):     access_links.append(("🎯 Aims & scope",         _safe_url("aims_url")))
            if _safe_url("apc_url"):      access_links.append(("💵 APC details",          _safe_url("apc_url")))
            if _safe_url("waiver_url"):   access_links.append(("💰 Waiver info",           _safe_url("waiver_url")))
            if access_links:
                lcols = st.columns(len(access_links))
                for col, (label, href) in zip(lcols, access_links):
                    col.link_button(label, href, use_container_width=True)

            # AI explanation
            exp = explanations[idx] if explanations and idx < len(explanations) else ""
            if exp:
                st.info(f"💡 **Why this journal?** {exp}")

    # Export
    st.divider()
    st.subheader("Export results")
    export_cols = ["title", "publisher", "country", "sjr_quartile", "apc_usd",
                   "weeks_to_pub", "license", "consultation_score", "nlp_score",
                   "confidence", "is_core", "has_waiver", "doaj_url"]
    export_df = top10[[c for c in export_cols if c in top10.columns]].copy()
    export_df.insert(0, "rank", range(1, len(export_df) + 1))
    export_df["query_title"] = title_input
    export_df["query_area"]  = area_input or ""

    rows_html = ""
    for i, (_, row) in enumerate(export_df.iterrows()):
        apc_val  = float(row.get("apc_usd") or 0)
        exp_text = explanations[i] if explanations and i < len(explanations) else ""
        exp_cell = f"<br><em style='color:#555;font-size:11px'>💡 {exp_text}</em>" if exp_text else ""
        rows_html += f"""<tr>
          <td>{int(row['rank'])}</td>
          <td><a href="{row.get('doaj_url','#')}">{row['title']}</a>{exp_cell}</td>
          <td>{row.get('publisher','—')}</td><td>{row.get('country','—')}</td>
          <td>{row.get('sjr_quartile') or 'Unranked'}</td>
          <td>{"Free" if apc_val == 0 else f"${apc_val:,.0f}"}</td>
          <td>{int(row.get('weeks_to_pub') or 0)}</td>
          <td>{float(row.get('consultation_score',0)):.3f}</td>
          <td>{CONFIDENCE_BADGE.get(str(row.get('confidence','')),'—')}</td></tr>"""

    ai_note   = "<p><em>AI explanations powered by Claude (Anthropic)</em></p>" if any(explanations) else ""
    html_report = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>AI Powered Journal Recommender</title><style>
body{{font-family:Arial,sans-serif;font-size:13px;margin:30px}}
h1{{color:#2c3e50}}table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ddd;padding:8px;text-align:left;vertical-align:top}}
th{{background:#2c3e50;color:white}}tr:nth-child(even){{background:#f9f9f9}}
a{{color:#2980b9}}.meta{{color:#666;margin-bottom:20px}}</style></head><body>
<h1>AI Powered Journal Recommender</h1>
<div class="meta"><b>Article:</b> {title_input}<br>
<b>Area:</b> {area_input or '—'}<br>
<b>Priorities:</b> Speed {speed_w:.0%} · Prestige {prestige_w:.0%} · Cost {cost_w:.0%}</div>
{ai_note}<table><thead><tr>
<th>#</th><th>Journal</th><th>Publisher</th><th>Country</th>
<th>Quartile</th><th>APC</th><th>Weeks</th><th>Score</th><th>Match</th>
</tr></thead><tbody>{rows_html}</tbody></table></body></html>"""

    ex1, ex2 = st.columns(2)
    with ex1:
        st.download_button("Download CSV",
                           data=export_df.to_csv(index=False).encode("utf-8"),
                           file_name="journal_recommendations.csv", mime="text/csv",
                           use_container_width=True)
    with ex2:
        st.download_button("Download HTML report",
                           data=html_report.encode("utf-8"),
                           file_name="journal_recommendations.html", mime="text/html",
                           use_container_width=True,
                           help="Open in browser → File → Print → Save as PDF")
