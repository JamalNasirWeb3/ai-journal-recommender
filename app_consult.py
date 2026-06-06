"""Sprint 6/7: Consultation mode — personalised journal ranking for a specific article.

When the FastAPI backend (Sprint 7) is running at localhost:8000, the app logs in via
the API and uses the /match endpoint (with rate-limiting and history tracking).
Falls back to direct NLP calls when the API is unavailable.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import nlp as nlp_engine
from api_client import APIClient, APIError
import llm as llm_engine

PARQUET = Path("journals_scored.parquet")

CONFIDENCE_BADGE = {
    "High":   "🟢 High",
    "Medium": "🟡 Medium",
    "Low":    "🔴 Low",
}

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    for col in ("is_predatory", "is_core", "has_doi", "has_waiver", "plagiarism_check"):
        df[col] = df[col].fillna(False).astype(bool)
    return df


@st.cache_resource
def load_nlp_model():
    """Load sentence-transformer model once per session."""
    return nlp_engine.load_model()


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Powered Journal Recommender",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔬 AI Powered Journal Recommender")
st.caption(
    "Enter your article details and get a personalised top-10 open-access journal ranking — Consultation Mode"
)

# ---------------------------------------------------------------------------
# Sidebar — priority sliders + options
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Sidebar — API login (Sprint 7 integration)
# ---------------------------------------------------------------------------

_client = APIClient()
_api_online = _client.health()

st.sidebar.header("Account")
if _api_online:
    if "api_token" not in st.session_state:
        st.session_state.api_token = None
        st.session_state.api_user = None

    if st.session_state.api_token is None:
        with st.sidebar.expander("Log in", expanded=True):
            _email = st.text_input("Email", key="login_email")
            _pwd   = st.text_input("Password", type="password", key="login_pwd")
            _col1, _col2 = st.columns(2)
            if _col1.button("Login", use_container_width=True):
                try:
                    st.session_state.api_token = _client.login(_email, _pwd)
                    st.session_state.api_user  = _client.me(st.session_state.api_token)
                    st.rerun()
                except APIError as e:
                    st.error(e.detail)
    else:
        _user = st.session_state.api_user or {}
        st.sidebar.success(f"Logged in as **{_user.get('email','')}**  \n"
                           f"Role: `{_user.get('role','')}`")
        if st.sidebar.button("Logout"):
            st.session_state.api_token = None
            st.session_state.api_user  = None
            st.rerun()
else:
    st.sidebar.caption("API offline — using direct NLP mode")

st.sidebar.divider()
st.sidebar.header("Your priorities")
st.sidebar.caption("Drag sliders to reflect what matters most. Values are auto-normalised.")

speed_raw    = st.sidebar.slider("⚡ Speed (fast publication)",    0, 100, 33)
prestige_raw = st.sidebar.slider("🏆 Prestige (impact / indexing)", 0, 100, 34)
cost_raw     = st.sidebar.slider("💰 Cost (low APC)",              0, 100, 33)

_total = speed_raw + prestige_raw + cost_raw or 1
speed_w    = speed_raw    / _total
prestige_w = prestige_raw / _total
cost_w     = cost_raw     / _total

st.sidebar.markdown(
    f"**Effective weights:** "
    f"Speed {speed_w:.0%} · Prestige {prestige_w:.0%} · Cost {cost_w:.0%}"
)

st.sidebar.divider()
st.sidebar.subheader("Options")
top_k_candidates = st.sidebar.select_slider(
    "NLP candidate pool", options=[50, 100, 200, 500], value=200,
    help="Larger pool = more diverse re-ranking. Slower for 500.",
)
excl_predatory = st.sidebar.checkbox("Exclude predatory journals", value=True)
only_core = st.sidebar.checkbox("Scopus / WoS only (is_core)")

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------

with st.form("query_form"):
    col_title, col_area = st.columns([3, 2])
    with col_title:
        title_input = st.text_input(
            "Article title *",
            placeholder="e.g. Deep learning for early cancer detection in CT scans",
        )
    with col_area:
        area_input = st.text_input(
            "Topic area",
            placeholder="e.g. machine learning, oncology",
        )
    abstract_input = st.text_area(
        "Abstract (optional — improves matching)",
        placeholder="Paste your abstract here…",
        height=130,
    )
    submitted = st.form_submit_button("Find matching journals", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Detect input level
# ---------------------------------------------------------------------------

if submitted:
    if not title_input.strip():
        st.error("Please enter an article title.")
        st.stop()

    level = 1
    if area_input.strip():
        level = 2
    if abstract_input.strip():
        level = 3

    _level_labels = {1: "title only", 2: "title + area", 3: "title + area + abstract"}
    st.info(f"Input level: **{_level_labels[level]}** — using NLP level {level} matching")

    token = st.session_state.get("api_token")
    api_response = None

    # ------------------------------------------------------------------
    # Path A: API mode (logged in + API online)
    # ------------------------------------------------------------------
    if token and _api_online:
        with st.spinner("Matching via API…"):
            try:
                api_response = _client.match(
                    token=token,
                    title=title_input.strip(),
                    area=area_input.strip() or None,
                    abstract=abstract_input.strip() or None,
                    top_k=10,
                    speed=speed_raw,
                    prestige=prestige_raw,
                    cost=cost_raw,
                )
            except APIError as e:
                if e.status_code == 429:
                    st.error(e.detail)
                    st.stop()
                st.warning(f"API error ({e.status_code}): {e.detail} — falling back to direct mode")
                api_response = None

    if api_response:
        # Show usage counter
        used = api_response.get("queries_used", 0)
        remaining = api_response.get("queries_remaining")
        if remaining is not None:
            st.sidebar.info(f"Queries this month: **{used} / 20**")
        else:
            st.sidebar.info(f"Queries this month: **{used}** (unlimited)")

        # Convert API response to DataFrame for unified card rendering
        top10 = pd.DataFrame(api_response["results"])
        top10["consultation_score"] = top10["publication_score"]

    # ------------------------------------------------------------------
    # Path B: Direct mode (no API / not logged in)
    # ------------------------------------------------------------------
    else:
        df = load_data()
        with st.spinner("Loading NLP model…"):
            load_nlp_model()

        pre_df = df[~df["is_predatory"]].copy() if excl_predatory else df.copy()
        if only_core:
            pre_df = pre_df[pre_df["is_core"]]

        with st.spinner(f"Running semantic matching across {len(pre_df):,} journals…"):
            try:
                nlp_results = nlp_engine.match(
                    title=title_input.strip(),
                    area=area_input.strip() or None,
                    abstract=abstract_input.strip() or None,
                    df=pre_df,
                    top_k=top_k_candidates,
                )
            except FileNotFoundError as exc:
                st.error(str(exc))
                st.stop()

        if len(nlp_results) == 0:
            st.warning("No results found. Try relaxing the filters.")
            st.stop()

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

    if len(top10) == 0:
        st.warning("No results found. Try relaxing the filters.")
        st.stop()

    # ------------------------------------------------------------------
    # AI explanations — one sentence per journal from Claude
    # ------------------------------------------------------------------
    _explain_cache_key = f"llm_exp_{hash((title_input, area_input or '', (abstract_input or '')[:100]))}"

    if llm_engine.is_available():
        if _explain_cache_key not in st.session_state:
            journals_for_llm = [
                {
                    "title":         str(row.get("title") or ""),
                    "subjects":      str(row.get("subjects") or row.get("cluster_label") or ""),
                    "sjr_quartile":  row.get("sjr_quartile"),
                    "apc_usd":       float(row.get("apc_usd") or 0),
                    "is_core":       bool(row.get("is_core") or False),
                    "cluster_label": row.get("cluster_label"),
                }
                for _, row in top10.iterrows()
            ]
            with st.spinner("Generating AI explanations..."):
                st.session_state[_explain_cache_key] = llm_engine.explain_recommendations(
                    title=title_input.strip(),
                    area=area_input.strip() or None,
                    abstract=abstract_input.strip() or None,
                    journals=journals_for_llm,
                )
        explanations = st.session_state[_explain_cache_key]
    else:
        explanations = [""] * len(top10)

    # ------------------------------------------------------------------
    # Results header
    # ------------------------------------------------------------------
    st.divider()
    st.subheader(f"Top {len(top10)} journals for: *{title_input[:80]}*")
    st.caption(
        f"Formula — B2 NLP 40% · B3 Editorial 25% · "
        f"Speed {speed_w*35:.0f}% · Prestige {prestige_w*35:.0f}% · Cost {cost_w*35:.0f}% "
        f"(speed+prestige+cost share the remaining 35%)"
    )

    # ------------------------------------------------------------------
    # Result cards
    # ------------------------------------------------------------------
    for idx, row in top10.iterrows():
        rank = idx + 1
        badge = CONFIDENCE_BADGE.get(str(row.get("confidence", "")), "⚪ —")
        quartile = row.get("sjr_quartile")
        quartile_str = str(quartile) if pd.notna(quartile) else "Unranked"
        apc = float(row.get("apc_usd", 0) or 0)
        apc_str = "Free ($0)" if apc == 0 else f"${apc:,.0f}"
        weeks = int(row.get("weeks_to_pub") or 0)
        license_str = str(row.get("license") or "—")
        doaj_url = str(row.get("doaj_url") or "#")

        with st.container(border=True):
            # --- Header row ---
            h1, h2, h3 = st.columns([7, 1.5, 1.5])
            with h1:
                st.markdown(f"**#{rank} &nbsp; [{row['title']}]({doaj_url})**")
                st.caption(
                    f"{row.get('publisher', '—')} &nbsp;·&nbsp; "
                    f"{row.get('country', '—')} &nbsp;·&nbsp; "
                    f"{license_str}"
                )
            with h2:
                st.metric("Match", badge)
            with h3:
                st.metric("Quartile", quartile_str)

            # --- Score bars + detail ---
            bar_col, detail_col = st.columns([3, 2])

            with bar_col:
                scores = {
                    "Overall score":  float(row["consultation_score"]),
                    "NLP relevance":  float(row["nlp_score"]),
                    "Prestige (B4)":  float(row.get("b4_score", 0) or 0),
                    "Speed (weeks)":  float(row.get("weeks_score", 0) or 0),
                    "Cost (APC)":     float(row.get("b5_score", 0) or 0),
                }
                colors = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#1abc9c"]
                fig = go.Figure()
                for (label, val), color in zip(reversed(list(scores.items())), reversed(colors)):
                    fig.add_trace(go.Bar(
                        x=[val], y=[label],
                        orientation="h",
                        marker_color=color,
                        text=[f"{val:.3f}"],
                        textposition="inside",
                        insidetextanchor="start",
                        showlegend=False,
                        width=0.6,
                    ))
                fig.update_layout(
                    xaxis=dict(range=[0, 1], showticklabels=False, showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False),
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=175,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            with detail_col:
                d1, d2 = st.columns(2)
                d1.metric("APC", apc_str)
                d2.metric("Pub. time", f"{weeks} wks" if weeks else "—")

                flags = []
                if row.get("is_core"):
                    flags.append("✅ Scopus/WoS")
                if row.get("has_waiver"):
                    flags.append("💰 Waiver available")
                if row.get("has_doi"):
                    flags.append("🔗 DOI")
                if row.get("plagiarism_check"):
                    flags.append("🛡 Plagiarism check")
                if flags:
                    st.markdown("  \n".join(flags))

                st.link_button("Open in DOAJ ↗", doaj_url, use_container_width=True)

            # --- Access links row ---
            access_links = []
            if row.get("url") and not pd.isna(row.get("url", float("nan"))):
                access_links.append(("🌐 Journal website", str(row["url"])))
            if row.get("authors_url") and not pd.isna(row.get("authors_url", float("nan"))):
                access_links.append(("📝 Submit / Author guide", str(row["authors_url"])))
            if row.get("aims_url") and not pd.isna(row.get("aims_url", float("nan"))):
                access_links.append(("🎯 Aims & scope", str(row["aims_url"])))
            if row.get("apc_url") and not pd.isna(row.get("apc_url", float("nan"))):
                access_links.append(("💵 APC details", str(row["apc_url"])))
            if row.get("waiver_url") and not pd.isna(row.get("waiver_url", float("nan"))):
                access_links.append(("💰 Waiver info", str(row["waiver_url"])))

            if access_links:
                link_cols = st.columns(len(access_links))
                for col, (label, href) in zip(link_cols, access_links):
                    col.link_button(label, href, use_container_width=True)

            # AI explanation — full width, below score bars and detail columns
            explanation = explanations[idx] if explanations and idx < len(explanations) else ""
            if explanation:
                st.info(f"💡 **Why this journal?** {explanation}")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Export results")

    export_cols = [
        "title", "publisher", "country", "sjr_quartile",
        "apc_usd", "weeks_to_pub", "license",
        "consultation_score", "nlp_score", "confidence",
        "is_core", "has_waiver", "doaj_url",
    ]
    export_df = top10[[c for c in export_cols if c in top10.columns]].copy()
    export_df.insert(0, "rank", range(1, len(export_df) + 1))
    export_df["query_title"] = title_input
    export_df["query_area"]  = area_input or ""

    ex1, ex2 = st.columns(2)

    with ex1:
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="journal_recommendations.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with ex2:
        # Generate a styled HTML report for print-to-PDF
        rows_html = ""
        for i, (_, row) in enumerate(export_df.iterrows()):
            apc_val = float(row.get("apc_usd") or 0)
            exp_text = explanations[i] if explanations and i < len(explanations) else ""
            exp_cell = f"<br><em style='color:#555;font-size:11px'>💡 {exp_text}</em>" if exp_text else ""
            rows_html += f"""
            <tr>
              <td>{int(row['rank'])}</td>
              <td><a href="{row.get('doaj_url','#')}">{row['title']}</a>{exp_cell}</td>
              <td>{row.get('publisher','—')}</td>
              <td>{row.get('country','—')}</td>
              <td>{row.get('sjr_quartile') or 'Unranked'}</td>
              <td>{"Free" if apc_val == 0 else f"${apc_val:,.0f}"}</td>
              <td>{int(row.get('weeks_to_pub') or 0)}</td>
              <td>{float(row.get('consultation_score',0)):.3f}</td>
              <td>{CONFIDENCE_BADGE.get(str(row.get('confidence','')),'—')}</td>
            </tr>"""

        ai_note = "<p><em>AI explanations powered by Claude (Anthropic)</em></p>" if any(explanations) else ""

        html_report = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>AI Powered Journal Recommender — Results</title>
<style>
  body {{ font-family: Arial, sans-serif; font-size: 13px; margin: 30px; }}
  h1 {{ color: #2c3e50; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
  th {{ background-color: #2c3e50; color: white; }}
  tr:nth-child(even) {{ background-color: #f9f9f9; }}
  a {{ color: #2980b9; }}
  .meta {{ color: #666; margin-bottom: 20px; }}
</style>
</head>
<body>
<h1>AI Powered Journal Recommender</h1>
<div class="meta">
  <b>Article:</b> {title_input}<br>
  <b>Area:</b> {area_input or '—'}<br>
  <b>Priorities:</b> Speed {speed_w:.0%} · Prestige {prestige_w:.0%} · Cost {cost_w:.0%}
</div>
{ai_note}
<table>
  <thead>
    <tr>
      <th>#</th><th>Journal</th><th>Publisher</th><th>Country</th>
      <th>Quartile</th><th>APC</th><th>Weeks</th><th>Score</th><th>Match</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
</body>
</html>"""

        st.download_button(
            "Download HTML report",
            data=html_report.encode("utf-8"),
            file_name="journal_recommendations.html",
            mime="text/html",
            use_container_width=True,
            help="Open in browser and use File → Print → Save as PDF",
        )
