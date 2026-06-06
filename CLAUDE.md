# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AI Powered Journal Recommender** — an intelligent web platform that helps researchers, PhD students, and academics find the best open-access journals for publishing scientific articles. The system scores and ranks 22,890+ journals from the DOAJ (Directory of Open Access Journals) using a multi-block scoring model, NLP semantic matching, and external data sources.

**Two user-facing modes:**
- **Exploration mode** — interactive dashboard (world map, scatter plots, heatmaps, cluster filters) with no article input required
- **Consultation mode** — user inputs article title, topic area, and/or abstract; receives a personalised top-10 journal ranking with publication score, APC cost (USD), estimated publication weeks, SCImago quartile, and match confidence

## Tech Stack

- **Data pipeline**: Python — pandas, numpy, pyarrow, scikit-learn, sentence-transformers, requests, BeautifulSoup4
- **Backend (Sprint 7)**: FastAPI, SQLAlchemy, SQLite, JWT auth
- **Frontend (current)**: Streamlit + Plotly — adhoc prototype for Sprints 5–6
- **Frontend (future)**: Next.js + Tailwind CSS — replaces Streamlit in a later sprint
- **LLM**: Anthropic Claude API (`anthropic` SDK) — recommendation explanations
- **Optional / desirable**: umap-learn, KeyBERT or spaCy, ReportLab
- **External data sources**: SCImago SJR CSV, OpenAlex API, Beall's List

## Five-Layer Architecture

```
Layer 1: Data Sources      DOAJ CSV · SCImago CSV · OpenAlex API · Beall's List
Layer 2: Python ETL        ingest.py → enrich.py → score.py → nlp.py
Layer 3: Master file       journals_scored.parquet  (+ NLP artefacts)
Layer 4: REST API          FastAPI (api/)  ← JWT auth, roles, history
Layer 5: Presentation      Streamlit app.py (Sprints 5–6) → Next.js (future)
```

`api_client.py` is the thin HTTP client that `app.py` uses to route Consultation mode requests through the FastAPI backend when a user is logged in; falls back to direct NLP if the API is offline.

## Scoring Model

**Full publication score** (used in Consultation mode):

```
Publication Score = 0.40 × B2 + 0.25 × B3 + 0.20 × B4 + 0.15 × B5
```

| Block | Weight | Description | When computed |
|-------|--------|-------------|---------------|
| **B2** | 40% | Topic Compatibility — NLP thematic fit score (`nlp_score`) | Query-time (nlp.py) |
| **B3** | 25% | Editorial process — `review_score` 70% + `plagiarism_check` 20% + `has_doi` 10% | Pre-computed (score.py) |
| **B4** | 20% | Quality signals — `sjr_norm` 35% + `h_norm` 15% + `is_core` 30% + `volume_score` 20% | Pre-computed (score.py) |
| **B5** | 15% | Cost control — `apc_score` 80% + `has_waiver` 20% | Pre-computed (score.py) |

**Hard exclusion**: `is_predatory=True` → score = 0.0, excluded from rankings

**`final_score` in parquet** = `(B3×0.25 + B4×0.20 + B5×0.15) / 0.60` — static quality proxy (0–1) used in Exploration mode and pre-filtering; does not include B2 since B2 requires a user query.

**Consultation mode formula** in `app.py` / `api/routers/match.py`:
```
publication_score = nlp_score×0.40 + b3_score×0.25 + weeks_score×speed_w×0.35
                    + b4_score×prestige_w×0.35 + b5_score×cost_w×0.35
```
User priority sliders (speed / prestige / cost) redistribute the 35% adjustable pool; B2 and B3 weights are always fixed.

**NLP internal composition** (B2, inside `nlp.py`):
```
nlp_score = L1×0.30 + L2×0.10 + L3×0.60
```
- L1 (30%): TF-IDF cosine similarity over journal keywords + subjects
- L2 (10%): LCC-code overlap with user topic area (only when `area` is provided; weight redistributed to L3 otherwise)
- L3 (60%): sentence-transformer cosine similarity (`all-MiniLM-L6-v2`)

Confidence badge thresholds (based on raw L3 similarity): High ≥ 0.35, Medium ≥ 0.20, Low < 0.20.

## Data Assets

| File | Status | Description |
|------|--------|-------------|
| `doaj_journalcsv_20260507_2321_utf8.csv` | Source | DOAJ dump, 22,890 rows, 52 columns |
| `journals_clean.parquet` | **Exists** | Sprint 1 output — cleaned, normalized, APC in USD |
| `journals_enriched.parquet` | **Exists** | Sprint 2 output — adds SJR, OpenAlex, Beall's flag |
| `journals_scored.parquet` | **Exists** | Sprint 3 output — B3/B4/B5 scores + KMeans cluster labels; master file feeds all UI and API |
| `journal_embeddings.npz` | **Exists** | Sprint 4 — sentence-transformer embeddings (22890×384, float32) |
| `tfidf_vectorizer.pkl` | **Exists** | Sprint 4 — fitted TF-IDF vectorizer |
| `tfidf_matrix.npz` | **Exists** | Sprint 4 — sparse TF-IDF matrix |
| `.openalex_cache.json` | Auto-generated | OpenAlex API cache; delete to force re-fetch |

## Data Pipeline

```
DOAJ CSV
  → ingest.py          (Sprint 1 ✓) → journals_clean.parquet
  → enrich.py          (Sprint 2 ✓) → journals_enriched.parquet
  → score.py           (Sprint 3 ✓) → journals_scored.parquet  ← feeds all UI
  → nlp.py             (Sprint 4 ✓) → embeddings cache
  → app.py             (Sprint 5–6 ✓) → Streamlit unified app
  → api/               (Sprint 7 ✓) → FastAPI + JWT + SQLite
  → deploy/            (Sprint 8)   → Render/Railway + automation
```

## Sprint Reference

### Sprint 1 — `ingest.py` (delivered)

Produces `journals_clean.parquet` with all 52 DOAJ columns renamed to snake_case, APC converted to USD, and four normalized score columns: `weeks_score`, `apc_score`, `volume_score`, `review_score`.

### Sprint 2 — `enrich.py` (delivered)

Adds: `sjr_score`, `sjr_quartile`, `h_index` (SCImago), `openalex_id`, `articles_per_year`, `is_core` (OpenAlex `is_core` field — Scopus/WoS proxy), `is_predatory` (Beall's List).

**SCImago file**: download from scimagojr.com/journalrank.php, save as `scimago*.csv`. **CLI flags**: `--skip-openalex`, `--skip-bealls`.

### Sprint 3 — `score.py` (delivered)

Adds: `b3_score`, `b4_score`, `b5_score`, `final_score` (static quality proxy), `cluster` (Int8, -1=excluded), `cluster_label`. Five KMeans clusters: High Impact, Core Indexed, Rigorous Review, Low Cost, High Volume.

### Sprint 4 — `nlp.py` (delivered)

Run `python nlp.py` once to build artefacts (~5 min on CPU). Public API: `match(title, area, abstract, df, top_k)` → DataFrame with `nlp_score` (B2), `l3_sim`, `confidence`, `rank`.

### Sprint 5–6 — `app.py` (delivered, port 8501)

Unified Streamlit entry point — sidebar radio selects Exploration or Consultation mode. `app_explore.py` and `app_consult.py` are legacy single-mode files kept for reference; `app.py` is the primary entry point.

**`llm.py`** — Claude API integration (`claude-haiku-4-5-20251001`). Generates a one-sentence explanation per journal. Results cached in `st.session_state` per query hash to avoid redundant API calls. Degrades gracefully (explanations hidden) if `ANTHROPIC_API_KEY` is absent.

### Sprint 7 — `api/` (delivered, port 8000)

FastAPI backend with JWT auth, SQLite persistence, and Streamlit integration.

**File structure**: `api/main.py` (app + lifespan), `api/db.py`, `api/models.py`, `api/schemas.py`, `api/auth.py`, `api/deps.py`, `api/routers/{auth,journals,match,history}.py`, `api_client.py` (Streamlit→API client).

**Endpoints**:
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/auth/register` | — | Create account (`role`: client/internal) |
| POST | `/auth/login` | — | Returns JWT bearer token |
| GET | `/auth/me` | Required | Current user info |
| POST | `/auth/logout` | Required | Blacklist current token |
| POST | `/auth/promote/{id}` | Required (internal) | Promote user to internal role |
| POST | `/match` | Required | NLP matching + B2 scoring; saves to history |
| GET | `/explore` | — | Filtered paginated journal list |
| GET | `/journal/{issn}` | — | Single journal by ISSN |
| GET | `/history` | Required | User's past searches |
| GET | `/history/{id}/results` | Required | Full results for a past search |

**Rate limiting**: client role → 50 queries/month (`CLIENT_MONTHLY_LIMIT` in `api/routers/match.py`, HTTP 429 when exceeded); internal → unlimited. Tracked in `usage` table by `user_id + YYYY-MM`.

**Database** (`journal_platform.db`): tables `users`, `searches` (stores results_json), `usage`, `token_blacklist`.

**Interactive API docs**: http://localhost:8000/docs

### Sprint 8 — Deploy + automation (planned)

Deploy FastAPI + Streamlit to Render.com or Railway.app (free tier). Monthly automation script: auto-downloads new DOAJ CSV, re-runs pipeline, updates `journals_scored.parquet`. Full README documentation, monitoring.

### Future — Agentic AI Layer (planned)

Six agents: Research Agent, Journal Reputation Agent, Matching Agent, Risk Agent, Publication Advisor Agent, Report Generation Agent.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Optional | Enables per-journal Claude explanation cards in Consultation mode |
| `JWT_SECRET_KEY` | Production | Signs JWT tokens; ephemeral random key used if absent (tokens invalidated on restart) |
| `ADMIN_EMAIL` | Optional | Seeds an internal admin user at API startup |
| `ADMIN_PASSWORD` | With `ADMIN_EMAIL` | Password for seeded admin user |

## Non-obvious Implementation Details

**Startup prerequisites**: The FastAPI server loads `journals_scored.parquet` into `app.state.df` at startup (via `lifespan`) and calls `nlp_engine.load_model()` to pre-warm the sentence-transformer. Both the parquet and the NLP artefacts (`journal_embeddings.npz`, `tfidf_vectorizer.pkl`, `tfidf_matrix.npz`) **must exist before starting the API**. If they don't, run the full pipeline first (`ingest.py` → `enrich.py` → `score.py` → `nlp.py`).

**Path sensitivity**: `nlp.py`, `ingest.py`, `score.py`, and `app.py` all use relative `Path("...")` references and must be run from the project root. The `api/` package works around this with `sys.path.insert(0, str(Path(__file__).parent.parent))` to import `nlp` from the project root.

**Full-feature mode requires both services**: Run `uvicorn api.main:app --reload` (port 8000) and `streamlit run app.py` (port 8501) simultaneously. Streamlit falls back to direct NLP if the API is offline, but search history, auth, and monthly usage tracking are API-only.

**Guest query cap**: Unauthenticated users in Consultation mode get `GUEST_SESSION_LIMIT = 3` free queries per browser session (tracked in `st.session_state`), then are prompted to log in.

**API vs direct NLP differences**: The API `/match` endpoint always fetches 200 NLP candidates then returns `top_k` (hardcoded). The direct NLP path in `app.py` uses the user-configurable "NLP candidate pool" slider (50–500).

**JWT tokens expire in 24 hours** (`ACCESS_TOKEN_EXPIRE_HOURS = 24` in `api/auth.py`). Token revocation is implemented via a `token_blacklist` table checked on every authenticated request.

**LLM explanations are session-cached**: `llm.py` results are stored in `st.session_state` keyed by a hash of `(title, area, abstract[:100])` to avoid redundant API calls within a session.

## Commands

```bash
# Pipeline (run in order on fresh data; all commands from project root)
python ingest.py
python enrich.py                         # full; flags: --skip-openalex, --skip-bealls
python score.py
python nlp.py                            # builds embeddings (~5 min on CPU)

# Full-feature mode: run both services in separate terminals
uvicorn api.main:app --reload            # → http://localhost:8000  (docs: /docs)
streamlit run app.py                     # → http://localhost:8501
# set ANTHROPIC_API_KEY to enable per-journal Claude explanations in consultation mode

# nlp.py smoke test (runs automatically when called as __main__)
python nlp.py                            # prints top-5 results for a test query after build

# Legacy single-mode apps (kept for reference, not primary entry point)
# streamlit run app_explore.py
# streamlit run app_consult.py
```
