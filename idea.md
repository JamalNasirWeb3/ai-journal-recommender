AgenticAI powered  research platform

Project Overview: We are building an intelligent web platform that helps researchers, PhD students, and academics find the best open-access journals to publish their scientific articles. The system scores and ranks 22,890+ journals from the DOAJ directory using a multi-block scoring model, NLP semantic matching, and external data sources (SCImago SJR, OpenAlex API, Beall's List). 
The platform has two modes: Exploration mode: interactive dashboard with world map, scatter plots, heatmaps and cluster filters — no article needed.
Consultation mode: the user inputs a title, topic area and/or abstract and receives a personalized ranking of the top 10 most compatible journals with scores, APC cost in USD, estimated publication weeks, SCImago quartile, and confidence level of the match. 

This is multi phase/ sprints project. Currently we are going to develop initial two phases
Sprint 1 — Data ingestion (DELIVERED): The base script ingest.py. It loads the DOAJ CSV (22,890 rows, 52 columns), cleans all columns, normalizes scores to 0-1 scale, parses APC amounts from multiple currencies to USD, and outputs journals_clean.parquet. 

 Sprint 2 — External enrichment: Cross-reference journals by ISSN with SCImago SJR CSV (quartile, H-index, SJR score), OpenAlex API (articles/year, Scopus/WoS indexing), and Beall's List (predatory journal flag). Output: journals_enriched.parquet

Tech Stack


Python/FastAPI

 pandas, numpy, pyarrow, scikit-learn, sentence-transformers, Streamlit, Plotly, requests, BeautifulSoup4, SQLite, SQLAlchemy, JWT auth, Git Desirable (not blocking): umap-learn, KeyBERT or spaCy

Nextjs, tailwindcss (futuristic)
## more sprints

Sprint 3 — Scoring engine + clusters: Calculate weighted score: B3 (editorial process) 25% + B4 (quality signals) 20% + B5 (cost control) 15%. Apply hard exclusion filter for predatory journals. Run 5 KMeans clusters (APC, rigor, topic, geography, speed). Output: journals_scored.parquet — the master file that feeds everything else. 

Sprint 4 — NLP matching engine: Build adaptive input detector (title only / title + area / title + area + abstract). Implement 3-level NLP strategy: keyword extraction with KeyBERT or spaCy for level 1, LCC code mapping for level 2, sentence-transformers embeddings + cosine similarity for level 3. Pre-calculate and cache all journal scope embeddings. Model: all-MiniLM-L6-v2. 

Sprint 5 — Exploration mode (Streamlit): Build interactive dashboard with sidebar filters (topic area, country, quartile, APC tier, weeks, language, toggles), world bubble map with Plotly, APC vs Score scatter plot, area x country heatmap, filterable table with CSV export. 


Sprint 6 — Consultation mode (Streamlit): Adaptive form with title required and area + abstract optional. Priority sliders for speed / prestige / cost with weights that sum to 100%. Top-10 results with score breakdown bars, APC in USD, weeks, quartile, CC license, confidence badge (Low / Medium / High), direct DOAJ link, PDF and CSV export.

Sprint 7 — REST API + auth + roles: FastAPI backend with JWT authentication. Endpoints: POST /match, GET /journal by ISSN, GET /explore, POST /auth/login, GET /history. SQLite database with tables for users, searches and usage tracking. Two roles: client (20 queries/month limit) and internal (unlimited). Connect Streamlit to the API

Sprint 8 — Deploy + automation + docs: Deploy FastAPI and Streamlit to Render.com or Railway.app (free tier). 

Monthly pipeline automation script that auto-downloads the new DOAJ CSV and regenerates the scored parquet. Full README documentation.


