# Steam Game Success Prediction

**STAT GR5243 — Applied Data Science, Spring 2026 Final Project**

Predicting the success of Steam games from pre-release metadata, store content, and early signals using an end-to-end ML pipeline (data collection → cleaning → EDA + unsupervised → feature engineering → supervised modeling → interpretation → interactive dashboard).

---

## Motivation

Thousands of games are released on Steam every year, but the success rate is heavily skewed — a small fraction capture the vast majority of players and revenue. This project asks: **given only information available at (or shortly after) release, can we predict whether a game will succeed?** We also examine which factors most drive success, surfacing actionable insights for developers and publishers.

## Research Questions

1. **Main (classification):** Given a game's pre-release features, predict whether it will become a "successful" game (positive_review_ratio ≥ 80% AND total_reviews ≥ 500).
2. **Secondary (regression, optional):** Predict review score and estimated owners.
3. **Unsupervised exploration:** Are there latent clusters of games beyond official genres?

## Project Pipeline

```
Steam API ──┐
SteamSpy ───┼── Raw tables ── Cleaning ── EDA + Unsupervised ── Features ── Models ── Dashboard
Scraping ───┘                                                                         (Streamlit)
```

## Repository Layout

```
steam-game-success/
├── README.md                   # this file
├── requirements.txt            # Python dependencies
├── .gitignore
├── data/
│   ├── raw/                    # raw API pulls / scraped HTML (gitignored)
│   ├── interim/                # partially processed data
│   └── processed/              # final modeling-ready tables
├── notebooks/                  # analysis notebooks
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_interpretation.ipynb
├── src/                        # reusable source code
│   ├── data/                   # API + scraping
│   ├── features/               # feature engineering
│   ├── models/                 # train / evaluate
│   └── visualization/          # plotting utilities
├── dashboard/
│   └── app.py                  # Streamlit app
├── reports/
│   ├── final_report.md         # written report (deliverable 1)
│   └── figures/                # saved figures for report
└── tests/                      # unit tests
```

## Setup

```bash
# 1. Clone
git clone https://github.com/<your-org>/steam-game-success.git
cd steam-game-success

# 2. Create venv (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install
pip install -r requirements.txt
```

## How to Reproduce

```bash
# Step 1 — collect raw data
python -m src.data.steam_api          # pulls app list + details
python -m src.data.steamspy_api       # pulls player-count estimates
python -m src.data.scraper            # scrapes store pages + reviews

# Step 2 — build features
python -m src.features.build_features

# Step 3 — train & evaluate models
python -m src.models.train
python -m src.models.evaluate

# Step 4 — launch dashboard
streamlit run dashboard/app.py
```

Alternatively, walk through the notebooks `01_*.ipynb` → `05_*.ipynb` in order.

## Data Sources

| Source | Type | Contents |
|--------|------|----------|
| [Steam Web API](https://steamcommunity.com/dev) | Official API | App list, metadata (price, release date, developer, tags, etc.) |
| [SteamSpy API](https://steamspy.com/api.php) | Third-party API | Owner estimates, playtime, peak concurrent users |
| Steam Store pages | Web scraping | Long descriptions, system requirements, media counts |
| Steam community reviews | Web scraping | User reviews (for sentiment and topic analysis) |

## Methods Overview

- **Unsupervised:** K-Means / DBSCAN on tag vectors, UMAP for visualization, LDA topic modeling on descriptions.
- **Supervised:** Logistic Regression (baseline), Random Forest, XGBoost, and a stacked ensemble.
- **Validation:** Time-based train/test split + stratified 5-fold CV on the training portion.
- **Metrics:** ROC-AUC (primary), F1, Precision, Recall, PR-AUC, confusion matrices.
- **Interpretability:** SHAP values, partial dependence plots, case studies.

## Team & Contributions

| Member | Role | Key Contributions |
|--------|------|-------------------|
| TBD    | Data Engineer | API clients, scrapers, data storage |
| TBD    | EDA & Viz Lead | EDA, unsupervised analysis, dashboard front-end |
| TBD    | Feature Engineer | Feature construction, preprocessing pipeline |
| TBD    | Modeling Lead | Model training, tuning, interpretability |
| TBD    | PM / Writer | Report, slides, README, repo hygiene |

*Fill in names before submission — grading rubric requires explicit individual contributions.*

## Deliverables

- [x] Project proposal / roadmap
- [ ] Monday (4/28) in-class progress presentation
- [ ] Raw + cleaned datasets
- [ ] Final written report (`reports/final_report.md`)
- [ ] Fully reproducible code (this repo)
- [ ] Optional: Streamlit dashboard (bonus +10pt)
- [ ] Final presentation

## License

For coursework use only.
