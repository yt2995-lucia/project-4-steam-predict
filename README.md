# Steam Game Success Prediction

Final project for STAT GR5243 — Applied Data Science, Spring 2026.

**Live app:** https://project-4-steam-predict-c7kek4uqopudy52zodmsmu.streamlit.app

We predict whether a game on Steam will reach "Very Positive" reception (positive review ratio ≥ 80%) using only information available before launch — price, genre, tags, supported languages, release timing, and latent structure pulled out of game descriptions and tag co-occurrence. The intended use case is sequel and IP-extension decisions: cases where the studio knows the next title will find an audience, but doesn't yet know whether that audience will be happy with it.

## Data

| Source | What we pull |
|---|---|
| Steam Web API | App list and metadata — price, release date, languages, genres, categories, tags |
| SteamSpy | Owner estimates, peak concurrent users, playtime |
| Steam store pages | Long descriptions, screenshots, system requirements (web scraped) |
| Steam community reviews | User reviews used for descriptive EDA |

After cleaning, the modeling sample is 995 games — drawn from Steam's top ~1,000 titles by ownership — each labeled successful or not.

## Pipeline

The repo splits into three stages, each runnable on its own.

**Collection and cleaning.** `collect_data.py` pulls raw API and SteamSpy data and scrapes store pages. `clean_data.py` resolves missing values, normalizes types, and writes `data/interim/cleaned.csv`.

**Unsupervised structure discovery.** `GroupB/GroupB.ipynb` runs K-Means (6 clusters), UMAP (2D projection), and LDA (10 topics over the descriptions). Output goes into `data/processed/unsupervised_output.csv` and is fed back as input features for the supervised model — that's how the unsupervised work earns its keep beyond just visualization.

**Supervised modeling.** `src/build_features.py` produces a 92-feature matrix that combines raw metadata, multi-hot encoded list fields, LDA topic loadings, and the unsupervised cluster IDs and UMAP coordinates. `src/model_fitting.py` then tunes Logistic Regression, Random Forest, and XGBoost with 5-fold stratified CV. `src/Evaluation.ipynb` evaluates on a held-out test split.

One choice worth calling out: anything observable only *after* a game ships — total reviews, owner counts, peak CCU, metacritic score — was deliberately excluded from the feature matrix. Those signals correlate 0.6 to 0.8 with the target and would push validation ROC-AUC into the 0.95+ range, but they would also make the model useless at deployment time, where pre-release info is all anyone has.

## Results

| Model | Validation ROC-AUC | F1 | Recall |
|---|---|---|---|
| Logistic Regression | 0.811 | 0.865 | 0.897 |
| Random Forest | 0.817 | 0.873 | 0.938 |
| **XGBoost (final)** | **0.847** | **0.879** | **0.945** |

XGBoost won: highest ROC-AUC and F1, and recall above 0.94 — useful here because missing a future-successful game is the costly error in a sequel-funding context. Shallow trees (depth 2, 150 estimators) keep inference fast and SHAP-interpretable.

## Repository layout

```
.
├── app.py                          Streamlit dashboard
├── requirements.txt                pinned deps for Streamlit Cloud
├── runtime.txt                     python-3.12
├── data/
│   ├── raw/                        API responses + scraped pages
│   ├── interim/cleaned.csv
│   └── processed/                  features.csv, target.csv, unsupervised_output.csv
├── src/
│   ├── collect_data.py             API pulls + scraping
│   ├── clean_data.py               cleaning pipeline
│   ├── groupA_descriptive_eda.py
│   ├── groupB_unsupervised.ipynb   unsupervised structure (K-Means, UMAP, LDA)
│   ├── build_features.py
│   ├── model_fitting.py
│   └── Evaluation.ipynb
├── models/                         joblib-saved trained pipelines
├── outputs/modeling/               CV results, run metadata
└── figures/groupA, groupB/         generated figures
```

## Running locally

```
git clone https://github.com/yt2995-lucia/project-4-steam-predict.git
cd project-4-steam-predict
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The cached feature matrix and trained `.joblib` models are committed, so the app launches without re-running the upstream pipeline.

On macOS, XGBoost additionally needs OpenMP. If you see a `libxgboost.dylib could not be loaded` error on first run, install it once:

```
brew install libomp
```

To reproduce the pipeline from scratch (all commands run from the project root):

```
python src/collect_data.py
python src/clean_data.py
# then open src/groupB_unsupervised.ipynb and run all cells
python src/build_features.py
python src/model_fitting.py
# evaluation lives in src/Evaluation.ipynb
```

The collection step is rate-limited by the Steam API and takes a while; the rest run in a few minutes.

## Team

| Member | Contribution |
|---|---|
| Yumeng Xu | Slides Design and Production, Final Report Production, Website UI Design |
| Yueyou Tao | Data Collection and Cleaning, App Design and Coding, Project Framework Construction|
| Guicheng Zheng | Partial EDA, Model fitting|
| Runji Gao | Partial EDA, unsupervised factor design, model selection and evaluation |

## License

Coursework only — not licensed for redistribution.
