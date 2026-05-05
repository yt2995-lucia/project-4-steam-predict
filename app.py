"""
app.py — Streamlit dashboard for Steam Game Success Predictor
STAT GR5243 Applied Data Science · Spring 2026 Final Project

Tabs:
    1. Overview          — Project framing, dataset stats, model comparison table
    2. Predictor         — Form input → real XGBoost prediction → verdict
    3. Explore           — UMAP scatter, K-Means clusters, success rate by cluster
    4. Model Performance — Validation metrics, ROC/PR comparison, model selection rationale

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Steam Game Success Predictor",
    page_icon="🎮",
    layout="wide",
)


# ============================================================
# Constants & Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
TARGET_PATH = PROJECT_ROOT / "data" / "processed" / "target.csv"
UNSUPERVISED_PATH = PROJECT_ROOT / "data" / "processed" / "unsupervised_output.csv"
CLEANED_PATH = PROJECT_ROOT / "data" / "interim" / "cleaned.csv"

MODELING_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "modeling" / "model_fitting_summary.csv"
MODELING_METADATA_PATH = PROJECT_ROOT / "outputs" / "modeling" / "modeling_run_metadata.json"

LOGREG_MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression_best.joblib"
RF_MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_best.joblib"
XGB_MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_best.joblib"

# Decision thresholds — placed above the 73% base rate so that a "greenlight"
# verdict carries information beyond the prior.
BASE_RATE = 0.7337
GREENLIGHT_THRESHOLD = 0.80
REVIEW_THRESHOLD = 0.65

# Friendly cluster names (qualitative labels assigned during EDA).
CLUSTER_NAMES = {
    0: "Cluster 0",
    1: "Cluster 1",
    2: "Cluster 2",
    3: "Cluster 3",
    4: "Cluster 4",
    5: "Cluster 5",
}


# ============================================================
# Data & Model Loading (cached)
# ============================================================
@st.cache_data
def load_features() -> pd.DataFrame:
    return pd.read_csv(FEATURES_PATH)


@st.cache_data
def load_target() -> pd.DataFrame:
    return pd.read_csv(TARGET_PATH)


@st.cache_data
def load_unsupervised() -> pd.DataFrame:
    return pd.read_csv(UNSUPERVISED_PATH)


@st.cache_data
def load_cleaned() -> pd.DataFrame:
    if CLEANED_PATH.exists():
        return pd.read_csv(CLEANED_PATH, low_memory=False)
    return pd.DataFrame()


@st.cache_data
def load_summary() -> pd.DataFrame:
    return pd.read_csv(MODELING_SUMMARY_PATH)


@st.cache_data
def load_metadata() -> dict:
    return json.loads(MODELING_METADATA_PATH.read_text())


@st.cache_resource
def load_models() -> dict:
    return {
        "Logistic Regression": joblib.load(LOGREG_MODEL_PATH),
        "Random Forest": joblib.load(RF_MODEL_PATH),
        "XGBoost": joblib.load(XGB_MODEL_PATH),
    }


@st.cache_data
def get_feature_defaults(features: pd.DataFrame) -> dict:
    """
    Default values for every feature column (excluding appid).

    Most columns use the median; one-hot cluster columns use the mode
    (so the predictor sits in a real, in-distribution cluster by default).
    """
    cols = [c for c in features.columns if c != "appid"]
    defaults = features[cols].median().to_dict()

    cluster_cols = [c for c in cols if c.startswith("cluster_")]
    if cluster_cols:
        cluster_sums = features[cluster_cols].sum()
        modal_cluster = cluster_sums.idxmax()
        for c in cluster_cols:
            defaults[c] = 1 if c == modal_cluster else 0

    return defaults


def build_feature_vector(
    defaults: dict, feature_columns: list, overrides: dict
) -> pd.DataFrame:
    """1-row DataFrame with default values overlaid by user overrides."""
    row = {col: defaults[col] for col in feature_columns}
    for k, v in overrides.items():
        if k in row:
            row[k] = v
    return pd.DataFrame([row], columns=feature_columns)


def get_verdict(prob: float) -> tuple[str, str, str]:
    """Return (verdict_text, emoji, color)."""
    if prob >= GREENLIGHT_THRESHOLD:
        return "GREENLIGHT THE SEQUEL", "🟢", "#22C55E"
    elif prob >= REVIEW_THRESHOLD:
        return "REVIEW · NEAR BASE RATE", "🟡", "#F59E0B"
    else:
        return "RECONSIDER · BELOW AVERAGE", "🔴", "#EF4444"


# ============================================================
# Load everything once
# ============================================================
features = load_features()
target = load_target()
unsupervised = load_unsupervised()
cleaned = load_cleaned()
summary = load_summary()
metadata = load_metadata()
models = load_models()
defaults = get_feature_defaults(features)
FEATURE_COLUMNS = [c for c in features.columns if c != "appid"]


# ============================================================
# Header
# ============================================================
st.title("🎮 Will Your Steam Game Succeed?")
st.caption("STAT GR5243 — Applied Data Science · Spring 2026 Final Project")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "🔮 Predictor", "🗺️ Explore", "📈 Model Performance"]
)


# ============================================================
# Tab 1 · Overview
# ============================================================
with tab1:
    st.header("Project Overview")

    n_games = len(target)
    n_success = int(target["is_successful"].sum())
    success_rate = n_success / n_games
    n_features = len(FEATURE_COLUMNS)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Games analyzed", f"{n_games:,}")
    c2.metric("Very Positive", f"{n_success:,}")
    c3.metric("Base success rate", f"{success_rate:.1%}")
    c4.metric("Engineered features", f"{n_features}")

    st.markdown(
        """
        ### Research Question
        For games that already have a player base, can we predict — using only
        **pre-release** information — whether they will achieve **"Very Positive"**
        reception (≥80% positive review ratio)?

        ### Why this matters
        A studio whose first game has found its footing must decide whether to
        greenlight a sequel. Reception is invisible until launch, but the budget
        decision must be made beforehand. This is a **sequel / IP-extension
        decision tool** — the audience is guaranteed; reception is not.
        """
    )

    st.subheader("Model Comparison · validation set")
    summary_display = summary[
        [
            "model",
            "best_cv_auc_on_training",
            "validation_auc",
            "validation_accuracy",
            "validation_f1",
            "validation_recall",
        ]
    ].copy()
    summary_display.columns = [
        "Model",
        "CV AUC (train)",
        "Val AUC",
        "Val Accuracy",
        "Val F1",
        "Val Recall",
    ]
    for col in summary_display.columns[1:]:
        summary_display[col] = summary_display[col].apply(lambda x: f"{x:.3f}")
    st.dataframe(summary_display, use_container_width=True, hide_index=True)

    st.info(
        "**Final model: XGBoost** — best validation ROC-AUC (0.847) and F1 (0.879). "
        "AUC sits comfortably above the 0.5 baseline yet far below the 0.95+ that "
        "would indicate post-release leakage — confirming the model is genuinely "
        "predicting from pre-release signal."
    )

    st.subheader("End-to-end pipeline at a glance")
    st.markdown(
        f"""
        | Stage | Output |
        |---|---|
        | Data collection | Steam Web API + SteamSpy + scraping → ~1,000 top-owned games |
        | Cleaning | `data/interim/cleaned.csv` ({len(cleaned) if not cleaned.empty else "—"} rows) |
        | Unsupervised structure | K-Means (6 clusters) + UMAP (2D) + LDA (10 topics) |
        | Feature engineering | {n_features} features · {n_games} games · zero NaN |
        | Supervised modeling | LogReg + Random Forest + XGBoost · 5-fold CV |
        | Final selection | XGBoost (val ROC-AUC = 0.847) |
        """
    )


# ============================================================
# Tab 2 · Predictor
# ============================================================
with tab2:
    st.header("Predict Reception for a Hypothetical Game")
    st.markdown(
        "Adjust the pre-release inputs below. The trained **XGBoost** model "
        "returns the probability the game will reach Very Positive (≥80% "
        "positive review ratio)."
    )

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Pricing & Scope")
        price = st.slider("Price (USD)", 0.0, 60.0, 19.99, step=0.5)
        n_languages = st.slider("Languages supported", 1, 28, 7)
        n_achievements = st.slider("Steam achievements", 0, 200, 20)
        release_month = st.selectbox(
            "Release month",
            list(range(1, 13)),
            index=10,
            format_func=lambda x: [
                "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
            ][x - 1],
        )

        st.subheader("Platforms")
        sup_win = st.checkbox("Windows", value=True)
        sup_mac = st.checkbox("Mac", value=False)
        sup_linux = st.checkbox("Linux", value=False)

    with col_r:
        st.subheader("Genre & Type")
        genre_options = {
            "Action": "genre_action",
            "Adventure": "genre_adventure",
            "Indie": "genre_indie",
            "RPG": "genre_rpg",
            "Strategy": "genre_strategy",
            "Simulation": "genre_simulation",
            "Free To Play": "genre_free_to_play",
            "Casual": "genre_casual",
            "Massively Multiplayer": "genre_massively_multiplayer",
            "Sports": "genre_sports",
            "Early Access": "genre_early_access",
            "Racing": "genre_racing",
        }
        primary_genre = st.selectbox(
            "Primary genre", list(genre_options.keys()), index=2
        )
        secondary_genres = st.multiselect(
            "Secondary genres",
            [g for g in genre_options.keys() if g != primary_genre],
            default=[],
        )

        st.subheader("Multiplayer mode")
        play_mode = st.radio(
            "",
            ["Single-player", "Multi-player", "Co-op", "Mixed"],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
        )

        st.subheader("Notable tags")
        tag_options = {
            "Story Rich": "tag_story_rich",
            "Atmospheric": "tag_atmospheric",
            "Open World": "tag_open_world",
            "Survival": "tag_survival",
            "Sandbox": "tag_sandbox",
            "Difficult": "tag_difficult",
            "First Person": "tag_first_person",
            "Shooter": "tag_shooter",
            "Funny": "tag_funny",
            "Sci-Fi": "tag_sci_fi",
            "Fantasy": "tag_fantasy",
            "Exploration": "tag_exploration",
        }
        selected_tags = st.multiselect(
            "Pick the tags that apply", list(tag_options.keys())
        )

    # ----------------------------------------------------
    # Build override dict from form state
    # ----------------------------------------------------
    overrides: dict = {
        "price_usd": price,
        "is_free": int(price == 0),
        "n_supported_languages": n_languages,
        "n_achievements": n_achievements,
        "release_month": float(release_month),
        "release_quarter": float((release_month - 1) // 3 + 1),
        "supports_windows": int(sup_win),
        "supports_mac": int(sup_mac),
        "supports_linux": int(sup_linux),
    }

    # Reset all genre flags, set primary + secondary
    for g_col in genre_options.values():
        overrides[g_col] = 0
    overrides[genre_options[primary_genre]] = 1
    for g in secondary_genres:
        overrides[genre_options[g]] = 1

    # Multiplayer category + tag flags
    is_single = play_mode in ("Single-player", "Mixed")
    is_multi = play_mode in ("Multi-player", "Mixed")
    is_coop = play_mode == "Co-op"
    overrides["cat_single_player"] = int(is_single)
    overrides["cat_multi_player"] = int(is_multi)
    overrides["cat_co_op"] = int(is_coop)
    overrides["tag_singleplayer"] = int(is_single)
    overrides["tag_multiplayer"] = int(is_multi)
    overrides["tag_co_op"] = int(is_coop)

    # Reset selectable tag flags, set chosen
    for t_col in tag_options.values():
        overrides[t_col] = 0
    for t in selected_tags:
        overrides[tag_options[t]] = 1

    # ----------------------------------------------------
    # Predict + Display
    # ----------------------------------------------------
    x_row = build_feature_vector(defaults, FEATURE_COLUMNS, overrides)
    prob_xgb = float(models["XGBoost"].predict_proba(x_row)[0, 1])
    verdict, emoji, color = get_verdict(prob_xgb)

    st.markdown("---")

    rc1, rc2, rc3 = st.columns([1, 1, 2])
    rc1.metric("Predicted probability", f"{prob_xgb:.1%}")
    rc2.metric(
        "Vs. base rate",
        f"{BASE_RATE:.1%}",
        delta=f"{(prob_xgb - BASE_RATE) * 100:+.1f} pp",
    )
    rc3.markdown(
        f"<div style='padding:18px 0 0 0;font-size:1.4em;font-weight:700;"
        f"color:{color};'>{emoji} {verdict}</div>",
        unsafe_allow_html=True,
    )

    st.progress(min(max(prob_xgb, 0.0), 1.0))
    st.caption(
        f"Thresholds: <{REVIEW_THRESHOLD:.0%} reconsider · "
        f"{REVIEW_THRESHOLD:.0%}–{GREENLIGHT_THRESHOLD:.0%} review · "
        f"≥{GREENLIGHT_THRESHOLD:.0%} greenlight  ·  "
        f"Base rate {BASE_RATE:.0%} — the model must clear this to add signal "
        f"beyond the prior."
    )

    with st.expander("Compare predictions across all three models"):
        compare_rows = []
        for name, mdl in models.items():
            p = float(mdl.predict_proba(x_row)[0, 1])
            compare_rows.append({"Model": name, "Probability": f"{p:.1%}"})
        st.dataframe(
            pd.DataFrame(compare_rows),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Three independent models trained on the same feature matrix. "
            "Disagreement signals input proximity to a decision boundary; "
            "agreement signals robustness."
        )


# ============================================================
# Tab 3 · Explore
# ============================================================
with tab3:
    st.header("UMAP Game Landscape")
    st.markdown(
        "Each point is one game in our dataset. K-Means found **6 latent clusters** "
        "in the feature space — clusters that don't always line up with Steam's "
        "official genre labels. Color by cluster or by success status."
    )

    color_by = st.radio(
        "Color points by",
        ["Cluster", "Success status"],
        horizontal=True,
    )

    plot_df = unsupervised.merge(target, on="appid", how="left")
    if not cleaned.empty and "name" in cleaned.columns:
        plot_df = plot_df.merge(
            cleaned[["appid", "name"]].drop_duplicates("appid"),
            on="appid",
            how="left",
        )

    plot_df["cluster_name"] = (
        plot_df["cluster_id"].map(CLUSTER_NAMES).fillna("Unknown")
    )
    plot_df["success_label"] = plot_df["is_successful"].map(
        {1: "Successful (Very Positive)", 0: "Below threshold"}
    ).fillna("Unknown")

    hover_data = ["appid", "cluster_name"]
    if "name" in plot_df.columns:
        hover_data.insert(0, "name")

    if color_by == "Cluster":
        fig = px.scatter(
            plot_df,
            x="umap_x",
            y="umap_y",
            color="cluster_name",
            hover_data=hover_data,
            height=600,
            title="UMAP projection · 6 K-Means clusters",
        )
    else:
        fig = px.scatter(
            plot_df,
            x="umap_x",
            y="umap_y",
            color="success_label",
            hover_data=hover_data,
            height=600,
            color_discrete_map={
                "Successful (Very Positive)": "#22C55E",
                "Below threshold": "#EF4444",
                "Unknown": "#9CA3AF",
            },
            title="UMAP projection · success vs. below threshold",
        )
    fig.update_layout(legend_title_text="")
    fig.update_traces(marker=dict(size=6, opacity=0.78))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Success rate by cluster")
    cluster_stats = (
        plot_df.dropna(subset=["is_successful"])
        .groupby("cluster_id")
        .agg(
            n_games=("appid", "count"),
            success_rate=("is_successful", "mean"),
            cluster_name=("cluster_name", "first"),
        )
        .reset_index()
    )
    cluster_stats["success_rate"] = cluster_stats["success_rate"].apply(
        lambda x: f"{x:.1%}"
    )
    cluster_stats = cluster_stats[
        ["cluster_id", "cluster_name", "n_games", "success_rate"]
    ]
    cluster_stats.columns = ["Cluster ID", "Cluster name", "# games", "Success rate"]
    st.dataframe(cluster_stats, hide_index=True, use_container_width=True)

    st.caption(
        "Latent structure that the supervised model gets to use as features: "
        "cluster_id one-hot + UMAP coordinates flow directly into the feature "
        "matrix. This is how Group B's unsupervised findings inject themselves "
        "into Group A's supervised task."
    )


# ============================================================
# Tab 4 · Model Performance
# ============================================================
with tab4:
    st.header("Model Performance Comparison")

    st.subheader("Validation metrics — three models, five metrics")
    perf_df = summary[
        [
            "model",
            "validation_auc",
            "validation_accuracy",
            "validation_precision",
            "validation_recall",
            "validation_f1",
        ]
    ].copy()
    perf_df.columns = ["Model", "ROC-AUC", "Accuracy", "Precision", "Recall", "F1"]

    melt = perf_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(
        melt,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        title="Validation set · 3 models × 5 metrics",
        height=480,
        text=melt["Score"].apply(lambda x: f"{x:.2f}"),
    )
    fig.update_layout(yaxis=dict(range=[0, 1]))
    fig.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, use_container_width=True)

    perf_table = perf_df.copy()
    for col in perf_table.columns[1:]:
        perf_table[col] = perf_table[col].apply(lambda x: f"{x:.3f}")
    st.dataframe(perf_table, hide_index=True, use_container_width=True)

    st.subheader("Why XGBoost was selected")
    st.markdown(
        """
        - **Highest validation ROC-AUC (0.847)** and **F1 (0.879)** of the three models
        - **Recall 0.945** — the model rarely misses a successful game, which matches
          the asymmetric cost of greenlighting a sequel (false negatives are expensive)
        - AUC sits in the **0.8 range** — high enough to add signal beyond the
          73% base rate, but well below the 0.95+ that would indicate post-release
          leakage
        - Tree depth = 2 with 150 estimators → simple, fast at inference, and
          interpretable via SHAP if needed
        """
    )

    st.subheader("Pipeline metadata")
    md_table = pd.DataFrame(
        [
            {"Setting": "Random state", "Value": metadata["random_state"]},
            {
                "Setting": "Train / Val / Test split",
                "Value": (
                    f"{metadata['train_size']:.0%} / "
                    f"{metadata['validation_size']:.0%} / "
                    f"{metadata['test_size']:.0%}"
                ),
            },
            {
                "Setting": "CV folds (StratifiedKFold)",
                "Value": metadata["n_splits_cv"],
            },
            {"Setting": "Scoring metric", "Value": metadata["scoring_metric"]},
            {"Setting": "Feature matrix shape", "Value": str(metadata["X_shape"])},
            {
                "Setting": "Train target ratio",
                "Value": (
                    f"{metadata['split_target_ratios']['train']['1']:.1%} success"
                ),
            },
        ]
    )
    st.dataframe(md_table, hide_index=True, use_container_width=True)

    st.info(
        "**Excluded as data leakage**: total_reviews, owners, CCU, recommendations, "
        "metacritic_score. These features correlate 0.6–0.8 with the target but are "
        "post-release outcomes. Including them would inflate ROC-AUC into the 0.95+ "
        "range while making the model useless at deployment time, where pre-release "
        "is the only information available."
    )


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "Final project for **STAT GR5243 Applied Data Science** (Spring 2026). "
        "End-to-end pipeline: data collection → cleaning → unsupervised structure "
        "discovery → feature engineering → supervised classification."
    )

    st.markdown("### Research Question")
    st.markdown(
        "*For games that already have a player base, can we predict — using only "
        "pre-release information — whether they will achieve Very Positive "
        "reception?*"
    )

    st.markdown("### Final Model")
    st.markdown(
        "**XGBoost**\n\n"
        "Validation ROC-AUC = 0.847\n\nValidation F1 = 0.879"
    )

    st.markdown("### Team")
    st.markdown(
        "- Yumeng Xu\n"
        "- Yueyou Tao\n"
        "- Guicheng Zheng\n"
        "- Runji Gao"
    )

    st.markdown("---")
    st.caption(
        "Data: Steam Web API + SteamSpy + scraping · "
        f"{len(target)} games · {len(FEATURE_COLUMNS)} engineered features"
    )
