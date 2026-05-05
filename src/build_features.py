"""
build_features.py — Feature engineering pipeline (Step 3 of the project).

Inputs:
    data/interim/cleaned.csv               output of clean_data.py (Step 1)
    data/processed/unsupervised_output.csv output of GroupB unsupervised (Step 2)

Outputs (kept as three files so X / y / display info stay separate):
    data/processed/features.csv  X — model input matrix, pre-release features only
    data/processed/target.csv    y — appid + is_successful
    data/processed/meta.csv      appid + name + release_year (for SHAP / app display)

Core design rule — leakage prevention:
    Only features that are knowable before launch (or at design time) are
    included. Anything observable only after release — reviews, owner counts,
    playtime, recommendations, metacritic — is excluded.

Feature groups:
    Numeric (~10)        : price, language count, achievements, screenshots, ...
    Boolean (4)          : is_free + 3 platform flags
    Date-derived (4)     : year, month, quarter, day-of-week
    Genre multi-hot (12) : top-12 official genres
    Category multi-hot (15) : single-player / multi-player / Steam Cloud / ...
    Tag multi-hot (30)   : top-30 community-defined tags
    Cluster one-hot      : K-Means cluster id, drop_first to avoid collinearity
    UMAP coords (2)      : 2D projection from the user-tag space
    LDA topics (9)       : 10 topics minus 1 (the simplex sums to 1)

Leakage caveats to disclose in the report:
    - Community user_tags / cluster_id / umap coordinates are derived from
      tags assigned by players post-release. We treat them as pre-release
      proxies on the argument that an experienced developer can anticipate
      which tags their game will attract at design time. This is a soft
      assumption and is called out in the limitations section.
    - LDA topics come from short_description (genuinely pre-release text);
      no leakage there.
    - metacritic_score is post-release and is excluded entirely.

Usage:
    python src/build_features.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEANED_CSV = PROJECT_ROOT / "data" / "interim" / "cleaned.csv"
UNSUP_CSV = PROJECT_ROOT / "data" / "processed" / "unsupervised_output.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"


# ============================================================
# Toggles (handy for ablation experiments)
# ============================================================
TOP_N_GENRES = 12
TOP_N_CATEGORIES = 15
TOP_N_TAGS = 30
N_TOPICS = 10  # number of LDA topics chosen by Group B

# Whether to include features with potential leakage risk (documented in report)
INCLUDE_USER_TAGS = True   # community-tag multi-hot
INCLUDE_CLUSTER = True     # K-Means cluster id (tag-based)
INCLUDE_UMAP = True        # UMAP coordinates (tag-based)
INCLUDE_TOPICS = True      # LDA topics (description-based, strictly pre-release)


# ============================================================
# Helpers
# ============================================================
def safe_json_loads(x, default):
    """cleaned.csv stores list/dict columns as JSON strings — decode them safely."""
    if pd.isna(x):
        return default
    try:
        return json.loads(x)
    except (json.JSONDecodeError, TypeError):
        return default


def to_int_bool(x):
    """Coerce 'True' / 'False' strings (or Python bools) into 1 / 0."""
    if pd.isna(x):
        return 0
    s = str(x).strip().lower()
    if s in ("true", "1", "yes"):
        return 1
    return 0


def slugify(name: str) -> str:
    """Turn a genre / category / tag name into a column-name-safe slug."""
    return (
        str(name)
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(".", "")
        .replace(",", "")
    )


def get_top_n_from_lists(series: pd.Series, n: int) -> list[str]:
    """Given a Series of list-of-strings, return the n most frequent values."""
    counter: dict[str, int] = {}
    for items in series:
        if not isinstance(items, list):
            continue
        for item in items:
            counter[item] = counter.get(item, 0) + 1
    return [k for k, _ in sorted(counter.items(), key=lambda kv: kv[1], reverse=True)[:n]]


def multi_hot(series: pd.Series, top_values: list[str], prefix: str) -> pd.DataFrame:
    """Expand a Series of list-of-strings into a multi-hot DataFrame."""
    cols: dict[str, list[int]] = {f"{prefix}_{slugify(v)}": [] for v in top_values}
    for items in series:
        items_set = set(items) if isinstance(items, list) else set()
        for v in top_values:
            cols[f"{prefix}_{slugify(v)}"].append(1 if v in items_set else 0)
    return pd.DataFrame(cols)


# ============================================================
# 1. Load data
# ============================================================
def load_data():
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)
    cleaned = pd.read_csv(CLEANED_CSV)
    unsup = pd.read_csv(UNSUP_CSV)
    print(f"  cleaned.csv             : {cleaned.shape}")
    print(f"  unsupervised_output.csv : {unsup.shape}")

    # Decode list columns
    cleaned["genres_list"] = cleaned["genres"].apply(lambda x: safe_json_loads(x, []))
    cleaned["categories_list"] = cleaned["categories"].apply(lambda x: safe_json_loads(x, []))
    cleaned["tags_list"] = cleaned["store_user_tags"].apply(lambda x: safe_json_loads(x, []))

    return cleaned, unsup


# ============================================================
# 2. Numeric / boolean / date features
# ============================================================
def build_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Step 2: Building simple (numeric/bool/date) features ---")

    out = pd.DataFrame()
    out["appid"] = df["appid"]

    # Numeric pre-release features
    numeric_cols = [
        "price_usd",
        "required_age",
        "n_supported_languages",
        "n_achievements",
        "n_screenshots_api",
        "n_movies",
        "n_dlc",
        "description_len_chars",
        "description_len_words",
        "n_tags",
    ]
    for col in numeric_cols:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    print(f"  numeric  : {len([c for c in numeric_cols if c in df.columns])} cols")

    # Boolean 0/1
    bool_cols = ["is_free", "supports_windows", "supports_mac", "supports_linux"]
    for col in bool_cols:
        if col in df.columns:
            out[col] = df[col].apply(to_int_bool)
    print(f"  boolean  : {len([c for c in bool_cols if c in df.columns])} cols")

    # Date features (release_year/month/quarter/dow already split in cleaned.csv)
    date_cols = ["release_year", "release_month", "release_quarter", "release_dow"]
    for col in date_cols:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    print(f"  date     : {len([c for c in date_cols if c in df.columns])} cols")

    print(f"  -> simple features so far: {out.shape[1] - 1}  (+1 appid)")
    return out


# ============================================================
# 3. Expand list columns (genre / category / tag)
# ============================================================
def build_list_features(df: pd.DataFrame, simple_df: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Step 3: Building multi-hot features from list columns ---")

    top_genres = get_top_n_from_lists(df["genres_list"], TOP_N_GENRES)
    genre_df = multi_hot(df["genres_list"], top_genres, prefix="genre")
    print(f"  genres     : top-{TOP_N_GENRES} -> {genre_df.shape[1]} cols  {top_genres}")

    top_cats = get_top_n_from_lists(df["categories_list"], TOP_N_CATEGORIES)
    cat_df = multi_hot(df["categories_list"], top_cats, prefix="cat")
    print(f"  categories : top-{TOP_N_CATEGORIES} -> {cat_df.shape[1]} cols  {top_cats[:5]}...")

    if INCLUDE_USER_TAGS:
        top_tags = get_top_n_from_lists(df["tags_list"], TOP_N_TAGS)
        tag_df = multi_hot(df["tags_list"], top_tags, prefix="tag")
        print(f"  user tags  : top-{TOP_N_TAGS} -> {tag_df.shape[1]} cols  {top_tags[:5]}...")
    else:
        tag_df = pd.DataFrame()
        print("  user tags  : SKIPPED (INCLUDE_USER_TAGS=False)")

    return pd.concat(
        [
            simple_df.reset_index(drop=True),
            genre_df.reset_index(drop=True),
            cat_df.reset_index(drop=True),
            tag_df.reset_index(drop=True),
        ],
        axis=1,
    )


# ============================================================
# 4. Add unsupervised features (cluster / umap / topic)
# ============================================================
def add_unsupervised_features(df: pd.DataFrame, unsup: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Step 4: Adding unsupervised features ---")

    df = df.merge(unsup, on="appid", how="left")
    n_missing = df["cluster_id"].isna().sum() if "cluster_id" in df.columns else 0
    if n_missing > 0:
        print(f"  WARN: {n_missing} games missing from unsup (will use defaults)")

    # Cluster one-hot (drop_first to avoid collinearity in linear models)
    if INCLUDE_CLUSTER and "cluster_id" in df.columns:
        cluster_dummies = pd.get_dummies(
            df["cluster_id"].fillna(-1).astype(int),
            prefix="cluster",
            drop_first=True,
        ).astype(int)
        df = df.drop(columns=["cluster_id"])
        df = pd.concat(
            [df.reset_index(drop=True), cluster_dummies.reset_index(drop=True)], axis=1
        )
        print(f"  cluster (one-hot): {cluster_dummies.shape[1]} cols")
    else:
        df = df.drop(columns=["cluster_id"], errors="ignore")
        print("  cluster: SKIPPED")

    # UMAP coordinates
    if INCLUDE_UMAP and "umap_x" in df.columns:
        df["umap_x"] = pd.to_numeric(df["umap_x"], errors="coerce").fillna(0)
        df["umap_y"] = pd.to_numeric(df["umap_y"], errors="coerce").fillna(0)
        print("  umap coords: 2 cols")
    else:
        df = df.drop(columns=["umap_x", "umap_y"], errors="ignore")
        print("  umap coords: SKIPPED")

    # LDA topics (drop one to avoid the simplex-sum-to-1 collinearity)
    topic_cols = [f"topic_{i}" for i in range(N_TOPICS)]
    if INCLUDE_TOPICS:
        for c in topic_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(1.0 / N_TOPICS)
        df = df.drop(columns=[f"topic_{N_TOPICS - 1}"], errors="ignore")
        kept_topics = [f"topic_{i}" for i in range(N_TOPICS - 1)]
        print(f"  topics: {len(kept_topics)} cols (dropped topic_{N_TOPICS - 1})")
    else:
        df = df.drop(columns=topic_cols, errors="ignore")
        print("  topics: SKIPPED")

    return df


# ============================================================
# 5. Drop intermediate columns; produce a clean features DataFrame
# ============================================================
def cleanup_features(features: pd.DataFrame) -> pd.DataFrame:
    """Remove anything that should not enter X (keep appid + numeric features only)."""
    drop_keywords = (
        # post-release fields, in case any survived earlier filtering
        "positive",
        "negative",
        "userscore",
        "owners",
        "average",
        "median",
        "ccu",
        "recommendations",
        "metacritic",
        "review",
        "is_successful",
    )
    drop_cols = [
        c for c in features.columns if any(k in c.lower() for k in drop_keywords)
    ]
    if drop_cols:
        print(f"\n--- Step 5: Dropping {len(drop_cols)} post-release / target columns ---")
        for c in drop_cols:
            print(f"    - {c}")
        features = features.drop(columns=drop_cols)
    return features


# ============================================================
# 6. Split out target / meta + align rows
# ============================================================
def split_target_and_meta(features: pd.DataFrame, df_full: pd.DataFrame):
    print("\n--- Step 6: Splitting target / meta + dropping NaN target rows ---")

    # Target
    target = df_full[["appid", "is_successful"]].copy()
    target["is_successful"] = pd.to_numeric(target["is_successful"], errors="coerce")

    valid_appids = set(target.dropna(subset=["is_successful"])["appid"])

    n_before = len(features)
    features = features[features["appid"].isin(valid_appids)].reset_index(drop=True)
    n_after = len(features)
    print(f"  rows: {n_before} -> {n_after}  (dropped {n_before - n_after} with NaN target)")

    # Align target row order
    target = target[target["appid"].isin(valid_appids)]
    target = target.set_index("appid").loc[features["appid"]].reset_index()
    target["is_successful"] = target["is_successful"].astype(int)

    # Meta
    meta_cols = ["appid", "name", "release_year"]
    meta = df_full[[c for c in meta_cols if c in df_full.columns]].copy()
    meta = meta.set_index("appid").loc[features["appid"]].reset_index()

    return features, target, meta


# ============================================================
# 7. Main
# ============================================================
def main():
    print("=" * 60)
    print("build_features.py")
    print("=" * 60)

    cleaned, unsup = load_data()

    simple = build_simple_features(cleaned)
    with_lists = build_list_features(cleaned, simple)
    full = add_unsupervised_features(with_lists, unsup)
    full = cleanup_features(full)

    features, target, meta = split_target_and_meta(full, cleaned)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feat_path = OUT_DIR / "features.csv"
    targ_path = OUT_DIR / "target.csv"
    meta_path = OUT_DIR / "meta.csv"

    features.to_csv(feat_path, index=False)
    target.to_csv(targ_path, index=False)
    meta.to_csv(meta_path, index=False)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"  features.csv : {features.shape}  -> {feat_path}")
    print(f"  target.csv   : {target.shape}    -> {targ_path}")
    print(f"  meta.csv     : {meta.shape}      -> {meta_path}")
    print(f"\n  is_successful=1 rate: {target['is_successful'].mean():.4f}")
    print(f"  total positive: {(target['is_successful'] == 1).sum()}")
    print(f"  total negative: {(target['is_successful'] == 0).sum()}")


if __name__ == "__main__":
    main()
