"""
build_features.py — Feature Engineering pipeline (Step 3 of project)

输入:
    data/interim/cleaned.csv               ← Step 1 (clean_data.py) 的输出
    data/processed/unsupervised_output.csv ← Step 2 (Group B) 修好的输出

输出 (三个文件分开, 便于建模时分别管理 X / y / 展示信息):
    data/processed/features.csv  ← X, 模型输入矩阵, 只含 pre-release 特征
    data/processed/target.csv    ← y, 只有 appid + is_successful
    data/processed/meta.csv      ← appid + name + release_year (给 SHAP / app 显示用)

核心设计原则 — 防止 data leakage:
    只用游戏发售前 / 设计阶段就能确定的特征。
    严格排除任何 review / owners / playtime / recommendations 等 "用户行为发生后" 的字段。

特征类别:
    数值直传      : ~10 列 (price, lang, achievements, screenshots, ...)
    布尔 0/1      : 4 列  (is_free + 3 个 platform)
    日期数值化    : 4 列  (year, month, quarter, dow)
    Genre 多热    : top-12 列 (官方分类, pre-release)
    Category 多热 : top-15 列 (Single-player / Multi-player / Steam Cloud / ...)
    Tag 多热      : top-30 列 (社区 tag, 严格说是 post-release, 但反映设计风格)
    Cluster 独热  : ~5 列 (KMeans cluster, drop_first 防共线)
    UMAP 坐标    : 2 列  (从 store_user_tags 算出)
    LDA topic    : 9 列  (10 个 topic - 1 = 9, 因为概率和=1)

数据 leakage 风险标注 (在论文 / presentation 里要诚实声明):
    - store_user_tags / cluster_id / umap_x / umap_y 都是基于玩家社区 tag 算的, 严格说是
      post-release. 但这些 tag 反映了游戏的设计风格 (玩法 / 题材 / 画风), 从理论上
      "一个老练的开发者在设计阶段就能预判自己游戏会被打成什么 tag", 因此我们把它们
      当作 pre-release 信号. 在 limitations 章节会说明这一点.
    - LDA topic 是基于 short_description 算的, 是真正的 pre-release 文本, 没有 leakage.
    - metacritic_score 是发售后才有的 → 完全排除.

运行:
    python src/build_features.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEANED_CSV = PROJECT_ROOT / "data" / "interim" / "cleaned.csv"
UNSUP_CSV = PROJECT_ROOT / "data" / "processed" / "unsupervised_output.csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"


# ============================================================
# 配置开关 (实验时可改)
# ============================================================
TOP_N_GENRES = 12
TOP_N_CATEGORIES = 15
TOP_N_TAGS = 30
N_TOPICS = 10  # GroupB 设的 LDA topic 数

# 是否包含潜在 leakage 风险的特征 (论文要说明)
INCLUDE_USER_TAGS = True   # 社区 tag 的 multi-hot
INCLUDE_CLUSTER = True     # KMeans cluster id (基于 tag)
INCLUDE_UMAP = True        # UMAP 坐标 (基于 tag)
INCLUDE_TOPICS = True      # LDA topic (基于 description, 严格 pre-release)


# ============================================================
# 工具函数
# ============================================================
def safe_json_loads(x, default):
    """cleaned.csv 里 list/dict 列存的是 JSON 字符串, 反序列化用."""
    if pd.isna(x):
        return default
    try:
        return json.loads(x)
    except (json.JSONDecodeError, TypeError):
        return default


def to_int_bool(x):
    """把 'True'/'False' 字符串 (或 Python bool) 转成 1/0."""
    if pd.isna(x):
        return 0
    s = str(x).strip().lower()
    if s in ("true", "1", "yes"):
        return 1
    return 0


def slugify(name: str) -> str:
    """把 genre / category / tag 名字转成合法的列名后缀."""
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
    """对一列 list-of-strings, 统计 value 出现次数, 返回 top n 的 list."""
    counter: dict[str, int] = {}
    for items in series:
        if not isinstance(items, list):
            continue
        for item in items:
            counter[item] = counter.get(item, 0) + 1
    return [k for k, _ in sorted(counter.items(), key=lambda kv: kv[1], reverse=True)[:n]]


def multi_hot(series: pd.Series, top_values: list[str], prefix: str) -> pd.DataFrame:
    """把一列 list-of-strings 展开成 multi-hot dataframe."""
    cols: dict[str, list[int]] = {f"{prefix}_{slugify(v)}": [] for v in top_values}
    for items in series:
        items_set = set(items) if isinstance(items, list) else set()
        for v in top_values:
            cols[f"{prefix}_{slugify(v)}"].append(1 if v in items_set else 0)
    return pd.DataFrame(cols)


# ============================================================
# 1. 读数据
# ============================================================
def load_data():
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)
    cleaned = pd.read_csv(CLEANED_CSV)
    unsup = pd.read_csv(UNSUP_CSV)
    print(f"  cleaned.csv             : {cleaned.shape}")
    print(f"  unsupervised_output.csv : {unsup.shape}")

    # 反序列化 list 列
    cleaned["genres_list"] = cleaned["genres"].apply(lambda x: safe_json_loads(x, []))
    cleaned["categories_list"] = cleaned["categories"].apply(lambda x: safe_json_loads(x, []))
    cleaned["tags_list"] = cleaned["store_user_tags"].apply(lambda x: safe_json_loads(x, []))

    return cleaned, unsup


# ============================================================
# 2. 数值 / 布尔 / 日期特征
# ============================================================
def build_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Step 2: Building simple (numeric/bool/date) features ---")

    out = pd.DataFrame()
    out["appid"] = df["appid"]

    # Numeric pre-release 特征
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

    # 日期 (release_year/month/quarter/dow 已在 cleaned.csv 里拆好)
    date_cols = ["release_year", "release_month", "release_quarter", "release_dow"]
    for col in date_cols:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    print(f"  date     : {len([c for c in date_cols if c in df.columns])} cols")

    print(f"  -> simple features so far: {out.shape[1] - 1}  (+1 appid)")
    return out


# ============================================================
# 3. List 列展开 (genre / category / tag)
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
# 4. 加入 unsupervised 输出 (cluster / umap / topic)
# ============================================================
def add_unsupervised_features(df: pd.DataFrame, unsup: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Step 4: Adding unsupervised features ---")

    df = df.merge(unsup, on="appid", how="left")
    n_missing = df["cluster_id"].isna().sum() if "cluster_id" in df.columns else 0
    if n_missing > 0:
        print(f"  WARN: {n_missing} games missing from unsup (will use defaults)")

    # Cluster one-hot (drop_first 防止线性模型共线)
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

    # UMAP 坐标
    if INCLUDE_UMAP and "umap_x" in df.columns:
        df["umap_x"] = pd.to_numeric(df["umap_x"], errors="coerce").fillna(0)
        df["umap_y"] = pd.to_numeric(df["umap_y"], errors="coerce").fillna(0)
        print("  umap coords: 2 cols")
    else:
        df = df.drop(columns=["umap_x", "umap_y"], errors="ignore")
        print("  umap coords: SKIPPED")

    # LDA topics (drop 1 个避免概率和=1 的共线)
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
# 5. Drop 中间列, 输出干净的 features dataframe
# ============================================================
def cleanup_features(features: pd.DataFrame) -> pd.DataFrame:
    """删除任何不该进 X 的列 (只保留 appid + 数值特征)."""
    drop_keywords = (
        # post-release 字段, 万一前面忘了排
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
# 6. Target / Meta 切出来 + 对齐
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

    # 对齐 target 顺序
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
