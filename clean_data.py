"""
clean_data.py — 把 collect_data.py 抓下来的 5 个 raw 文件合并清洗成一张干净表

输入 (来自 data/raw/):
    - steam_details.jsonl        Steam 官方 API 详情 (price, release_date, developers, genres...)
    - steamspy.csv               SteamSpy 玩家统计 (positive/negative 评论数, owners 区间, tags)
    - scraped/store_pages.jsonl  商店页面爬的 (description, user_tags, n_screenshots)
    - scraped/reviews.jsonl      用户评论原文

输出 (到 data/interim/):
    - cleaned.parquet            每行一款游戏, 所有字段清洗好
    - cleaned_preview.csv        前 50 行的 csv, 肉眼检查用

运行:
    pip install pandas pyarrow
    python clean_data.py

这一步只做"清洗", 不做 feature engineering (TF-IDF / tag 聚类之类):
    - 把 price 从 cents -> dollars
    - 把 release_date 字符串 -> datetime
    - 把嵌套的 genres/categories/platforms 拍平成 list
    - 解析 SteamSpy 的 owners 区间 "2,000,000 .. 5,000,000" -> min/max
    - 解析 SteamSpy 的 tags 字符串 -> dict
    - 打 target 标签 is_successful
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pandas as pd


# ============================================================
# 路径
# ============================================================
RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")

STEAM_DETAILS = RAW_DIR / "steam_details.jsonl"
STEAMSPY_CSV = RAW_DIR / "steamspy.csv"
STORE_PAGES = RAW_DIR / "scraped" / "store_pages.jsonl"
REVIEWS = RAW_DIR / "scraped" / "reviews.jsonl"


# ============================================================
# 工具函数
# ============================================================
def read_jsonl(path: Path) -> list[dict]:
    """读 jsonl, 每行一个 JSON 对象。"""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def strip_html(text: str | None) -> str:
    """粗暴把 <html> 标签去掉, 留纯文本。"""
    if not text:
        return ""
    # 去 tag
    text = re.sub(r"<[^>]+>", " ", text)
    # 压缩空白
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_owners_range(s: str | None) -> tuple[int | None, int | None]:
    """
    "2,000,000 .. 5,000,000" -> (2000000, 5000000)
    """
    if not isinstance(s, str):
        return (None, None)
    m = re.findall(r"[\d,]+", s)
    if len(m) >= 2:
        try:
            lo = int(m[0].replace(",", ""))
            hi = int(m[1].replace(",", ""))
            return (lo, hi)
        except ValueError:
            return (None, None)
    return (None, None)


def parse_tag_dict(s: str | None) -> dict[str, int]:
    """
    SteamSpy 的 tags 列在 csv 里是 python dict 的字符串形式, 比如
    "{'Adventure': 1883, 'Indie': 912, ...}"
    用 ast.literal_eval 安全地解析。
    """
    if not isinstance(s, str) or not s.strip():
        return {}
    try:
        d = ast.literal_eval(s)
        if isinstance(d, dict):
            return {str(k): int(v) for k, v in d.items()}
    except (ValueError, SyntaxError):
        pass
    return {}


def parse_release_date(raw: dict | None) -> pd.Timestamp | None:
    """Steam API 返回的 release_date 是 {'coming_soon': bool, 'date': 'Feb 9, 2016'}。"""
    if not isinstance(raw, dict):
        return None
    date_str = raw.get("date", "")
    if not date_str:
        return None
    try:
        return pd.to_datetime(date_str, errors="coerce")
    except Exception:
        return None


def count_supported_languages(lang_str: str | None) -> int:
    """
    Steam API 的 supported_languages 是 HTML 字符串, 语言用逗号分隔。
    '<strong>*</strong>' 之类的标记要去掉。
    """
    if not isinstance(lang_str, str):
        return 0
    clean = strip_html(lang_str)
    # 去掉星号 + 注释
    clean = re.sub(r"\*", "", clean)
    clean = re.sub(r"languages with full audio support", "", clean, flags=re.I)
    parts = [p.strip() for p in clean.split(",") if p.strip()]
    return len(parts)


# ============================================================
# Step 1: 读 Steam 官方 API 详情 (steam_details.jsonl)
# ============================================================
def load_steam_details() -> pd.DataFrame:
    print(f"[1/5] Loading {STEAM_DETAILS}...")
    rows = read_jsonl(STEAM_DETAILS)
    records = []
    for r in rows:
        appid = r.get("appid") or r.get("steam_appid")
        if appid is None:
            continue

        # 价格: price_overview.initial 是 cents. is_free=True 就是 0.
        price_overview = r.get("price_overview") or {}
        if r.get("is_free"):
            price_usd = 0.0
        else:
            initial_cents = price_overview.get("initial")
            price_usd = (initial_cents / 100.0) if isinstance(initial_cents, (int, float)) else None

        # 平台
        platforms = r.get("platforms") or {}

        # genres / categories
        genres = [g.get("description", "") for g in (r.get("genres") or []) if isinstance(g, dict)]
        categories = [c.get("description", "") for c in (r.get("categories") or []) if isinstance(c, dict)]

        # metacritic
        metacritic = (r.get("metacritic") or {}).get("score")

        # 成就
        achievements = (r.get("achievements") or {}).get("total") or 0

        # 媒体数量
        n_screenshots = len(r.get("screenshots") or [])
        n_movies = len(r.get("movies") or [])

        # 发售日
        release_date = parse_release_date(r.get("release_date"))
        coming_soon = (r.get("release_date") or {}).get("coming_soon", False)

        # DLC 数量
        n_dlc = len(r.get("dlc") or [])

        # 开发商 / 发行商
        developers = r.get("developers") or []
        publishers = r.get("publishers") or []

        records.append({
            "appid": int(appid),
            "name": r.get("name"),
            "type": r.get("type"),
            "is_free": bool(r.get("is_free", False)),
            "price_usd": price_usd,
            "required_age": r.get("required_age", 0),
            "n_supported_languages": count_supported_languages(r.get("supported_languages")),
            "supports_windows": bool(platforms.get("windows", False)),
            "supports_mac": bool(platforms.get("mac", False)),
            "supports_linux": bool(platforms.get("linux", False)),
            "genres": genres,
            "categories": categories,
            "n_achievements": achievements,
            "n_screenshots_api": n_screenshots,
            "n_movies": n_movies,
            "n_dlc": n_dlc,
            "metacritic_score": metacritic,
            "release_date": release_date,
            "coming_soon": bool(coming_soon),
            "developers": developers,
            "publishers": publishers,
            "short_description": r.get("short_description"),
            "detailed_description_raw": r.get("detailed_description"),  # 留 raw, 之后可能 re-clean
            "recommendations_total": (r.get("recommendations") or {}).get("total"),
        })

    df = pd.DataFrame(records)
    print(f"       -> {len(df):,} rows")
    return df


# ============================================================
# Step 2: 读 SteamSpy (steamspy.csv)
# ============================================================
def load_steamspy() -> pd.DataFrame:
    print(f"[2/5] Loading {STEAMSPY_CSV}...")
    df = pd.read_csv(STEAMSPY_CSV)

    # owners 区间拆成两列
    owners_parsed = df["owners"].apply(parse_owners_range)
    df["owners_min"] = owners_parsed.apply(lambda t: t[0])
    df["owners_max"] = owners_parsed.apply(lambda t: t[1])
    df["owners_mid"] = (df["owners_min"].fillna(0) + df["owners_max"].fillna(0)) / 2

    # tag 字典
    df["tag_dict"] = df["tags"].apply(parse_tag_dict)
    df["n_tags"] = df["tag_dict"].apply(len)

    # price 也是 cents, 转成 dollar
    df["steamspy_price_usd"] = pd.to_numeric(df["price"], errors="coerce") / 100.0
    df["steamspy_initialprice_usd"] = pd.to_numeric(df["initialprice"], errors="coerce") / 100.0

    # 只留我们真需要的列, 其他 drop
    keep_cols = [
        "appid",
        "developer", "publisher",
        "positive", "negative",
        "userscore",
        "owners_min", "owners_max", "owners_mid",
        "average_forever", "average_2weeks",
        "median_forever", "median_2weeks",
        "steamspy_price_usd", "steamspy_initialprice_usd",
        "ccu",
        "tag_dict", "n_tags",
    ]
    df = df[keep_cols].copy()
    df["appid"] = df["appid"].astype(int)

    print(f"       -> {len(df):,} rows")
    return df


# ============================================================
# Step 3: 读商店页面爬的内容 (store_pages.jsonl)
# ============================================================
def load_store_pages() -> pd.DataFrame:
    print(f"[3/5] Loading {STORE_PAGES}...")
    rows = read_jsonl(STORE_PAGES)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["appid"] = df["appid"].astype(int)

    # description 已经是纯文本 (scraper 里 get_text 过)
    df["description_len_chars"] = df["description"].fillna("").str.len()
    df["description_len_words"] = df["description"].fillna("").str.split().str.len()

    df = df.rename(columns={
        "description": "store_description",
        "user_tags": "store_user_tags",
        "n_screenshots": "n_screenshots_scraped",
    })
    print(f"       -> {len(df):,} rows")
    return df


# ============================================================
# Step 4: 读评论 (reviews.jsonl)
# ============================================================
def load_reviews() -> pd.DataFrame:
    print(f"[4/5] Loading {REVIEWS}...")
    rows = read_jsonl(REVIEWS)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["appid"] = df["appid"].astype(int)

    # 评论列表就先留着 (NLP 用), 另外存一些聚合特征
    df["n_reviews_scraped"] = df["reviews"].apply(lambda lst: len(lst) if isinstance(lst, list) else 0)
    df["avg_review_len"] = df["reviews"].apply(
        lambda lst: (sum(len(r) for r in lst) / len(lst)) if isinstance(lst, list) and lst else 0
    )
    print(f"       -> {len(df):,} rows")
    return df


# ============================================================
# Step 5: 合并 + 打 target 标签
# ============================================================
def merge_all(
    details: pd.DataFrame,
    steamspy: pd.DataFrame,
    store: pd.DataFrame,
    reviews: pd.DataFrame,
) -> pd.DataFrame:
    print("[5/5] Merging on appid...")
    df = details.merge(steamspy, on="appid", how="left", suffixes=("", "_sp"))
    if not store.empty:
        df = df.merge(store, on="appid", how="left")
    if not reviews.empty:
        df = df.merge(reviews, on="appid", how="left")

    # ---------- 清洗 ----------
    # 只留 type == 'game' (collect_data.py 应该已经过滤, 双保险)
    if "type" in df.columns:
        df = df[df["type"] == "game"].copy()

    # 去掉还没发售 / 没 release_date 的
    df = df[df["coming_soon"] != True].copy()

    # 把一些能填 0 的缺失填 0
    for col in ["n_achievements", "n_dlc", "n_screenshots_api", "n_movies",
                "n_screenshots_scraped", "n_reviews_scraped", "avg_review_len",
                "positive", "negative"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # required_age 在 Steam API 里有时是 int (0) 有时是 str ("17"), 统一转 int
    if "required_age" in df.columns:
        df["required_age"] = pd.to_numeric(df["required_age"], errors="coerce").fillna(0).astype(int)

    # metacritic_score 也可能混类型
    if "metacritic_score" in df.columns:
        df["metacritic_score"] = pd.to_numeric(df["metacritic_score"], errors="coerce")

    # recommendations_total 同理
    if "recommendations_total" in df.columns:
        df["recommendations_total"] = pd.to_numeric(df["recommendations_total"], errors="coerce")

    # userscore / ccu / average_forever 等 SteamSpy 数值列
    for col in ["userscore", "ccu", "average_forever", "average_2weeks",
                "median_forever", "median_2weeks", "owners_min", "owners_max", "owners_mid"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 价格: 如果 Steam API 没拿到, 用 SteamSpy 兜底
    df["price_usd"] = df["price_usd"].fillna(df["steamspy_price_usd"])
    df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")

    # ---------- 打 target ----------
    # 总评论数 = positive + negative
    df["total_reviews"] = df["positive"] + df["negative"]
    # 好评率 = positive / total
    df["positive_review_ratio"] = df.apply(
        lambda r: (r["positive"] / r["total_reviews"]) if r["total_reviews"] > 0 else None,
        axis=1,
    )
    # is_successful: 好评率 >= 80% 且 total_reviews >= 500
    df["is_successful"] = (
        (df["positive_review_ratio"] >= 0.80)
        & (df["total_reviews"] >= 500)
    ).astype("Int64")  # 可空整数

    # 如果 total_reviews == 0, 认为数据不足, target 置 NA
    df.loc[df["total_reviews"] == 0, "is_successful"] = pd.NA

    # ---------- 派生时间字段 ----------
    df["release_year"] = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month
    df["release_quarter"] = df["release_date"].dt.quarter
    df["release_dow"] = df["release_date"].dt.dayofweek  # Monday=0

    print(f"       -> merged: {len(df):,} rows x {df.shape[1]} cols")
    return df


# ============================================================
# 主流程
# ============================================================
def main():
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    details = load_steam_details()
    steamspy = load_steamspy()
    store = load_store_pages()
    reviews = load_reviews()

    df = merge_all(details, steamspy, store, reviews)

    # ---------- 摘要 ----------
    print("\n================ Summary ================")
    print(f"Total games:         {len(df):,}")
    if "is_successful" in df.columns:
        vc = df["is_successful"].value_counts(dropna=False)
        print(f"is_successful value counts:\n{vc.to_string()}")
    if "release_year" in df.columns:
        print(f"\nRelease year range:  {df['release_year'].min()} - {df['release_year'].max()}")
    if "price_usd" in df.columns:
        print(f"Price (USD) stats:   mean={df['price_usd'].mean():.2f}, median={df['price_usd'].median():.2f}")
    missing = df.isna().mean().sort_values(ascending=False).head(10)
    print("\nTop 10 columns by missing %:")
    print((missing * 100).round(1).astype(str) + "%")

    # ---------- 保存 ----------
    # parquet 保 list/dict 字段没问题; csv 做个 preview
    out_parquet = INTERIM_DIR / "cleaned.parquet"
    out_preview = INTERIM_DIR / "cleaned_preview.csv"

    df.to_parquet(out_parquet, index=False)
    # preview csv: 把 list/dict 字段转 str, 不然 excel 打不开
    preview = df.head(50).copy()
    for col in preview.columns:
        if preview[col].apply(lambda x: isinstance(x, (list, dict))).any():
            preview[col] = preview[col].astype(str)
    preview.to_csv(out_preview, index=False)

    print(f"\nSaved:")
    print(f"  {out_parquet}   ({len(df):,} rows)")
    print(f"  {out_preview}   (first 50 rows, for eyeballing)")
    print("\nDone! Next step: feature engineering (TF-IDF, tag clustering, etc.)")


if __name__ == "__main__":
    main()
