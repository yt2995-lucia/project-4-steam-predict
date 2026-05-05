"""
clean_data.py — Merge the 5 raw files produced by collect_data.py into a
single tidy table.

Inputs (from data/raw/):
    - steam_details.jsonl        Official Steam API details
                                 (price, release_date, developers, genres, ...)
    - steamspy.csv               SteamSpy stats (positive/negative counts,
                                 owner ranges, tags)
    - scraped/store_pages.jsonl  Scraped store pages (description, user tags,
                                 screenshot count)
    - scraped/reviews.jsonl      Raw user reviews

Output (to data/interim/):
    - cleaned.csv                One row per game, all columns cleaned (~999 rows)

Note: list / dict columns (genres, categories, developers, publishers, tag_dict)
are stored as JSON strings in the CSV. To read them back downstream:

    import json
    df = pd.read_csv("data/interim/cleaned.csv")
    for col in ["genres", "categories", "developers", "publishers", "tag_dict"]:
        df[col] = df[col].fillna("[]").apply(json.loads)

Usage:
    pip install pandas
    python src/clean_data.py

This step only does *cleaning*, no feature engineering (no TF-IDF, no tag
clustering). The transforms are:
    - Convert price from cents to dollars
    - Convert release_date strings to datetime
    - Flatten nested genres / categories / platforms into lists
    - Parse SteamSpy owner ranges "2,000,000 .. 5,000,000" -> (min, max)
    - Parse SteamSpy tag strings -> dict
    - Compute the target label `is_successful`
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================
RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")

STEAM_DETAILS = RAW_DIR / "steam_details.jsonl"
STEAMSPY_CSV = RAW_DIR / "steamspy.csv"
STORE_PAGES = RAW_DIR / "scraped" / "store_pages.jsonl"
REVIEWS = RAW_DIR / "scraped" / "reviews.jsonl"


# ============================================================
# Helpers
# ============================================================
def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file (one JSON object per line)."""
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
    """Strip HTML tags and collapse whitespace down to plain text."""
    if not text:
        return ""
    # Drop tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace
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
    SteamSpy stores its tag column as a Python-dict string in the CSV, e.g.
        "{'Adventure': 1883, 'Indie': 912, ...}"
    Use ast.literal_eval to parse it safely.
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
    """Steam API returns release_date as {'coming_soon': bool, 'date': 'Feb 9, 2016'}."""
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
    Steam API's supported_languages is an HTML string with comma-separated
    language names. Strip out markup like '<strong>*</strong>' and footnotes
    before splitting.
    """
    if not isinstance(lang_str, str):
        return 0
    clean = strip_html(lang_str)
    # Drop asterisks + footnote text
    clean = re.sub(r"\*", "", clean)
    clean = re.sub(r"languages with full audio support", "", clean, flags=re.I)
    parts = [p.strip() for p in clean.split(",") if p.strip()]
    return len(parts)


# ============================================================
# Step 1: Load Steam official-API details (steam_details.jsonl)
# ============================================================
def load_steam_details() -> pd.DataFrame:
    print(f"[1/5] Loading {STEAM_DETAILS}...")
    rows = read_jsonl(STEAM_DETAILS)
    records = []
    for r in rows:
        appid = r.get("appid") or r.get("steam_appid")
        if appid is None:
            continue

        # Price: price_overview.initial is in cents. is_free=True implies 0.
        price_overview = r.get("price_overview") or {}
        if r.get("is_free"):
            price_usd = 0.0
        else:
            initial_cents = price_overview.get("initial")
            price_usd = (initial_cents / 100.0) if isinstance(initial_cents, (int, float)) else None

        # Platforms
        platforms = r.get("platforms") or {}

        # genres / categories
        genres = [g.get("description", "") for g in (r.get("genres") or []) if isinstance(g, dict)]
        categories = [c.get("description", "") for c in (r.get("categories") or []) if isinstance(c, dict)]

        # Metacritic
        metacritic = (r.get("metacritic") or {}).get("score")

        # Achievements
        achievements = (r.get("achievements") or {}).get("total") or 0

        # Media counts
        n_screenshots = len(r.get("screenshots") or [])
        n_movies = len(r.get("movies") or [])

        # Release date
        release_date = parse_release_date(r.get("release_date"))
        coming_soon = (r.get("release_date") or {}).get("coming_soon", False)

        # DLC count
        n_dlc = len(r.get("dlc") or [])

        # Developer / publisher
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
            "detailed_description_raw": r.get("detailed_description"),  # keep raw for possible re-cleaning
            "recommendations_total": (r.get("recommendations") or {}).get("total"),
        })

    df = pd.DataFrame(records)
    print(f"       -> {len(df):,} rows")
    return df


# ============================================================
# Step 2: Load SteamSpy stats (steamspy.csv)
# ============================================================
def load_steamspy() -> pd.DataFrame:
    print(f"[2/5] Loading {STEAMSPY_CSV}...")
    df = pd.read_csv(STEAMSPY_CSV)

    # Split owner range into two columns + a midpoint
    owners_parsed = df["owners"].apply(parse_owners_range)
    df["owners_min"] = owners_parsed.apply(lambda t: t[0])
    df["owners_max"] = owners_parsed.apply(lambda t: t[1])
    df["owners_mid"] = (df["owners_min"].fillna(0) + df["owners_max"].fillna(0)) / 2

    # Tag dictionary
    df["tag_dict"] = df["tags"].apply(parse_tag_dict)
    df["n_tags"] = df["tag_dict"].apply(len)

    # Prices are also in cents — convert to dollars
    df["steamspy_price_usd"] = pd.to_numeric(df["price"], errors="coerce") / 100.0
    df["steamspy_initialprice_usd"] = pd.to_numeric(df["initialprice"], errors="coerce") / 100.0

    # Keep only the columns we actually use; drop the rest
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
# Step 3: Load scraped store pages (store_pages.jsonl)
# ============================================================
def load_store_pages() -> pd.DataFrame:
    print(f"[3/5] Loading {STORE_PAGES}...")
    rows = read_jsonl(STORE_PAGES)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["appid"] = df["appid"].astype(int)

    # Description is already plain text (the scraper already called get_text).
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
# Step 4: Load reviews (reviews.jsonl)
# ============================================================
def load_reviews() -> pd.DataFrame:
    print(f"[4/5] Loading {REVIEWS}...")
    rows = read_jsonl(REVIEWS)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["appid"] = df["appid"].astype(int)

    # Keep the raw review list (for downstream NLP) and also add a few aggregates.
    df["n_reviews_scraped"] = df["reviews"].apply(lambda lst: len(lst) if isinstance(lst, list) else 0)
    df["avg_review_len"] = df["reviews"].apply(
        lambda lst: (sum(len(r) for r in lst) / len(lst)) if isinstance(lst, list) and lst else 0
    )
    print(f"       -> {len(df):,} rows")
    return df


# ============================================================
# Step 5: Merge + label the target
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

    # ---------- Cleaning ----------
    # Keep type == 'game' only (collect_data.py should have filtered, but
    # this is a safety net).
    if "type" in df.columns:
        df = df[df["type"] == "game"].copy()

    # Drop rows that haven't released yet / have no release date.
    df = df[df["coming_soon"] != True].copy()

    # Fill safe-zero columns with 0
    for col in ["n_achievements", "n_dlc", "n_screenshots_api", "n_movies",
                "n_screenshots_scraped", "n_reviews_scraped", "avg_review_len",
                "positive", "negative"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # required_age comes through as int (0) sometimes, str ("17") other times — normalize.
    if "required_age" in df.columns:
        df["required_age"] = pd.to_numeric(df["required_age"], errors="coerce").fillna(0).astype(int)

    # metacritic_score may have mixed types
    if "metacritic_score" in df.columns:
        df["metacritic_score"] = pd.to_numeric(df["metacritic_score"], errors="coerce")

    # recommendations_total likewise
    if "recommendations_total" in df.columns:
        df["recommendations_total"] = pd.to_numeric(df["recommendations_total"], errors="coerce")

    # SteamSpy numeric columns
    for col in ["userscore", "ccu", "average_forever", "average_2weeks",
                "median_forever", "median_2weeks", "owners_min", "owners_max", "owners_mid"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Price: fall back to SteamSpy's value if the Steam API didn't return one.
    df["price_usd"] = df["price_usd"].fillna(df["steamspy_price_usd"])
    df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")

    # ---------- Target ----------
    # Total reviews = positive + negative
    df["total_reviews"] = df["positive"] + df["negative"]
    # Positive ratio = positive / total
    df["positive_review_ratio"] = df.apply(
        lambda r: (r["positive"] / r["total_reviews"]) if r["total_reviews"] > 0 else None,
        axis=1,
    )
    # is_successful: positive ratio >= 80% AND total_reviews >= 500
    df["is_successful"] = (
        (df["positive_review_ratio"] >= 0.80)
        & (df["total_reviews"] >= 500)
    ).astype("Int64")  # nullable integer

    # If total_reviews == 0, treat the row as having insufficient signal.
    df.loc[df["total_reviews"] == 0, "is_successful"] = pd.NA

    # ---------- Derived time features ----------
    df["release_year"] = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month
    df["release_quarter"] = df["release_date"].dt.quarter
    df["release_dow"] = df["release_date"].dt.dayofweek  # Monday = 0

    print(f"       -> merged: {len(df):,} rows x {df.shape[1]} cols")
    return df


# ============================================================
# Main entry point
# ============================================================
def main():
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    details = load_steam_details()
    steamspy = load_steamspy()
    store = load_store_pages()
    reviews = load_reviews()

    df = merge_all(details, steamspy, store, reviews)

    # ---------- Summary ----------
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

    # ---------- Save ----------
    # Single CSV. list/dict fields are JSON-serialized; downstream code should
    # call json.loads to restore them.
    out_csv = INTERIM_DIR / "cleaned.csv"

    out_df = df.copy()
    # Detect list / dict columns and serialize them as JSON strings.
    list_dict_cols = []
    for col in out_df.columns:
        if out_df[col].apply(lambda x: isinstance(x, (list, dict))).any():
            list_dict_cols.append(col)
            out_df[col] = out_df[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x
            )

    out_df.to_csv(out_csv, index=False)

    print(f"\nSaved:")
    print(f"  {out_csv}   ({len(df):,} rows x {df.shape[1]} cols)")
    if list_dict_cols:
        print(f"\nlist/dict columns were JSON-encoded; downstream readers should use:")
        print(f"  import json")
        print(f"  df = pd.read_csv('{out_csv}')")
        print(f"  for col in {list_dict_cols}:")
        print(f"      df[col] = df[col].fillna('[]').apply(json.loads)")
    print("\nDone! Next step: feature engineering (TF-IDF, tag clustering, etc.)")


if __name__ == "__main__":
    main()
