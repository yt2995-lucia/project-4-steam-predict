"""
collect_data.py — One-shot Steam game data collection.

This script does three things:
    1. Pull basic game metadata from the official Steam Web API
       (price, release date, developer, tags, ...)
    2. Pull owner / playtime estimates from SteamSpy
    3. Scrape Steam store pages for long descriptions and user reviews

Usage:
    pip install requests beautifulsoup4 lxml tqdm pandas
    python src/collect_data.py --limit 100       # quick smoke test
    python src/collect_data.py --limit 5000      # full run

Outputs go to ./data/raw/ :
    - steam_details.jsonl      one JSON object per game (basic metadata)
    - steamspy.csv             player-count statistics
    - store_pages.jsonl        descriptions + user-defined tags
    - reviews.jsonl            raw user reviews

A 5000-game run takes 2-3 hours because of polite rate limiting
(without the delays the Steam / SteamSpy endpoints will block you).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================
STEAM_APPLIST_URL = "https://api.steampowered.com/ISteamApps/GetAppList/v0002/?format=json"
STEAM_APPDETAILS_URL = "https://store.steampowered.com/api/appdetails"
STEAMSPY_URL = "https://steamspy.com/api.php"
STORE_URL = "https://store.steampowered.com/app/{appid}"
REVIEWS_URL = "https://store.steampowered.com/appreviews/{appid}"

# Polite delays — both Steam and SteamSpy rate-limit aggressive callers.
DELAY_STEAM = 1.5
DELAY_STEAMSPY = 1.0
DELAY_SCRAPE = 2.0

HEADERS = {"User-Agent": "Mozilla/5.0 (5243 course project)"}
# Cookies that bypass the age-verification interstitial.
AGE_COOKIES = {"birthtime": "568022401", "mature_content": "1"}


# ============================================================
# Part 1: Game list (fetched via SteamSpy — more reliable than Steam's
# official GetAppList endpoint)
# ============================================================
def fetch_steam_applist(n_pages: int = 1) -> pd.DataFrame:
    """
    Pull a list of popular games from SteamSpy.
    Each page returns ~1000 games, sorted by ownership estimate.
    n_pages=1 is the top 1000 — enough for a course project.

    Why SteamSpy instead of Steam's official GetAppList?
    - Steam's GetAppList sometimes 404s in certain network environments.
    - SteamSpy is stable and already filters out the noise; Steam's full
      app list is roughly half DLC and soundtracks, which we don't want.
    """
    print(f"[1/4] Fetching game list from SteamSpy ({n_pages} page × 1000 games)...")
    rows = []
    for page in range(n_pages):
        try:
            resp = requests.get(
                STEAMSPY_URL,
                params={"request": "all", "page": page},
                headers=HEADERS,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            for appid_str, info in data.items():
                rows.append({
                    "appid": int(appid_str),
                    "name": info.get("name", ""),
                })
            time.sleep(DELAY_STEAMSPY)
        except Exception as e:
            print(f"       [warn] page {page}: {e}")

    df = pd.DataFrame(rows).drop_duplicates(subset=["appid"]).reset_index(drop=True)
    df = df[df["name"].astype(str).str.strip() != ""].reset_index(drop=True)
    print(f"       -> {len(df):,} games in list")
    return df


def fetch_steam_details(appid: int) -> dict | None:
    """Fetch full metadata for a single appid; return None on failure."""
    try:
        resp = requests.get(
            STEAM_APPDETAILS_URL,
            params={"appids": appid, "l": "english", "cc": "us"},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json().get(str(appid), {})
        if not data.get("success"):
            return None
        payload = data["data"]
        # Keep only items typed as "game" — drop soundtracks, software, DLC.
        if payload.get("type") != "game":
            return None
        payload["appid"] = appid  # ensure appid is preserved in the record
        return payload
    except Exception as e:
        print(f"       [warn] appid={appid}: {e}")
        return None


def collect_steam_details(appids: list[int], out_path: Path) -> list[int]:
    """Fetch details for every appid and write to disk; return successful ids."""
    print(f"[2/4] Fetching Steam details for {len(appids)} apps...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    successful_ids = []
    with out_path.open("w") as f:
        for appid in tqdm(appids, desc="  steam_api"):
            details = fetch_steam_details(int(appid))
            if details:
                f.write(json.dumps(details) + "\n")
                successful_ids.append(int(appid))
            time.sleep(DELAY_STEAM)
    print(f"       -> saved {len(successful_ids)} games to {out_path}")
    return successful_ids


# ============================================================
# Part 2: SteamSpy API
# ============================================================
def fetch_steamspy(appid: int) -> dict | None:
    try:
        resp = requests.get(
            STEAMSPY_URL,
            params={"request": "appdetails", "appid": appid},
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"       [warn] steamspy appid={appid}: {e}")
        return None


def collect_steamspy(appids: list[int], out_path: Path) -> None:
    print(f"[3/4] Fetching SteamSpy stats for {len(appids)} apps...")
    rows = []
    for appid in tqdm(appids, desc="  steamspy"):
        record = fetch_steamspy(appid)
        if record:
            rows.append(record)
        time.sleep(DELAY_STEAMSPY)
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"       -> saved {len(df)} rows to {out_path}")


# ============================================================
# Part 3: Store-page + review scraper
# ============================================================
def scrape_store_page(appid: int) -> dict:
    """Scrape the description, user tags, and screenshot count from a store page."""
    url = STORE_URL.format(appid=appid)
    resp = requests.get(url, headers=HEADERS, cookies=AGE_COOKIES, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    # Long description
    desc_node = soup.select_one("#game_area_description")
    description = desc_node.get_text(" ", strip=True) if desc_node else None

    # User-defined tags — finer-grained than official genres
    # (e.g. "Roguelike", "Souls-like").
    tags = [a.get_text(strip=True) for a in soup.select("a.app_tag")]

    # Media counts
    n_screenshots = len(soup.select(".highlight_screenshot"))

    return {
        "appid": appid,
        "description": description,
        "user_tags": tags,
        "n_screenshots": n_screenshots,
    }


def scrape_reviews(appid: int, n: int = 30) -> list[str]:
    """Scrape the most recent n English reviews."""
    params = {
        "json": 1,
        "filter": "recent",
        "language": "english",
        "num_per_page": min(n, 100),
        "purchase_type": "all",
    }
    try:
        resp = requests.get(REVIEWS_URL.format(appid=appid), params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        return [r["review"] for r in data.get("reviews", [])][:n]
    except Exception as e:
        print(f"       [warn] reviews appid={appid}: {e}")
        return []


def collect_scraped(appids: list[int], out_dir: Path, n_reviews: int = 30) -> None:
    print(f"[4/4] Scraping store pages + reviews for {len(appids)} apps...")
    out_dir.mkdir(parents=True, exist_ok=True)
    store_file = out_dir / "store_pages.jsonl"
    review_file = out_dir / "reviews.jsonl"

    with store_file.open("w") as sf, review_file.open("w") as rf:
        for appid in tqdm(appids, desc="  scraping"):
            try:
                store_data = scrape_store_page(appid)
                sf.write(json.dumps(store_data) + "\n")
            except Exception as e:
                print(f"       [warn] store appid={appid}: {e}")

            reviews = scrape_reviews(appid, n=n_reviews)
            rf.write(json.dumps({"appid": appid, "reviews": reviews}) + "\n")

            time.sleep(DELAY_SCRAPE)
    print(f"       -> saved to {store_file} and {review_file}")


# ============================================================
# Main entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Collect Steam game data from multiple sources.")
    parser.add_argument("--limit", type=int, default=100,
                        help="How many apps to sample from Steam's full list (default: 100 for testing)")
    parser.add_argument("--n-reviews", type=int, default=30,
                        help="Number of reviews to scrape per game")
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw"),
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: fetch the full app list and randomly sample --limit games.
    applist = fetch_steam_applist()
    sampled = applist.sample(n=min(args.limit, len(applist)), random_state=args.seed)
    sampled.to_csv(args.out_dir / "sampled_appids.csv", index=False)
    appids_to_fetch = sampled["appid"].astype(int).tolist()

    # Step 2: pull official-API details; keep only the rows typed as "game".
    successful_ids = collect_steam_details(
        appids_to_fetch,
        args.out_dir / "steam_details.jsonl",
    )

    # Only continue downstream work for the games that came back successfully.
    if not successful_ids:
        print("No successful games retrieved — stopping.")
        return

    # Step 3: SteamSpy stats
    collect_steamspy(successful_ids, args.out_dir / "steamspy.csv")

    # Step 4: store pages + reviews
    collect_scraped(successful_ids, args.out_dir / "scraped", n_reviews=args.n_reviews)

    print("\n All done! Check the data/raw/ folder.")


if __name__ == "__main__":
    main()
