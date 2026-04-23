"""
collect_data.py — 一键抓取 Steam 游戏数据

这一个脚本干三件事：
    1. 从 Steam 官方 API 拿游戏基础信息 (价格、发售日、开发商、tags...)
    2. 从 SteamSpy 拿玩家数/销量估算
    3. 爬 Steam 商店页面拿游戏描述和用户评论

运行方式:
    pip install requests beautifulsoup4 lxml tqdm pandas
    python collect_data.py --limit 100       # 先小规模测试
    python collect_data.py --limit 5000      # 正式跑

结果会保存到 ./data/raw/ 文件夹:
    - steam_details.jsonl      每款游戏一行 JSON, 包含基础信息
    - steamspy.csv             玩家统计
    - store_pages.jsonl        游戏描述 + tags
    - reviews.jsonl            用户评论原文

预计跑 5000 款游戏要 2-3 小时 (因为要礼貌地间隔请求, 避免被 ban)。
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
# 配置
# ============================================================
STEAM_APPLIST_URL = "https://api.steampowered.com/ISteamApps/GetAppList/v0002/?format=json"
STEAM_APPDETAILS_URL = "https://store.steampowered.com/api/appdetails"
STEAMSPY_URL = "https://steamspy.com/api.php"
STORE_URL = "https://store.steampowered.com/app/{appid}"
REVIEWS_URL = "https://store.steampowered.com/appreviews/{appid}"

# 礼貌延时 — Steam/SteamSpy 都会限流
DELAY_STEAM = 1.5
DELAY_STEAMSPY = 1.0
DELAY_SCRAPE = 2.0

HEADERS = {"User-Agent": "Mozilla/5.0 (5243 course project)"}
# 绕过年龄验证页面的 cookie
AGE_COOKIES = {"birthtime": "568022401", "mature_content": "1"}


# ============================================================
# Part 1: 获取游戏列表 (通过 SteamSpy, 比 Steam 官方 API 更稳)
# ============================================================
def fetch_steam_applist(n_pages: int = 1) -> pd.DataFrame:
    """
    从 SteamSpy 拿热门游戏列表。
    每页返回 ~1000 款游戏 (按 owners 估算排序)。
    n_pages=1 就是 top 1000 游戏, 足够做课堂项目了。

    为什么用 SteamSpy 而不是 Steam 官方 GetAppList?
    - Steam 的 GetAppList 在某些网络环境下会 404
    - SteamSpy 稳定, 而且直接返回的就是"有意义的"热门游戏
      (Steam 全量列表里一半是 DLC/soundtrack, 没法用)
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
    """对单个 appid 拿详细信息; 失败返回 None。"""
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
        # 只保留"游戏"类型, 跳过 soundtrack / software / DLC
        if payload.get("type") != "game":
            return None
        payload["appid"] = appid  # 确保存下 appid
        return payload
    except Exception as e:
        print(f"       [warn] appid={appid}: {e}")
        return None


def collect_steam_details(appids: list[int], out_path: Path) -> list[int]:
    """对每个 appid 拿详情并保存; 返回成功抓到的 appid 列表。"""
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
# Part 3: 商店页面 + 评论爬虫
# ============================================================
def scrape_store_page(appid: int) -> dict:
    """抓取游戏商店页面的描述、用户标签、截图数量。"""
    url = STORE_URL.format(appid=appid)
    resp = requests.get(url, headers=HEADERS, cookies=AGE_COOKIES, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    # 长描述
    desc_node = soup.select_one("#game_area_description")
    description = desc_node.get_text(" ", strip=True) if desc_node else None

    # 用户标签 (比官方 genre 细, 比如 "Roguelike", "Souls-like")
    tags = [a.get_text(strip=True) for a in soup.select("a.app_tag")]

    # 媒体数量
    n_screenshots = len(soup.select(".highlight_screenshot"))

    return {
        "appid": appid,
        "description": description,
        "user_tags": tags,
        "n_screenshots": n_screenshots,
    }


def scrape_reviews(appid: int, n: int = 30) -> list[str]:
    """抓取最近 n 条英文评论的正文。"""
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
# 主流程
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

    # Step 1: 拿全量 applist, 随机采样 --limit 个
    applist = fetch_steam_applist()
    sampled = applist.sample(n=min(args.limit, len(applist)), random_state=args.seed)
    sampled.to_csv(args.out_dir / "sampled_appids.csv", index=False)
    appids_to_fetch = sampled["appid"].astype(int).tolist()

    # Step 2: 官方 API 拿详情; 只保留是 game 的
    successful_ids = collect_steam_details(
        appids_to_fetch,
        args.out_dir / "steam_details.jsonl",
    )

    # 只对抓到的有效游戏继续后面两步
    if not successful_ids:
        print("No successful games retrieved — stopping.")
        return

    # Step 3: SteamSpy 统计
    collect_steamspy(successful_ids, args.out_dir / "steamspy.csv")

    # Step 4: 商店页面 + 评论
    collect_scraped(successful_ids, args.out_dir / "scraped", n_reviews=args.n_reviews)

    print("\n All done! Check the data/raw/ folder.")


if __name__ == "__main__":
    main()
