import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os


# =========================
# 0. Read Data
# =========================

df = pd.read_csv("data/interim/cleaned.csv")

output_dir = "figures/groupA"
os.makedirs(output_dir, exist_ok=True)


# =========================
# 1. Basic Cleaning
# =========================

numeric_cols = [
    "price_usd",
    "positive_review_ratio",
    "is_successful",
    "release_year",
    "release_month",
    "n_supported_languages",
    "n_achievements",
    "n_screenshots_api",
    "n_movies",
    "n_dlc",
    "metacritic_score",
    "owners_mid",
    "ccu",
    "description_len_words",
    "n_tags",
    "total_reviews",
    "recommendations_total"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["is_successful"] = df["is_successful"].fillna(0).astype(int)


def safe_json_loads(x, default_value):
    if pd.isna(x):
        return default_value
    try:
        return json.loads(x)
    except:
        return default_value


df["genres_list"] = df["genres"].apply(lambda x: safe_json_loads(x, []))
df["tags_list"] = df["store_user_tags"].apply(lambda x: safe_json_loads(x, []))


# =========================
# Figure 1:
# Price Distribution by Success Status
# =========================

price_df = df[df["price_usd"].notna()].copy()
price_df = price_df[price_df["price_usd"] <= 100]

success_prices = price_df[price_df["is_successful"] == 1]["price_usd"]
fail_prices = price_df[price_df["is_successful"] == 0]["price_usd"]

plt.figure(figsize=(9, 5))

plt.boxplot(
    [fail_prices, success_prices],
    tick_labels=["Not Successful", "Successful"],
    showfliers=False
)

plt.title("Figure 1. Price Distribution by Success Status")
plt.xlabel("Success Status")
plt.ylabel("Price (USD)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "fig1_price_distribution_by_success.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Figure 1 saved.")
print("Median price - Not Successful:", round(fail_prices.median(), 2))
print("Median price - Successful:", round(success_prices.median(), 2))


# =========================
# Figure 2:
# Positive Review Ratio Distribution with 80% Threshold
# =========================

review_df = df[df["positive_review_ratio"].notna()].copy()

successful_reviews = review_df[review_df["is_successful"] == 1]["positive_review_ratio"]
unsuccessful_reviews = review_df[review_df["is_successful"] == 0]["positive_review_ratio"]

plt.figure(figsize=(9, 5))

plt.hist(
    unsuccessful_reviews,
    bins=25,
    alpha=0.6,
    edgecolor="black",
    label="Not Successful"
)

plt.hist(
    successful_reviews,
    bins=25,
    alpha=0.6,
    edgecolor="black",
    label="Successful"
)

plt.axvline(
    0.8,
    linestyle="--",
    linewidth=2,
    label="80% Positive Review Threshold"
)

plt.title("Figure 2. Positive Review Ratio Distribution and Success Threshold")
plt.xlabel("Positive Review Ratio")
plt.ylabel("Number of Games")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "fig2_review_ratio_threshold.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Figure 2 saved.")
print("Median positive review ratio:", round(review_df["positive_review_ratio"].median(), 4))
print("Overall success rate:", round(df["is_successful"].mean(), 4))


# =========================
# Figure 3:
# Annual Success Rate with Game Count
# =========================

year_df = df[df["release_year"].notna()].copy()
year_df["release_year"] = year_df["release_year"].astype(int)

# Focus on the main project window
year_df = year_df[
    (year_df["release_year"] >= 2015) &
    (year_df["release_year"] <= 2025)
]

annual_success = year_df.groupby("release_year").agg(
    success_rate=("is_successful", "mean"),
    game_count=("appid", "count")
).reset_index()

fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.bar(
    annual_success["release_year"],
    annual_success["game_count"],
    alpha=0.35,
    label="Number of Games"
)

ax1.set_xlabel("Release Year")
ax1.set_ylabel("Number of Games")

ax2 = ax1.twinx()

ax2.plot(
    annual_success["release_year"],
    annual_success["success_rate"],
    marker="o",
    linewidth=2,
    label="Success Rate"
)

ax2.set_ylabel("Success Rate")
ax2.set_ylim(0, 1)

plt.title("Figure 3. Annual Success Rate with Number of Games, 2015-2025")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "fig3_annual_success_rate.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Figure 3 saved.")
print(annual_success)


# =========================
# Figure 4:
# Success Rate by Major Genres
# =========================

genre_rows = []

for i in range(len(df)):
    genres = df.loc[i, "genres_list"]
    success = df.loc[i, "is_successful"]

    for genre in genres:
        genre_rows.append({
            "genre": genre,
            "is_successful": success
        })

genre_df = pd.DataFrame(genre_rows)

genre_summary = genre_df.groupby("genre").agg(
    success_rate=("is_successful", "mean"),
    game_count=("is_successful", "count")
).reset_index()

genre_summary = genre_summary[genre_summary["game_count"] >= 20]

top_genres = genre_summary.sort_values("game_count", ascending=False).head(12)
top_genres = top_genres.sort_values("success_rate", ascending=True)

overall_rate = df["is_successful"].mean()

plt.figure(figsize=(9, 6))

plt.barh(
    top_genres["genre"],
    top_genres["success_rate"]
)

plt.axvline(
    overall_rate,
    linestyle="--",
    linewidth=2,
    label="Overall Success Rate"
)

plt.title("Figure 4. Success Rate by Major Steam Genres")
plt.xlabel("Success Rate")
plt.ylabel("Genre")
plt.xlim(0, 1)

for i in range(len(top_genres)):
    rate = top_genres.iloc[i]["success_rate"]
    count = top_genres.iloc[i]["game_count"]
    plt.text(
        rate + 0.01,
        i,
        "n=" + str(int(count)),
        va="center"
    )

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "fig4_success_rate_by_genre.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Figure 4 saved.")
print(top_genres.sort_values("success_rate", ascending=False))


# =========================
# Figure 5:
# Correlation Heatmap: Pre-release vs Post-release Variables
# =========================

corr_df = df.copy()

if "owners_mid" in corr_df.columns:
    corr_df["log_owners_mid"] = np.log1p(corr_df["owners_mid"])

if "total_reviews" in corr_df.columns:
    corr_df["log_total_reviews"] = np.log1p(corr_df["total_reviews"])

if "ccu" in corr_df.columns:
    corr_df["log_ccu"] = np.log1p(corr_df["ccu"])

if "recommendations_total" in corr_df.columns:
    corr_df["log_recommendations_total"] = np.log1p(corr_df["recommendations_total"])


pre_release_cols = [
    "price_usd",
    "n_supported_languages",
    "n_achievements",
    "n_screenshots_api",
    "n_movies",
    "n_dlc",
    "description_len_words",
    "n_tags",
    "is_successful"
]

post_release_cols = [
    "log_owners_mid",
    "log_total_reviews",
    "log_ccu",
    "log_recommendations_total",
    "positive_review_ratio",
    "is_successful"
]

pre_release_cols = [col for col in pre_release_cols if col in corr_df.columns]
post_release_cols = [col for col in post_release_cols if col in corr_df.columns]

pre_corr = corr_df[pre_release_cols].corr()
post_corr = corr_df[post_release_cols].corr()


def save_heatmap(corr_matrix, title, filename):
    plt.figure(figsize=(8, 6))
    plt.imshow(corr_matrix, aspect="auto")
    plt.colorbar(label="Correlation")

    plt.xticks(
        range(len(corr_matrix.columns)),
        corr_matrix.columns,
        rotation=45,
        ha="right"
    )
    plt.yticks(
        range(len(corr_matrix.columns)),
        corr_matrix.columns
    )

    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            value = corr_matrix.iloc[i, j]
            if not pd.isna(value):
                plt.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8
                )

    plt.title(title)
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, filename),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


save_heatmap(
    pre_corr,
    "Figure 5A. Correlation Heatmap of Pre-release Features",
    "fig5a_prerelease_correlation.png"
)

save_heatmap(
    post_corr,
    "Figure 5B. Correlation Heatmap of Post-release Outcome Variables",
    "fig5b_postrelease_correlation.png"
)

print("Figure 5A and 5B saved.")
print("Pre-release features can be used for the main prediction task.")
print("Post-release variables are useful for EDA but should be excluded from pre-release modeling to avoid data leakage.")


# =========================
# Figure 6:
# Success Rate by Common Steam User Tags
# =========================

tag_rows = []

for i in range(len(df)):
    tags = df.loc[i, "tags_list"]
    success = df.loc[i, "is_successful"]

    for tag in tags:
        tag_rows.append({
            "tag": tag,
            "is_successful": success
        })

tag_df = pd.DataFrame(tag_rows)

tag_summary = tag_df.groupby("tag").agg(
    success_rate=("is_successful", "mean"),
    game_count=("is_successful", "count")
).reset_index()

tag_summary = tag_summary[tag_summary["game_count"] >= 30]

top_tags = tag_summary.sort_values("game_count", ascending=False).head(20)
top_tags = top_tags.sort_values("success_rate", ascending=True)

plt.figure(figsize=(10, 7))

plt.barh(
    top_tags["tag"],
    top_tags["success_rate"]
)

plt.axvline(
    overall_rate,
    linestyle="--",
    linewidth=2,
    label="Overall Success Rate"
)

plt.title("Figure 6. Success Rate by Common Steam User Tags")
plt.xlabel("Success Rate")
plt.ylabel("Steam User Tag")
plt.xlim(0, 1)

for i in range(len(top_tags)):
    rate = top_tags.iloc[i]["success_rate"]
    count = top_tags.iloc[i]["game_count"]
    plt.text(
        rate + 0.01,
        i,
        "n=" + str(int(count)),
        va="center"
    )

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "fig6_success_rate_by_tags.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Figure 6 saved.")
print(top_tags.sort_values("success_rate", ascending=False))


# =========================
# Final Dataset Summary
# =========================

print("==============================")
print("Basic Dataset Summary")
print("==============================")
print("Number of games:", len(df))
print("Overall success rate:", round(df["is_successful"].mean(), 4))
print("Median price:", round(df["price_usd"].median(), 2))
print("Median positive review ratio:", round(df["positive_review_ratio"].median(), 4))
print("Median total reviews:", round(df["total_reviews"].median(), 0))
print("All figures saved to:", output_dir)
