"""
app.py — Streamlit 可视化 dashboard

这是给老师 / 观众看的最终"成品网页", 跑起来会弹出一个可交互的本地网页。
分 4 个 tab:
    1. Overview   — 数据集概况, 成功率趋势图
    2. Explore    — UMAP 游戏地图, 按 genre/price 筛选
    3. Predictor  — 输入一款游戏的特征, 预测它成功的概率
    4. Insights   — 哪些特征最能预测成功 (SHAP)

运行方式:
    pip install streamlit pandas plotly
    streamlit run app.py

目前代码里很多地方是"占位符"(用 TODO 标记), 等我们真的跑完数据/模型之后,
把 TODO 的地方接上真实的数据和模型就行。现在这个骨架可以直接跑起来看效果。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

# ============================================================
# 页面配置 (必须是第一个 Streamlit 命令)
# ============================================================
st.set_page_config(
    page_title="Steam Game Success Predictor",
    page_icon="🎮",
    layout="wide",
)


# ============================================================
# 数据/模型加载 (带缓存, 避免每次交互都重新读)
# ============================================================
@st.cache_data
def load_features() -> pd.DataFrame:
    """读取处理好的特征表。跑完 feature engineering 后替换这里。"""
    path = Path("data/processed/features.csv")
    if path.exists():
        return pd.read_csv(path)
    # 占位: 返回空表, 后面的 tab 会显示提示
    return pd.DataFrame()


@st.cache_resource
def load_model():
    """读取训练好的模型。跑完 modeling 后替换这里。"""
    # import joblib
    # return joblib.load("models/best_model.joblib")
    return None


df = load_features()
model = load_model()


# ============================================================
# 顶栏
# ============================================================
st.title("🎮 Will Your Steam Game Succeed?")
st.caption("STAT GR5243 — Applied Data Science, Spring 2026 Final Project")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "🗺️ Explore", "🔮 Predictor", "💡 Insights"]
)


# ============================================================
# Tab 1: 数据全景
# ============================================================
with tab1:
    st.header("Dataset Overview")
    if df.empty:
        st.info(" Data not loaded yet. Run `python collect_data.py` first, "
                "then implement feature engineering.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total games", f"{len(df):,}")
        # TODO: 成功率、平均价格
        col2.metric("Successful games", "TODO")
        col3.metric("Avg price", "TODO")

        st.subheader("Success rate by release year × genre")
        st.write("TODO: heatmap — 哪些 genre 在哪些年份成功率最高")


# ============================================================
# Tab 2: 游戏地图
# ============================================================
with tab2:
    st.header("Explore the Game Landscape")
    st.write("UMAP projection of all games, colored by cluster. "
             "Filter by genre, price range, or release year to zoom in.")

    col1, col2 = st.columns([1, 3])
    with col1:
        genre_filter = st.multiselect("Genre", ["Action", "RPG", "Indie", "Strategy"])
        price_range = st.slider("Price range (USD)", 0.0, 80.0, (0.0, 30.0))
        year_range = st.slider("Release year", 2015, 2025, (2020, 2025))

    with col2:
        # TODO: plotly scatter, hover 显示游戏名 + 详情
        st.info("TODO: UMAP scatter plot with the above filters applied.")


# ============================================================
# Tab 3: 成功率预测器 (最吸引眼球的部分)
# ============================================================
with tab3:
    st.header("Predict Your Game's Success Probability")
    st.write("Input some features of a hypothetical game and see how likely it is to succeed.")

    col1, col2 = st.columns(2)
    with col1:
        price = st.slider("Price (USD)", 0.0, 80.0, 19.99, step=0.5)
        release_month = st.selectbox(
            "Release month",
            list(range(1, 13)),
            index=10,
            help="历史上 11 月发售的游戏成功率普遍较高",
        )
        n_languages = st.slider("# supported languages", 1, 30, 5)
    with col2:
        genre = st.selectbox("Primary genre", ["Action", "RPG", "Indie", "Strategy", "Simulation"])
        has_multiplayer = st.checkbox("Has multiplayer", value=False)
        is_sequel = st.checkbox("Is a sequel", value=False)
        description_len = st.slider("Description length (words)", 50, 2000, 400)

    if st.button(" Predict", type="primary"):
        if model is None:
            st.warning("Model not trained yet. Run `python src/models/train.py` "
                       "after collecting data and building features.")
            # 占位: 假装给个结果让 UI 能 demo
            st.metric("Success probability (demo)", "62%")
        else:
            # TODO: 构造 feature vector 喂给模型
            # features_row = build_single_feature_vector(...)
            # proba = model.predict_proba(features_row)[0, 1]
            # st.metric("Success probability", f"{proba:.0%}")
            pass

        # SHAP 解释 (bonus)
        st.subheader("Why this prediction?")
        st.write("TODO: SHAP waterfall — 哪些特征推高 / 拉低了这个概率")


# ============================================================
# Tab 4: 特征重要性 / 关键洞察
# ============================================================
with tab4:
    st.header("What Drives Success on Steam?")
    st.write("Top features ranked by SHAP importance across all games.")

    # TODO: 水平柱状图 of SHAP importance
    st.info("TODO: bar chart — top 15 features by mean |SHAP value|.")

    st.subheader("Surprising findings")
    st.markdown("""
    占位, 跑完模型后把发现填在这里:
    - *e.g. Games released in November have 1.4× higher success probability than July releases*
    - *e.g. Supporting Chinese increases success probability by X%*
    - *e.g. Sequels are more likely to succeed, but with diminishing returns after #3*
    """)


# ============================================================
# 侧栏 (显示项目信息)
# ============================================================
with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "Final project for **STAT GR5243 Applied Data Science** (Spring 2026). "
        "Predicts Steam game success from pre-release features using a stacked ML model."
    )
    st.markdown("### Team")
    st.markdown("- Member 1\n- Member 2\n- Member 3\n- Member 4\n- Member 5")
    st.markdown("---")
    st.caption("Data: Steam Web API + SteamSpy + scraping")
