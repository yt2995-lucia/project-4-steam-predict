# 5243 Final Project: Steam游戏成功预测
## End-to-End ML Pipeline Roadmap

---

## 一、项目定位 (One-liner)

**"What makes a Steam game successful? — 基于多源数据的游戏成功预测与分析"**

利用Steam Web API、SteamSpy、商店页面爬虫和玩家评论等多源数据，构建一个端到端的机器学习pipeline，预测一款Steam游戏发售后的"成功程度"，并解释哪些因素最能驱动游戏成功。

为什么选这个题目能拿高分：
- **数据复杂度高**：API + web scraping + 文本 + 结构化数据 = 满足Advanced [10pt]
- **故事性强**：游戏产业是学生熟悉又感兴趣的领域，pre时容易抓住听众
- **Unsupervised自然融入**：游戏tags / 描述 / 评论天然适合做聚类和topic modeling
- **Dashboard有用武之地**：可以做一个"Will my game succeed?"预测器，直观吸引人

---

## 二、核心研究问题 (Predictive Question)

**主任务 (Classification)**: 给定一款游戏发售时可知的特征（价格、类型、开发商、描述、发售档期等），预测它是否会成为一款"成功游戏"。

### Target Variable定义

`is_successful = 1` 若同时满足:
- positive_review_ratio >= 80%, 且
- total_reviews >= 500（避免小样本噪音）

---

## 三、数据来源（多源 = 高分）

| 数据源 | 类型 | 拿什么 |
|---|---|---|
| **Steam Web API** | 官方API | 游戏基础元数据（appid, 名称, 价格, 发售日, 开发商, 发行商, 平台, 语言, 成就数, tags, genres） |
| **SteamSpy API** | 第三方API | 所有者估计区间, 玩家数, 平均游戏时长, peak CCU |
| **Steam Store页面** | Web Scraping | 长文本描述, 系统要求, 截图/视频数量, DLC数量, 用户标签权重 |
| **Steam Community Reviews** | Web Scraping | 大量用户评论原文（做NLP） |

**数据规模目标**：抓取 5,000 - 20,000 款游戏（近5年发售的）。

---

## 四、EDA + Unsupervised Learning

### 描述性分析
- 游戏价格分布、好评率分布
- 年度发售数量趋势、不同genre表现对比
- Tags共现网络图、评论Word Cloud

### Unsupervised Learning（rubric重点）
1. **K-Means / DBSCAN聚类**：对tags和描述embedding做聚类，发现"隐藏类型"
2. **PCA/UMAP降维**：把高维tag向量降到2D可视化游戏地图
3. **Topic Modeling (LDA)**：对游戏描述做主题建模

---

## 五、特征工程

- **基础特征**: 价格, 发售月/季/年, 支持语言数, 成就数, tags multi-hot
- **工程特征**: 开发商历史战绩、发售档期竞争度、描述长度/情感、是否续作
- **文本特征**: TF-IDF + LDA topic分布
- **无监督衍生**: 聚类ID + PCA主成分

---

## 六、建模方案（至少3个 + Stacked）

| 模型 | 为什么选 |
|---|---|
| **Logistic Regression** | Baseline, 可解释 |
| **Random Forest** | Feature importance好, 非线性 |
| **XGBoost** | Tabular data的SOTA |
| **Stacked Ensemble** | Bonus |

**评估**: ROC-AUC + F1 + SHAP + 5-fold CV + 时间划分train/test

---

## 七、Dashboard (Bonus +10pt)

**Streamlit小网页**，4个页面：
1. Overview — 数据全景
2. Explore — UMAP游戏地图
3. Predictor — 输入参数预测成功率
4. Insights — 最重要特征

pre时live demo加分。

---

## 八、时间线

### Week 1 (到pre前, 4/23 - 4/28)
- [ ] 4/23-4/24: 定题分工, 建repo
- [ ] 4/24-4/26: 数据爬取（跑 `collect_data.py`）
- [ ] 4/26-4/27: 初步清洗 + 几张EDA图 + 1个baseline模型
- [ ] **4/28 周一 PRE**: 展示问题 + 数据pipeline + 初步EDA + 建模plan

### Week 2 (4/29 - 5/5)
- [ ] 4/29-4/30: 完整EDA + Unsupervised + 特征工程
- [ ] 5/1-5/2: 3个模型 + 调参 + 对比
- [ ] 5/3: Dashboard搭建
- [ ] 5/4: 报告撰写
- [ ] 5/5 晚11:59前提交

---

## 九、团队分工

| 角色 | 任务 |
|---|---|
| **Data Engineer** | 跑爬虫脚本, 数据存储 |
| **EDA & Viz Lead** | EDA notebook, 可视化, Unsupervised, dashboard |
| **Feature Engineer** | 特征构造, preprocessing |
| **Modeling Lead** | 训练3+模型, 调参, SHAP |
| **PM / Writer** | 报告, slides, README |

---

## 十、下周一Pre Slides结构 (10 min)

| # | 幻灯片 | 时长 |
|---|---|---|
| 1 | Title / Team | 30s |
| 2 | Motivation: 为什么Steam游戏？ | 1min |
| 3 | Predictive Question + Target | 1min |
| 4 | 数据来源（4种，突出复杂度） | 1.5min |
| 5 | Pipeline架构图 | 1min |
| 6 | 初步EDA (2-3张图) | 1.5min |
| 7 | Unsupervised preview (UMAP图) | 1min |
| 8 | 特征工程 + 建模plan | 1.5min |
| 9 | Dashboard mockup | 30s |
| 10 | Timeline + 下一步 | 30s |

**Pre要诀**: 让老师看到plan扎实、数据在手、方向清晰。不需要模型跑完，有1张EDA图+1个baseline AUC就够。
