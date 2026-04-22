# 电商用户行为分析：AARRR 框架下的用户增长与留存研究

> **数据来源**：Kaggle — Multi-Category Online Store Behavior Data (2019)  
> **分析工具**：Python · DuckDB · Tableau  
> **项目目标**：通过 AARRR 框架还原用户从获客到流失的完整生命周期，识别增长机会；涵盖 Cohort 留存曲线、SKU 销售分析、爆款预测（逻辑回归，时序隔离 AUC = 0.904）、库存管理分析四大扩展模块，并基于逻辑回归模型预测复购潜力（AUC = 0.799），设计可落地的 AB 测试方案

## 项目成果展示

- **[Tableau 交互式看板（Story 模式）](https://public.tableau.com/shared/34T6RQS35?:display_count=n&:origin=viz_share_link)**
- **[飞书版电商用户行为分析报告](https://my.feishu.cn/wiki/CH35wMNtYi285skk0TycgntAnfd?from=from_copylink)**

## 项目背景

本项目基于一家大型多品类在线商店 2019 年 10 月至 11 月的完整用户行为日志，包含浏览、加购、购买三类事件，原始数据约 **1.1 亿行**，涵盖 **531 万用户**。

数据集来自 [Kaggle — eCommerce behavior data from multi category store](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)，包含 `event_time`、`event_type`、`product_id`、`category_code`、`brand`、`price`、`user_id`、`user_session` 共 8 个字段。

---

## 数据说明

| 项目 | 内容 |
|---|---|
| 数据来源 | Kaggle Multi-Category Online Store |
| 时间范围 | 2019 年 10 月 1 日 — 2019 年 11 月 30 日 |
| 原始行数 | 约 1.1 亿行 |
| 全站用户数 | 5,316,649 |
| 有购买行为用户 | 697,470（占全站 13.1%）|
| 复购用户（≥2 购买 session）| 256,959（占购买用户 36.8%）|
| 原始字段数 | 8 个 |
| 分析后字段数 | 约 40 个（含衍生字段）|
| 品类代码缺失 | 约 40%（已通过 300+ 条 smart_repair 规则人工修复）|

---

## 核心发现

1. **激活瓶颈**：浏览→加购转化率仅 15.2%，是最大流失节点。仅需 apparel 品类从 0.9% 提升至 2%，全站购买用户可增加约 33%
2. **留存本质是习惯养成**：留存用户购买次数是流失用户的 3.6 倍，AOV 差距只有 1.1 倍——流失用户不是买不起，而是没养成复购习惯
3. **头部效应极显著**：重要价值客户仅占购买用户 38%，却贡献 79.3% GMV；ARPPU 是一般维持客户的 52.9 倍
4. **价格弹性低于预期**：所有主力品类弹性系数 <1，用户被时机触发而非被折扣打动；大促 GMV 达日均 7 倍印证了节点触发效应
5. **复购预测模型 AUC = 0.799**：主动去除数据泄露特征，`cart_count` 是最强正向信号；另构建首购7天无泄露模型（AUC = 0.653，Precision = 87.9%）可实时部署
6. **Cohort 留存**：October Cohort 次月购买留存率 **26.3%**，显著低于整体购买用户 73.7% 月留存——新用户首购后习惯养成是关键干预窗口
7. **SKU 帕累托**：活跃 SKU 共 160,594 个，**676 个 SKU 贡献 80% GMV**；品牌 GMV 榜首 Apple ¥236.6M；品类平均转化率差异超 5 倍，结构性优化空间显著
8. **爆款预测（逻辑回归 AUC = 0.904，时序隔离）**：用 10 月特征预测 11 月爆款标签，彻底避免数据泄露；`log_cart_per_day`（日均加购量）是最强先行信号，上架 5 天后即可评分
9. **库存风险分级**：HIGH RISK SKU 占总量 **6.3%**（高销速+高波动），建议提前备货 1.5x ROP；LOW RISK SKU 可转为 JIT 模式，节省仓储资金占用
10. **Sub 品类共购分析**：切换至 category_sub 粒度后发现高精度关联对——nutrition↔supplement（Lift=64.86）、iron↔ironing_board（Lift=7.73）、jeans↔shoes（Lift=7.02）；主流大品类跨类共购 Lift 仍偏低（1–1.7），推荐策略以品类内互补为主，高 Lift 小众对为辅

---

## 技术栈

| 工具 | 用途 |
|---|---|
| Python 3.x | 数据处理、特征工程、逻辑回归建模 |
| DuckDB | 1.1 亿行数据的 Out-of-Core SQL 聚合 |
| pandas / numpy | 向量化计算（RFM 打分、决策链分析）|
| scikit-learn | 逻辑回归 + StandardScaler + 模型评估 |
| pantab / tableauhyperapi | 导出 Tableau Hyper 文件（Arrow 流式写入）|
| Tableau Public | 交互式 Dashboard |

---

## 关键技术实现

### 1. 大规模数据处理：DuckDB Out-of-Core 计算

原始 1.1 亿行数据远超单机内存，通过 DuckDB 视图链式处理，全程不把原始数据加载进 pandas：

```python
import duckdb

con = duckdb.connect()
con.execute("SET memory_limit='12GB'")
con.execute("SET threads=2")
con.execute("SET preserve_insertion_order=false")

# 直接对 parquet 文件建视图，不读入内存
con.execute(f"CREATE VIEW oct AS SELECT * FROM read_parquet('{OCT_FILE}')")
con.execute(f"CREATE VIEW nov AS SELECT * FROM read_parquet('{NOV_FILE}')")

# 所有聚合通过 SQL 完成，pandas 只接触最终的小表
user_metrics = con.execute("""
    SELECT
        CAST(user_id AS VARCHAR) AS user_id,
        COUNT(*)                 AS total_actions,
        COUNT(DISTINCT user_session) AS total_sessions,
        COUNT(CASE WHEN event_type='purchase' THEN 1 END) AS purchase_count,
        SUM(CASE WHEN event_type='purchase' THEN price END) AS total_gmv
    FROM events_full
    GROUP BY user_id
""").df()
```

### 2. 品类标签修复：双层 smart_repair 规则

原始数据约 40% 的商品缺失品类代码，通过人工审核建立 300+ 条映射规则，优先用 `(category_id, brand)` 精细匹配，再用纯 `category_id` 兜底：

```python
# 精细模式：同一 category_id 下不同品牌归不同品类
DETAIL_BRAND_MAP = {
    (2053013563651392361, "lucente"): "furniture.universal.light",
    (2053013563651392361, "sokolov"): "accessories",
    # ... 300+ 条规则
}

# 在 DuckDB SQL 里动态生成 CASE WHEN，一次扫表完成修复
detail_sql = "\n".join(
    f"WHEN category_id={cid} AND brand_clean='{b}' THEN '{c}'"
    for (cid, b), c in DETAIL_BRAND_MAP.items()
)
```

### 3. 决策链分析：向量化替代行级循环

追踪每次购买前用户的历史行为，用 pandas 向量化操作降至 O(n)：

```python
# cumcount() = 该 session 在此用户历史里的位置 = 它前面有多少个 session
session_clean["pre_session_count"] = session_clean.groupby("user_id").cumcount()

# 同品类前序 session 占比 → 衡量用户的品类专注度
session_clean["_same_cat_n"] = session_clean.groupby(["user_id", "main_cat"]).cumcount()
session_clean["same_cat_ratio"] = np.where(
    session_clean["pre_session_count"] == 0,
    np.nan,
    session_clean["_same_cat_n"] / session_clean["pre_session_count"]
)
```

### 4. RFM 用户分层：基于分位数的自适应打分

```python
# 计算分位数作为打分边界，自适应数据分布（不硬编码阈值）
r_q = rfm["recency"].quantile([0.25, 0.5, 0.75, 0.9]).to_dict()

def score_recency(r):
    return 5 if r<=r_q[0.25] else(4 if r<=r_q[0.5] else(
           3 if r<=r_q[0.75] else(2 if r<=r_q[0.9] else 1)))

rfm["RFM_Total"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

def segment(score):
    if score >= 12: return "重要价值客户"
    elif score >= 9: return "重要发展客户"
    elif score >= 6: return "中坚活跃客户"
    elif score >= 4: return "一般维持客户"
    else:            return "流失/边缘客户"
```

### 5. 复购预测模型：逻辑回归（AUC = 0.799）

主动去除 `cart_to_purchase_rate` 等数据泄露特征，保留 8 个可实时获取的无泄露特征：

| 特征 | 系数 | 方向 | 业务解读 |
|---|---|---|---|
| `cart_count` | **+2.341** | ⬆️ | 最强正向信号，加购是购买意图最可靠的先行指标 |
| `total_sessions` | +1.307 | ⬆️ | 互动越深，复购越强 |
| `cart_no_purchase_sessions` | **-0.966** | ⬇️ | 反直觉：频繁加购却不买，实际复购更低 |
| `view_count` | -0.287 | ⬇️ | 纯浏览不加购 = 低购买意图 |
| `decision_style_score` | -0.246 | ⬇️ | 越犹豫复购越低 |
| `focus_style_score` | +0.163 | ⬆️ | 专注型用户品牌忠诚度高 |

另构建**首购7天无泄露模型**（Block 16）：仅用首购后7天行为特征，AUC = 0.653，Precision = 87.9%，用户首购第7天即可实时打分并触发干预。

### 6. T2 大表导出：Arrow 流式写入，零中间文件

```python
BATCH_SIZE = 200_000  # 每批 20 万行，内存约 50-100 MB

with HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper:
    with Connection(hyper.endpoint, "T2_fact_purchases.hyper",
                    CreateMode.CREATE_AND_REPLACE) as conn:
        conn.catalog.create_table_if_not_exists(T2_TABLE_DEF)
        arrow_reader = con.execute("SELECT ... FROM events_full WHERE event_type='purchase'") \
                          .fetch_arrow_reader(BATCH_SIZE)
        with Inserter(conn, T2_TABLE_DEF) as inserter:
            for arrow_batch in arrow_reader:
                df_batch = arrow_batch.to_pandas()
                inserter.add_rows(df_batch.itertuples(index=False, name=None))
            inserter.execute()
# 全程磁盘上只有最终 .hyper 文件在增长，无任何临时文件
```

### 7. 品类共购分析：Sub 品类粒度 + 双重过滤

在 category_sub 粒度下构建共购矩阵，识别高关联品类对：

```python
# 过滤噪声品类，按用户×月份粒度构建共购矩阵
EXCLUDE_CATS = {'unknown', 'smartphone'}
MIN_SUPPORT  = 50    # 品类本身至少出现 50 个用户月份
MIN_CO_COUNT = 30    # 品类对至少共购 30 次
MIN_LIFT     = 2.0   # 高关联阈值

pair_df['lift'] = (pair_df['co_count'] / len(user_month_cats)) / (
    (pair_df['support_a'] / len(user_month_cats)) *
    (pair_df['support_b'] / len(user_month_cats))
)
# nutrition↔supplement Lift=64.86，iron↔ironing_board Lift=7.73
```

---

## 文件结构

```
REES46 电商行为分析/
│
├── ecommerce_behavior_analysis.ipynb   # 主分析 Notebook（Block 0–23 完整流程）
├── 分析报告与运营建议.md               # 深度分析报告（含 AB 测试方案）
│
├── charts/                             # 图表输出
│   ├── tb_*.png                        # Tableau 全量数据图表
│   ├── 74_model_roc_curve.png          # ROC 曲线（复购预测）
│   ├── 73_model_feature_importance.png # 特征重要性（复购预测）
│   ├── block16_7day_model.png          # 首购7天模型结果
│   ├── block18_price_elasticity.png    # 价格弹性分析
│   ├── 19_market_basket_v2.png         # 品类共购矩阵（category_sub 粒度）
│   ├── cohort_retention_curve.png      # Cohort 留存曲线（Block 20）
│   ├── sku_analysis.png                # SKU 帕累托 & 品牌分析（Block 21）
│   ├── bestseller_prediction.png       # 爆款预测 ROC + 特征系数（Block 22）
│   └── inventory_management.png        # 库存风险分级图（Block 23）
│
└── README.md
```

### Notebook 模块索引

| Block | 内容 | 关键输出 |
|---|---|---|
| Block 0 | 快速恢复（Kernel 重启后运行） | 恢复所有中间变量 |
| Block 1 | 数据读取 & 分层抽样（一次性） | df_sample_30k.parquet |
| Block 1.5 | 数据质量说明与清洗策略 | 清洗文档 |
| Block 2 | 价格分布 EDA | price_distribution.png |
| Block 3 | 整体转化漏斗 | 浏览→加购→购买漏斗 |
| Block 4 | 品类转化漏斗 & 四象限气泡图 | category_bubble.png |
| Block 5 | RFM 用户分层 | rfm_distribution.png |
| Block 6 | Session 分析 | session 统计 |
| Block 7 | 决策链追溯（向量化） | 购买决策路径 |
| Block 8 | 专注度 × 决策风格交叉分析 | decision_style_session.png |
| Block 9 | 活跃度分析（日/时/周/大促） | daily_activity_promo.png |
| Block 10 | Top5 品类 × 时间热力图 | category_time_heatmap.png |
| Block 11 | RFM 分层 × 活跃度分析 | RFM × 时段热力图 |
| Block 12 | 决策风格 × 活跃度分析 | 决策风格活跃时段 |
| Block 13 | 用户留存分析（D1/D7/D30）| retention_behavior.png |
| Block 14 | Tableau 数据导出（T1+T2）| .hyper 文件 |
| Block 15 | 复购预测逻辑回归（AUC=0.799）| model_roc_curve.png |
| Block 16 | 首购7天无泄露模型（AUC=0.653）| block16_7day_model.png |
| Block 17 | 用户 LTV 估算（BG/NBD）| LTV 预测 |
| Block 18 | 价格弹性分析 | price_elasticity.png |
| Block 19 | 品类共购分析（category_sub 粒度，Market Basket）| market_basket_v2.png |
| **Block 20** | **Cohort 留存曲线** | **cohort_retention_curve.png** |
| **Block 21** | **SKU 分析（帕累托 + 品牌）** | **sku_analysis.png** |
| **Block 22** | **爆款预测（逻辑回归）** | **bestseller_prediction.png** |
| **Block 23** | **库存管理（Safety Stock + ROP）** | **inventory_management.png** |

---

## 如何复现

**环境：**
```bash
pip install duckdb pandas numpy scikit-learn pantab tableauhyperapi
```

**步骤：**

1. 从 Kaggle 下载原始数据，转为 parquet 格式：
   ```python
   import pandas as pd
   pd.read_csv('2019-Oct.csv').to_parquet('2019-Oct.parquet')
   pd.read_csv('2019-Nov.csv').to_parquet('2019-Nov.parquet')
   ```

2. 修改 notebook 顶部 `⚙️ 全局路径配置`：
   ```python
   DATA_PATH    = "你的数据路径/"
   OUTPUT_PATH  = DATA_PATH + "hyper_v2/"
   MEMORY_LIMIT = "12GB"   # 建议 = 物理内存 × 0.75
   ```

3. **运行顺序：**
   - 首次运行：Block 0（配置）→ Block 1（一次性抽样，约5分钟）→ Blocks 2–15（分析+导出，约30分钟）→ Blocks 16–23（扩展模块）
   - 重启 Kernel 后：仅需 Block 0（配置）→ Block 0（快速恢复，约10秒）

4. 将 `hyper_v2/` 中的 `.hyper` 文件导入 Tableau Desktop 即可连接 Dashboard

> ⚠️ Block 14 全量处理约需 20–30 分钟，建议内存 ≥ 16GB

---

## 关于作者

Cristina Yan  严梦圆  
yancristina071@gmail.com
