# 营销多智能体系统 (Marketing Multi-Agent System)

基于 **LangGraph + 多智能体协作 + RAG + 上下文Bandit路由** 的电商营销自动化系统，基于 Olist 巴西电商真实数据集构建。系统覆盖从用户洞察、用户分层、智能路由、商品推荐、营销策略生成、文案创作到效果评估的完整营销链路。

---

## 1. 项目目标与核心价值主张

### 1.1 项目目标

本项目旨在构建一个**端到端的智能营销决策系统**，通过多智能体协作实现个性化营销方案的全自动生成与优化。系统以单个用户为决策单元，综合分析用户行为数据、商品偏好和市场知识，输出可执行的营销建议。

### 1.2 核心价值主张

| 能力维度 | 具体价值 |
|---------|---------|
| **用户洞察** | 基于 RFM 模型 + K-Means 聚类实现精细化用户分层 |
| **智能路由** | 基于上下文 bandit 的动态策略路由，适配不同用户特征 |
| **知识增强** | RAG 检索营销知识库，生成专业级营销策略 |
| **内容生成** | LLM 驱动的个性化文案，支持多风格多渠道 |
| **效果预测** | 转化率与 ROI 预估，辅助营销决策优化 |

---

## 2. 环境搭建

### 2.1 系统要求

- **Python 版本**：3.11+（推荐）
- **操作系统**：Windows / macOS / Linux
- **内存**：推荐 8GB 以上
- **磁盘**：至少 2GB 可用空间

### 2.2 创建虚拟环境

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2.3 安装依赖

```bash
pip install -r requirements.txt
```

### 2.4 配置文件设置

在项目根目录创建 `.env` 文件：

```env
# ===================== 通义千问 API 配置 =====================
QWEN_API_KEY=你的真实API_KEY
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-turbo
QWEN_EMBEDDING_MODEL=text-embedding-v2

# ===================== Bandit 路由参数 =====================
BANDIT_N_ACTIONS=5
BANDIT_RANDOM_SEED=42
BANDIT_MIN_PSCORE=0.03
CONTEXTUAL_BANDIT_ALPHA=0.6
FUSION_UPLIFT_WEIGHT=0.45
FUSION_BANDIT_WEIGHT=0.55
```

### 2.5 数据准备

将 Olist 电商数据集 CSV 文件放入 `data/olist/` 目录，所需文件清单：

| 文件名 | 描述 | 是否必需 |
|-------|------|---------|
| `olist_orders_dataset.csv` | 订单主表 | ✅ 必需 |
| `olist_order_items_dataset.csv` | 订单商品明细 | ✅ 必需 |
| `olist_order_payments_dataset.csv` | 订单支付信息 | ✅ 必需 |
| `olist_customers_dataset.csv` | 客户信息 | ✅ 必需 |
| `olist_products_dataset.csv` | 商品信息 | ✅ 必需 |
| `olist_geolocation_dataset.csv` | 地理信息 | ⬜ 可选 |
| `olist_order_reviews_dataset.csv` | 评价信息 | ⬜ 可选 |
| `olist_sellers_dataset.csv` | 卖家信息 | ⬜ 可选 |
| `product_category_name_translation.csv` | 品类翻译 | ✅ 必需 |

> **数据获取**：Olist 数据集可从 [Kaggle Olist Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) 下载。

---

## 3. 智能体功能模块

本系统包含 **6 个核心智能体**，通过 LangGraph 工作流编排串联。

### 3.1 架构总览

```
analyze_user → recommend_products → generate_strategy → generate_content → evaluate
                      ↑
                      └── router_agent (智能路由)
```

### 3.2 用户分析 Agent (UserAnalysisAgent)

**文件位置**：`src/agents/user_analysis_agent.py`

**功能描述**：对用户进行多维度分析，完成 RFM 特征计算与聚类分层。

**核心算法**：
- **RFM 模型**：Recency（近度）、Frequency（频度）、Monetary（额度）
- **K-Means 聚类**：基于 RFM 特征的用户分层

**用户分层定义**：

| 分层 | RFM 条件 | 描述 |
|-----|---------|------|
| 高价值忠诚用户 | 近30天有购买，次数≥3，消费≥200 | 核心价值用户，需维护 |
| 潜力增长用户 | 近60天有购买，次数1-2，消费100-200 | 成长型用户，挖掘价值 |
| 新用户 | 近90天首次购买，次数=1 | 培育转化对象 |
| 一般价值用户 | 近180天有购买，消费50-100 | 常规运营对象 |
| 流失风险用户 | 180天以上无购买 | 召回优先级用户 |

**输出参数**：
```python
{
    "customer_unique_id": str,      # 用户唯一ID
    "recency": int,                  # 最近购买距今天数
    "frequency": int,                # 购买次数
    "monetary": float,              # 总消费金额
    "user_segment": str,             # 用户分层标签
    "favorite_categories": List[str],  # 偏好品类
    "avg_order_value": float         # 平均订单价值
}
```

### 3.3 商品推荐 Agent (ProductRecommendationAgent)

**文件位置**：`src/agents/product_recommendation_agent.py`

**功能描述**：基于用户偏好和品类特征，推荐个性化商品列表。

**推荐策略**：
1. **品类偏好匹配**：根据用户历史购买品类推荐同类商品
2. **热销补足**：结合当前热销商品补充推荐
3. **价格区间过滤**：根据用户消费水平适配价格带

**输入参数**：
```python
user_profile: {
    "favorite_categories": List[str],
    "monetary": float,
    "user_segment": str
}
```

**输出参数**：
```python
recommended_products: List[Dict[str, Any]]
# 每项包含: product_id, product_name, category, price, score
```

### 3.4 营销策略 Agent (StrategyAgent)

**文件位置**：`src/agents/strategy_agent.py`

**功能描述**：基于 RAG 知识库检索，生成个性化营销策略。

**RAG 工作流程**：
1. 用户查询 embedding
2. 向量数据库相似度检索
3. 知识库文档相关性匹配
4. LLM 综合生成策略建议

**知识库内容**：
- `knowledge_base/marketing_strategies.md`：用户分层运营策略
- `knowledge_base/copywriting_templates.md`：文案模板库

**输出参数**：
```python
strategy: {
    "segment_strategy": str,         # 分层策略
    "channel": List[str],            # 推荐触达渠道
    "timing": str,                   # 最佳触达时间
    "discount_type": str,            # 优惠类型
    "key_message": str               # 核心话术方向
}
```

### 3.5 文案生成 Agent (ContentGenerationAgent)

**文件位置**：`src/agents/content_generation_agent.py`

**功能描述**：基于用户画像、策略和推荐商品，生成个性化营销文案。

**生成要素**：
- 用户尊称（基于分层）
- 个性化商品推荐
- 专属优惠信息
- 行动号召（CTA）

**输入参数**：
```python
user_profile: Dict,      # 用户画像
strategy: Dict,          # 营销策略
recommended_products: List[Dict]  # 推荐商品
```

**输出参数**：
```python
content: str  # 生成的营销文案
```

### 3.6 效果评估 Agent (EvaluationAgent)

**文件位置**：`src/agents/evaluation_agent.py`

**功能描述**：基于历史订单特征预测营销效果。

**预测指标**：
| 指标 | 描述 |
|-----|------|
| 预期转化率 | 预估营销触达后的转化概率 |
| 销售额提升 | 相比基准的预期提升幅度 |
| ROI | 投资回报率估算 |

**算法**：RandomForestRegressor，基于用户特征和商品属性预测

### 3.7 路由 Agent (RouterAgent)

**文件位置**：`src/agents/router_agent.py`

**功能描述**：智能选择最优营销策略路由方案。

**路由策略池**：
| 策略 | 适用场景 |
|-----|---------|
| `uplift` | 注重用户增量响应 |
| `contextual_bandit` | 上下文特征丰富场景 |
| `fusion` | 平衡增量与 exploitation |
| `rl_policy` | 长期回报优化 |
| `rule_engine` | 冷启动/数据稀疏场景 |

**OPE（Off-Policy Evaluation）反馈机制**：
- 支持 DR、IPS、SNIPS 三种估计器
- 自适应学习率调整
- 策略优势动态追踪

**核心配置参数**：

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `ROUTER_LOW_PSCORE` | 0.08 | 低倾向分阈值 |
| `ROUTER_HIGH_COMPLEXITY` | 0.95 | 高复杂度阈值 |
| `ROUTER_MODEL_MIN_CONFIDENCE` | 0.45 | 模型最小置信度 |
| `ROUTER_OPE_FEEDBACK_ENABLED` | true | 是否启用OPE反馈 |

---

## 4. 完整使用示例

### 4.1 基础运行

```bash
python main.py
```

### 4.2 代码调用示例

```python
import config
from src.agents.user_analysis_agent import UserAnalysisAgent
from src.agents.product_recommendation_agent import ProductRecommendationAgent
from src.agents.strategy_agent import StrategyAgent
from src.agents.content_generation_agent import ContentGenerationAgent
from src.agents.evaluation_agent import EvaluationAgent
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from typing import Dict, Any, List

class State(BaseModel):
    customer_unique_id: str
    user_profile: Dict[str, Any] = {}
    recommended_products: List[Dict[str, Any]] = []
    strategy: Dict[str, Any] = {}
    content: str = ""
    evaluation_report: Dict[str, Any] = {}

# 初始化智能体
user_agent = UserAnalysisAgent()
product_agent = ProductRecommendationAgent(user_agent.order_items, user_agent.products)
strategy_agent = StrategyAgent()
content_agent = ContentGenerationAgent()
eval_agent = EvaluationAgent(user_agent.orders)

# 构建工作流
workflow = StateGraph(State)
workflow.add_node("analyze_user", lambda s: {"user_profile": user_agent.analyze_user(s.customer_unique_id)})
workflow.add_node("recommend_products", lambda s: {"recommended_products": product_agent.recommend_products(s.user_profile)})
workflow.add_node("generate_strategy", lambda s: {"strategy": strategy_agent.generate_strategy(s.user_profile)})
workflow.add_node("generate_content", lambda s: {"content": content_agent.generate_content(s.user_profile, s.strategy, s.recommended_products)})
workflow.add_node("evaluate", lambda s: {"evaluation_report": eval_agent.evaluate(s.user_profile, s.strategy, s.content)})

workflow.set_entry_point("analyze_user")
workflow.add_edge("analyze_user", "recommend_products")
workflow.add_edge("recommend_products", "generate_strategy")
workflow.add_edge("generate_strategy", "generate_content")
workflow.add_edge("generate_content", "evaluate")
workflow.add_edge("evaluate", END)

app = workflow.compile()

# 执行工作流
customer_id = user_agent.get_sample_users(n=1)[0]
result = app.invoke({"customer_unique_id": customer_id})

print(f"用户分层: {result['user_profile']['user_segment']}")
print(f"推荐商品: {result['recommended_products']}")
print(f"营销文案: {result['content']}")
print(f"效果预测: {result['evaluation_report']}")
```

### 4.3 典型应用场景

#### 场景一：高价值用户维护

```python
# 获取高价值用户样本
vip_users = user_agent.get_sample_users(segment="高价值忠诚用户", n=3)
for user_id in vip_users:
    result = app.invoke({"customer_unique_id": user_id})
    # 输出专属VIP营销方案
```

#### 场景二：流失用户召回

```python
# 获取流失风险用户
churn_users = user_agent.get_sample_users(segment="流失风险用户", n=5)
for user_id in churn_users:
    result = app.invoke({"customer_unique_id": user_id})
    # 输出召回营销方案（高折扣+专属关怀）
```

---

## 5. 配置指南

### 5.1 模型配置 (config.py)

| 配置项 | 默认值 | 说明 |
|-------|-------|------|
| `QWEN_MODEL` | `qwen-turbo` | LLM 模型名称 |
| `QWEN_EMBEDDING_MODEL` | `text-embedding-v2` | Embedding 模型 |
| `TEMPERATURE` | `0.7` | 生成温度参数 |
| `MAX_TOKENS` | `512` | 最大生成 token 数 |

### 5.2 算法参数

| 配置项 | 默认值 | 说明 |
|-------|-------|------|
| `KMEANS_CLUSTERS` | `5` | 用户聚类数量 |
| `RECOMMEND_PRODUCT_NUM` | `3` | 推荐商品数量 |

### 5.3 Bandit 路由参数

| 配置项 | 默认值 | 说明 |
|-------|-------|------|
| `BANDIT_N_ACTIONS` | `5` | 动作空间大小 |
| `BANDIT_MIN_PSCORE` | `0.03` | 倾向分下限 |
| `CONTEXTUAL_BANDIT_ALPHA` | `0.6` | 正则化参数 |
| `FUSION_UPLIFT_WEIGHT` | `0.45` | Uplift 权重 |
| `FUSION_BANDIT_WEIGHT` | `0.55` | Bandit 权重 |

### 5.4 最佳实践建议

1. **API Key 安全**：不要将真实 API Key 提交到版本控制系统，使用 `.env` 文件管理
2. **数据质量**：确保 Olist 数据集完整上传，缺失关键文件将导致初始化失败
3. **参数调优**：根据实际业务规模调整 `KMEANS_CLUSTERS` 和 `RECOMMEND_PRODUCT_NUM`
4. **冷启动**：Router Agent 在样本数少于 200 时会回退到规则引擎

---

## 6. 故障排除

### 6.1 常见问题

| 问题 | 可能原因 | 解决方案 |
|-----|---------|---------|
| `QWEN_API_KEY` 错误 | 未配置或配置错误 | 检查 `.env` 文件中的 `QWEN_API_KEY` |
| 数据加载失败 | CSV 文件路径错误 | 确认 `data/olist/` 目录下文件完整 |
| 向量检索为空 | 知识库文档缺失 | 检查 `knowledge_base/` 目录文件 |
| 用户不存在 | 传入的 customer_id 无效 | 使用 `user_agent.get_sample_users()` 获取有效 ID |
| 内存不足 | 数据集过大 | 考虑分批处理或增加内存 |

### 6.2 调试模式

在代码中添加调试输出：

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 各 Agent 内部已有打印语句，可查看详细执行流程
```

### 6.3 验证数据完整性

```python
from src.utils.data_loader import load_olist_data
data = load_olist_data()
print(data.keys())  # 检查各数据集是否加载成功
```

---

## 7. 项目结构

```
Marketing-Agent-System/
├── main.py                      # 项目主入口
├── config.py                    # 配置文件
├── requirements.txt             # 依赖清单
├── .env.example                 # 环境变量示例
├── README.md                    # 项目文档
├── data/
│   └── olist/                   # Olist 数据集
│       ├── olist_orders_dataset.csv
│       ├── olist_order_items_dataset.csv
│       ├── olist_order_payments_dataset.csv
│       ├── olist_customers_dataset.csv
│       ├── olist_products_dataset.csv
│       └── product_category_name_translation.csv
├── knowledge_base/               # 营销知识库
│   ├── marketing_strategies.md  # 营销策略
│   └── copywriting_templates.md  # 文案模板
└── src/
    ├── agents/                  # 智能体模块
    │   ├── user_analysis_agent.py
    │   ├── product_recommendation_agent.py
    │   ├── strategy_agent.py
    │   ├── content_generation_agent.py
    │   ├── evaluation_agent.py
    │   └── router_agent.py
    └── utils/                   # 工具模块
        ├── data_loader.py
        ├── feature_engineering.py
        ├── llm_utils.py
        ├── rag_service.py
        ├── bandit_suite.py
        └── offer_catalog.py
```

---

## 8. 技术栈

| 类别 | 技术 |
|-----|------|
| **工作流编排** | LangGraph |
| **LLM** | 通义千问 (Tongyi Qianwen) |
| **向量检索** | FAISS |
| **机器学习** | Scikit-learn, XGBoost |
| **数据处理** | Pandas, NumPy |
| **环境管理** | python-dotenv |

---

## 9. 贡献指南

### 9.1 代码提交规范

提交信息格式：
```
<type>: <subject>

<body>
```

**Type 类型**：
- `feat`: 新功能
- `fix`: 问题修复
- `docs`: 文档更新
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建/工具变更

**示例**：
```
feat: 添加用户生命周期价值预测模块

- 新增 LTV 预测模型
- 集成到评估 Agent
- 添加单元测试
```

### 9.2 Pull Request 流程

1. **Fork 仓库** → 创建特性分支
2. **开发** → 编写代码和测试
3. **提交** → 遵循提交规范
4. **Push** → 推送到远程分支
5. **PR** → 创建 Pull Request
6. **Review** → 代码审查
7. **Merge** → 合并到主分支

### 9.3 文档更新要求

- 新增功能需同步更新 README.md
- API 变更需更新相关说明
- 代码注释使用中文
- 文档中的路径使用相对路径

### 9.4 开发环境检查

提交前请确保：
- [ ] 代码通过现有测试
- [ ] 新增功能有对应测试
- [ ] 文档已更新
- [ ] 无 lint 错误

---

## 10. 扩展方向

- 策略与文案输出标准化为严格 JSON Schema
- 引入在线 A/B Test 反馈闭环
- 将评估模型替换为真实业务标签训练
- 增加多渠道模板（短信/邮件/Push）自动适配
- 将知识库更新与向量索引构建流程自动化
- 支持更多 LLM 提供商（OpenAI、Claude 等）

---

*本项目基于 Olist 巴西电商数据集构建，数据来源：[Kaggle Olist Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)*
