# 营销自动化多智能体系统 (Marketing Agent System)

一个基于 **LangGraph + 多智能体 + RAG + 机器学习评估** 的电商营销自动化项目，覆盖从用户洞察到策略生成、文案生成和效果评估的完整链路。

---

## 1. 项目概览

本项目围绕“给单个用户生成个性化营销方案”构建了 5 个智能体，并通过工作流编排串联：

1. 用户分析 Agent（RFM + 聚类分层）
2. 商品推荐 Agent（基于品类偏好 + 热销补足）
3. 营销策略 Agent（RAG 检索知识库 + LLM 生成策略）
4. 文案生成 Agent（LLM 生成个性化短信文案）
5. 效果评估 Agent（模型预测转化率与 ROI）

---

## 2. 核心能力

- 基于 Olist 电商数据进行用户特征工程与分层
- 基于用户偏好推荐商品（支持价格区间过滤）
- 基于知识库检索增强（RAG）生成策略
- 结合推荐商品生成营销文案
- 基于历史订单特征估计营销效果（转化率/提升幅度/ROI）
- 使用 LangGraph 构建端到端工作流

---

## 3. 实际项目结构

```text
Marketing-Agent-System/
├── .env
├── config.py
├── requirements.txt
├── README.md
├── data/
│   └── olist/
│       ├── olist_orders_dataset.csv
│       ├── olist_order_items_dataset.csv
│       ├── olist_order_payments_dataset.csv
│       ├── olist_customers_dataset.csv
│       ├── olist_products_dataset.csv
│       └── product_category_name_translation.csv
├── knowledge_base/
│   ├── marketing_rules.md
│   ├── copywriting_templates.md
│   └── marketing_strategies.md
└── src/
    ├── main.py
    ├── agents/
    │   ├── user_analysis_agent.py
    │   ├── product_recommendation_agent.py
    │   ├── strategy_agent.py
    │   ├── content_generation_agent.py
    │   └── evaluation_agent.py
    └── utils/
        ├── data_loader.py
        ├── feature_engineering.py
        └── rag_service.py
```

---

## 4. 技术栈

- Python 3.11（推荐）
- LangGraph
- LangChain / LangChain OpenAI / LangChain Community
- FAISS（向量检索）
- Pandas / NumPy
- Scikit-learn / XGBoost（当前评估逻辑主要使用 RandomForestRegressor）
- python-dotenv

---

## 5. 工作流说明（LangGraph）

工作流入口在 `analyze_user`，依次执行：

`analyze_user -> recommend_products -> generate_strategy -> generate_content -> evaluate -> END`

最终输出包含：

- 用户分层与画像
- 推荐商品列表
- 个性化营销策略
- 营销文案
- 效果预测报告（转化率、提升幅度、ROI、建议）

---

## 6. 环境准备

### 6.1 创建并激活虚拟环境（可选但推荐）

```bash
python -m venv .venv
source .venv/bin/activate
```

### 6.2 安装依赖

```bash
pip install -r requirements.txt
```

---

## 7. 配置说明

### 7.1 配置 `.env`

在项目根目录创建/填写 `.env`：

```env
OPENAI_API_KEY=your-openai-key-here
DASHSCOPE_API_KEY=your-dashscope-key-here
```

### 7.2 配置模型

[config.py](file:///Users/liuwen.107/PycharmProjects/Marketing-Agent-System/config.py) 默认：

- `MODEL_NAME = "gpt-4o"`
- 使用 `OPENAI_API_KEY`

如果切换到其他模型，请同步修改 [config.py](file:///Users/liuwen.107/PycharmProjects/Marketing-Agent-System/config.py) 中模型配置。

---

## 8. 数据准备

请将 Olist 相关 CSV 文件放到 `data/olist/` 目录，至少包含：

- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_order_payments_dataset.csv`
- `olist_customers_dataset.csv`
- `olist_products_dataset.csv`
- `product_category_name_translation.csv`

---

## 9. 运行项目

请在项目根目录运行（推荐方式）：

```bash
python -m src.main
```

---

## 10. 常见问题排查

### 10.1 `ModuleNotFoundError: No module named 'config'`
不要用 `python src/main.py`，请使用：

```bash
python -m src.main
```

### 10.2 `FileNotFoundError`（知识库或数据路径）
项目已在 [config.py](file:///Users/liuwen.107/PycharmProjects/Marketing-Agent-System/config.py) 中使用绝对路径拼接；请确认：

- `knowledge_base/` 存在且包含 `.md` 文件
- `data/olist/` 数据齐全

### 10.3 `Unresolved reference 'langchain_community'` 等
说明当前解释器缺少依赖，执行：

```bash
pip install -r requirements.txt
```

并确认 IDE 使用的是项目虚拟环境解释器。

---

## 11. 输出示例（简化）

```text
============================================================
升级版营销自动化多智能体系统启动
============================================================
[用户分析Agent] ...
[商品推荐Agent] ...
[策略Agent] ...
[文案生成Agent] ...
[效果评估Agent] ...
============================================================
最终营销方案汇总
============================================================
用户ID：xxx
用户分层：高价值忠诚用户
...
```

---

## 12. 后续可扩展方向

- 将策略与文案输出标准化为严格 JSON Schema
- 引入在线 A/B Test 反馈闭环
- 将评估模型替换为真实业务标签训练
- 增加多渠道模板（短信/邮件/Push）自动适配
- 将知识库更新与向量索引构建流程自动化