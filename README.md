# 营销自动化多智能体系统 (Marketing Multi-Agent System)

## 项目简介
这是一个基于大模型和多智能体技术的营销自动化系统，模拟了从用户分析到效果评估的完整营销闭环。

### 核心智能体
1. **用户分析Agent**：基于公开电商数据，对用户进行 RFM 分层和画像
2. **营销策略Agent**：根据用户画像，匹配知识库生成个性化营销策略
3. **文案生成Agent**：调用大模型，生成定制化营销文案
4. **效果评估Agent**：模拟 A/B 测试，输出效果评估报告

## 项目结构
```
Marketing-Agent-System/
├── README.md                  # 项目说明（本文件）
├── requirements.txt           # 依赖包列表
├── config.py                  # 配置文件（API Key 等）
├── data/                      # 数据集存放目录
│   └── README.md              # 数据集下载说明
├── src/
│   ├── __init__.py
│   ├── main.py                # 主程序入口
│   ├── agents/                # 智能体实现
│   │   ├── __init__.py
│   │   ├── user_analysis_agent.py
│   │   ├── strategy_agent.py
│   │   ├── content_generation_agent.py
│   │   └── evaluation_agent.py
│   └── utils/                 # 工具函数
│       ├── __init__.py
│       ├── data_processor.py
│       └── rag_helper.py
└── knowledge_base/            # 营销知识库（RAG 用）
    └── marketing_rules.md
```

## 技术栈
- **编程语言**：Python 3.9+
- **多智能体框架**：LangChain
- **大模型**：OpenAI GPT-4o / 通义千问
- **机器学习**：Pandas, Scikit-learn
- **数据来源**：天池公开淘宝用户行为数据集

## 项目亮点
基于真实电商用户行为数据构建多智能体营销闭环，覆盖用户分析、策略决策、文案生成、效果评估全流程，结合传统用户分层（RFM）与大模型能力实现自动化营销。

## 快速开始

### 1. 环境准备
```bash
git clone https://github.com/lwlearning/Marketing-Agent-System.git
cd Marketing-Agent-System
pip install -r requirements.txt
```

### 2. 配置 API Key
编辑 config.py，填入你的 API Key：
```python
OPENAI_API_KEY = "sk-your-key"
MODEL_NAME = "gpt-4o"
```

### 3. 下载数据集
1. 访问天池淘宝用户行为数据集：https://tianchi.aliyun.com/dataset/dataDetail?dataId=649
2. 下载 UserBehavior.csv
3. 放入 data/ 目录

### 4. 运行项目
```bash
python src/main.py
```

## 运行效果示例
```
==================================================
营销自动化多智能体系统启动
==================================================
[用户分析Agent] 正在分析用户数据...
[用户分析Agent] 用户分层结果：
中价值用户    4500
低价值用户    3200
高价值用户    2300

[策略Agent] 生成的策略：
{'segment': '中价值用户', 'offer': '满199减50优惠券'}

[文案生成Agent] 生成的文案：
尊敬的用户，您专属的满199减50优惠券已到账！

[效果评估Agent] 评估报告：
预计点击率提升 6.23%
```

## 项目亮点
1. 完全公开数据集，合规安全
2. 完整多智能体流程：分析 → 策略 → 生成 → 评估
3. 模块化架构，易于扩展
4. 兼容国内外大模型
5. 开箱即用，可直接运行

## 注意事项
- 请确保 data/UserBehavior.csv 路径正确
- API Key 不要上传到公开仓库