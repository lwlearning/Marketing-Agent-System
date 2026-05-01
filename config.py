# -*- coding: utf-8 -*-
"""
营销多智能体系统 - 纯通义千问版配置文件
基于Olist巴西电商数据集
无任何OpenAI依赖
"""
import os
from dotenv import load_dotenv
# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 模型目录
MODEL_DIR = os.path.join(BASE_DIR, "models")
CONVERSION_MODEL_PATH = os.path.join(MODEL_DIR, "conversion_v1.joblib")

# 加载环境变量
load_dotenv()

# ====================== 1. 通义千问 核心配置 ======================
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-turbo")
QWEN_EMBEDDING_MODEL = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v2")

# LLM生成参数
TEMPERATURE = 0.7
MAX_TOKENS = 512

# ====================== 2. 文件路径配置（自动匹配，不用改） ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "olist")          # Olist数据集路径
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "knowledge_base")  # 营销知识库
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db")        # RAG向量库缓存

# ====================== 3. 算法参数 ======================
KMEANS_CLUSTERS = 5
RECOMMEND_PRODUCT_NUM = 3

BANDIT_N_ACTIONS = int(os.getenv("BANDIT_N_ACTIONS", "5"))
BANDIT_RANDOM_SEED = int(os.getenv("BANDIT_RANDOM_SEED", "42"))
BANDIT_MIN_PSCORE = float(os.getenv("BANDIT_MIN_PSCORE", "0.03"))

CONTEXTUAL_BANDIT_ALPHA = float(os.getenv("CONTEXTUAL_BANDIT_ALPHA", "0.6"))
FUSION_UPLIFT_WEIGHT = float(os.getenv("FUSION_UPLIFT_WEIGHT", "0.45"))
FUSION_BANDIT_WEIGHT = float(os.getenv("FUSION_BANDIT_WEIGHT", "0.55"))

ROUTER_LOW_PSCORE = float(os.getenv("ROUTER_LOW_PSCORE", "0.08"))
ROUTER_HIGH_COMPLEXITY = float(os.getenv("ROUTER_HIGH_COMPLEXITY", "0.95"))
ROUTER_CHURN_RECENCY = float(os.getenv("ROUTER_CHURN_RECENCY", "90"))
ROUTER_FORCE_RULE_ENGINE_ON_MISSING_CONTEXT = os.getenv(
    "ROUTER_FORCE_RULE_ENGINE_ON_MISSING_CONTEXT", "true"
).strip().lower() in {"1", "true", "yes", "y", "on"}

ROUTER_MODEL_MIN_CONFIDENCE = float(os.getenv("ROUTER_MODEL_MIN_CONFIDENCE", "0.45"))
ROUTER_FORCE_RULE_ENGINE_PSCORE = float(os.getenv("ROUTER_FORCE_RULE_ENGINE_PSCORE", "0.02"))
ROUTER_MODEL_RANDOM_SEED = int(os.getenv("ROUTER_MODEL_RANDOM_SEED", "42"))
ROUTER_MODEL_MIN_SAMPLES = int(os.getenv("ROUTER_MODEL_MIN_SAMPLES", "200"))

ROUTER_OPE_FEEDBACK_ENABLED = os.getenv(
    "ROUTER_OPE_FEEDBACK_ENABLED", "true"
).strip().lower() in {"1", "true", "yes", "y", "on"}
ROUTER_OPE_FEEDBACK_METRIC = os.getenv("ROUTER_OPE_FEEDBACK_METRIC", "DR")
ROUTER_OPE_FEEDBACK_LR = float(os.getenv("ROUTER_OPE_FEEDBACK_LR", "0.15"))
ROUTER_OPE_FEEDBACK_DECAY = float(os.getenv("ROUTER_OPE_FEEDBACK_DECAY", "0.02"))
ROUTER_OPE_FEEDBACK_CLIP = float(os.getenv("ROUTER_OPE_FEEDBACK_CLIP", "0.25"))
ROUTER_POLICY_TEMPERATURE = float(os.getenv("ROUTER_POLICY_TEMPERATURE", "1.0"))


# ====================== 4. 安全校验 ======================
if not QWEN_API_KEY:
    raise ValueError("请在 .env 文件中填写你的通义千问 QWEN_API_KEY")