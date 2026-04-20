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

# ====================== 4. 安全校验 ======================
if not QWEN_API_KEY:
    raise ValueError("请在 .env 文件中填写你的通义千问 QWEN_API_KEY")