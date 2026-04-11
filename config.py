import os
from dotenv import load_dotenv

load_dotenv()

# 大模型配置 (选择一个即可)
# 1. OpenAI GPT-4o
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-key-here")
MODEL_NAME = "gpt-4o"

# 2. 通义千问 (国内推荐)
# DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "your-dashscope-key-here")
# MODEL_NAME = "qwen-max"

# 项目根目录（避免受运行工作目录影响）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 数据路径
DATA_PATH = os.path.join(BASE_DIR, "data", "UserBehavior.csv")
KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, "knowledge_base", "marketing_rules.md")