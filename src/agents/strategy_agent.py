from src.utils.rag_service import RAGService
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import config
from src.utils.llm_utils import parse_llm_output
import json


# -------------------------- Schema --------------------------
class StrategyOutput(BaseModel):
    优惠类型和力度: str = Field(description="优惠策略")
    最佳触达渠道: str
    最佳触达时间: str
    营销重点和话术风格: str
    预期目标: str


# -------------------------- 工具函数 --------------------------
def sanitize_input(data):
    """
    保证传入 LLM 的数据是纯 Python 类型（避免 warning）
    """
    try:
        return json.loads(json.dumps(data))
    except Exception:
        return data


# -------------------------- Strategy Agent --------------------------
class StrategyAgent:
    def __init__(self):
        self.rag = RAGService()

        self.llm = ChatOpenAI(
            model=config.QWEN_MODEL,
            api_key=config.QWEN_API_KEY,
            base_url=config.QWEN_BASE_URL,
            temperature=0.3
        )

        # ✅ prompt 预编译（性能更好）
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的电商营销专家。请根据用户画像和营销知识库，生成个性化的营销策略。"),
            ("user", """
用户画像：
- 用户分层：{user_segment}
- 最近购买：{recency}天前
- 购买次数：{frequency}次
- 总消费：${monetary:.2f}
- 平均订单金额：${avg_order_value:.2f}
- 最喜欢的品类：{favorite_categories}

营销知识库参考：
{relevant_knowledge}

要求：
1. 输出JSON格式
2. 字段必须包含：
   - 优惠类型和力度
   - 最佳触达渠道
   - 最佳触达时间
   - 营销重点和话术风格
   - 预期目标
3. 不要输出解释

直接输出JSON：
            """)
        ])

    def generate_strategy(self, user_profile):
        print("[策略Agent] 正在生成个性化营销策略...")

        # -------------------------- RAG --------------------------
        query = f"针对{user_profile['user_segment']}的营销策略"
        relevant_knowledge = self.rag.retrieve(query)

        knowledge_text = "\n".join([
            f"{i+1}. {k}" for i, k in enumerate(relevant_knowledge)
        ])

        # -------------------------- 输入构造 --------------------------
        inputs = {
            "user_segment": user_profile["user_segment"],
            "recency": user_profile["recency"],
            "frequency": user_profile["frequency"],
            "monetary": user_profile["monetary"],
            "avg_order_value": user_profile["avg_order_value"],
            "favorite_categories": user_profile["favorite_categories"],
            "relevant_knowledge": knowledge_text
        }

        inputs = sanitize_input(inputs)  # ✅ 防止 warning

        # -------------------------- 调用 LLM --------------------------
        try:
            chain = self.prompt | self.llm
            result = chain.invoke(inputs)

            # ✅ 统一解析（核心）
            strategy = parse_llm_output(result, StrategyOutput, strict=False)

            if not strategy:
                raise ValueError("LLM返回无法解析")

        except Exception as e:
            print(f"[策略Agent] LLM生成失败，使用兜底策略: {e}")
            strategy = self._get_default_strategy(user_profile["user_segment"])

        # -------------------------- 输出 --------------------------
        print(f"[策略Agent] 营销策略生成完成：")
        print(f"  - 优惠：{strategy.get('优惠类型和力度', '无')}")
        print(f"  - 渠道：{strategy.get('最佳触达渠道', '无')}")
        print(f"  - 时间：{strategy.get('最佳触达时间', '无')}\n")

        return strategy

    # -------------------------- fallback --------------------------
    def _get_default_strategy(self, user_segment):
        default_strategies = {
            "高价值忠诚用户": {
                "优惠类型和力度": "专属9折优惠券 + 免费配送 + 新品优先体验",
                "最佳触达渠道": "短信 + APP Push + 邮件",
                "最佳触达时间": "工作日晚上8-10点",
                "营销重点和话术风格": "尊贵、专属、个性化",
                "预期目标": "复购率提升15%，客单价提升10%"
            },
            "潜力增长用户": {
                "优惠类型和力度": "满100减20优惠券 + 第二件8折",
                "最佳触达渠道": "APP Push + 短信",
                "最佳触达时间": "周末下午2-6点",
                "营销重点和话术风格": "实惠、性价比、推荐",
                "预期目标": "复购率提升20%，购买频次提升15%"
            },
            "新用户": {
                "优惠类型和力度": "首单立减30元 + 新人专享价",
                "最佳触达渠道": "APP Push + 弹窗",
                "最佳触达时间": "下单后7天内",
                "营销重点和话术风格": "欢迎、福利、引导",
                "预期目标": "首单转化率提升25%，留存率提升20%"
            },
            "一般价值用户": {
                "优惠类型和力度": "满50减10优惠券 + 限时折扣",
                "最佳触达渠道": "APP Push",
                "最佳触达时间": "工作日中午12-2点",
                "营销重点和话术风格": "优惠、便捷",
                "预期目标": "活跃度提升15%，转化率提升10%"
            },
            "流失风险用户": {
                "优惠类型和力度": "回归专属5折优惠券 + 免运费",
                "最佳触达渠道": "短信 + 邮件",
                "最佳触达时间": "流失后30天、60天、90天",
                "营销重点和话术风格": "关怀、召回、惊喜",
                "预期目标": "召回率提升10%"
            }
        }
        return default_strategies.get(user_segment, default_strategies["一般价值用户"])