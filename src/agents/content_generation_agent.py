from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import config


class ContentGenerationAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.MODEL_NAME,
            api_key=config.OPENAI_API_KEY,
            temperature=0.7
        )

    def generate_content(self, user_profile, strategy):
        """生成营销文案"""
        print("[文案生成Agent] 正在生成营销文案...")

        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的营销文案专家。请根据用户画像和营销策略，生成一条个性化的营销短信文案。"),
            ("user", """
用户画像：
- 用户分层：{segment}
- 历史购买次数：{total_purchases}
- 最近购买天数：{days_since_last_purchase}

营销策略：
- 优惠：{offer}
- 触达渠道：{channel}
- 文案风格：{tone}

请生成一条50字以内的短信文案。
            """)
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "segment": strategy['segment'],
            "total_purchases": user_profile['total_purchases'],
            "days_since_last_purchase": user_profile['days_since_last_purchase'],
            "offer": strategy['offer'],
            "channel": strategy['channel'],
            "tone": strategy['tone']
        })

        content = response.content.strip()
        print(f"[文案生成Agent] 生成的文案：\n{content}\n")
        return content