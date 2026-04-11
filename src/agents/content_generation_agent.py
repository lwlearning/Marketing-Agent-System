from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import config


class ContentGenerationAgent:
    def __init__(self):
        # 通义千问兼容模式
        self.llm = ChatOpenAI(
            model=config.QWEN_MODEL,
            api_key=config.QWEN_API_KEY,
            base_url=config.QWEN_BASE_URL,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS
        )

    def generate_content(self, user_profile, strategy, recommended_products):
        print("[文案生成Agent] 正在生成个性化营销文案...")

        products_text = ""
        for i, product in enumerate(recommended_products, 1):
            products_text += f"{i}. {product['product_name']} - 原价${product['price']:.2f}\n"

        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的电商文案专家。请根据用户画像、营销策略和商品推荐，生成一条吸引人的营销短信文案。"),
            # # 2. (可选) assistant：给模型示范回答（少样本学习）
            # ("assistant", "【新人福利】首单立减30！推荐爆款笔记本$29.9，速购~"),
            ("user", """
用户画像：
- 用户分层：{user_segment}
- 总消费：${monetary:.2f}
- 最喜欢的品类：{favorite_categories}

营销策略：
- 优惠：{offer}
- 话术风格：{tone}

推荐商品：
{products_text}

请生成一条70字以内的短信文案，包含优惠信息和至少一个推荐商品。
            """)
        ])

        chain = prompt | self.llm
        response = chain.invoke({
            "user_segment": user_profile["user_segment"],
            "monetary": user_profile["monetary"],
            "favorite_categories": user_profile["favorite_categories"],
            "offer": strategy.get("优惠类型和力度", "专属优惠"),
            "tone": strategy.get("营销重点和话术风格", "友好"),
            "products_text": products_text
        })

        content = response.content.strip()
        print(f"[文案生成Agent] 营销文案生成完成：\n{content}\n")
        return content