from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import config
import json


def sanitize_input(data):
    try:
        return json.loads(json.dumps(data))
    except:
        return data


class ContentGenerationAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.QWEN_MODEL,
            api_key=config.QWEN_API_KEY,
            base_url=config.QWEN_BASE_URL,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS
        )

        # ✅ prompt 提前编译（更高效）
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的电商文案专家。请根据用户画像、营销策略和商品推荐，生成一条吸引人的营销短信文案。"),
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

要求：
1. 70字以内
2. 必须包含优惠信息
3. 至少包含一个商品
4. 风格符合话术要求
5. 输出纯文本，不要解释
            """)
        ])

    def generate_content(self, user_profile, strategy, recommended_products):
        print("[文案生成Agent] 正在生成个性化营销文案...")

        # ✅ 1. 强制 strategy 转为 dict（防 warning）
        strategy = sanitize_input(strategy)

        # ✅ 2. 构造商品文本
        products_text = "\n".join([
            f"{i+1}. {p['product_name']} - ${p['price']:.2f}"
            for i, p in enumerate(recommended_products)
        ])

        # ✅ 3. 构造输入
        inputs = {
            "user_segment": user_profile["user_segment"],
            "monetary": user_profile["monetary"],
            "favorite_categories": user_profile["favorite_categories"],
            "offer": strategy.get("优惠类型和力度", "专属优惠"),
            "tone": strategy.get("营销重点和话术风格", "友好"),
            "products_text": products_text
        }

        inputs = sanitize_input(inputs)

        # ✅ 4. 调用 LLM + fallback
        try:
            chain = self.prompt | self.llm
            response = chain.invoke(inputs)
            content = response.content.strip()

            if not content:
                raise ValueError("空结果")

        except Exception as e:
            print(f"[文案生成Agent] 生成失败，使用兜底: {e}")
            content = self._fallback_content(strategy, recommended_products)

        print(f"[文案生成Agent] 营销文案生成完成：\n{content}\n")
        return content

    def _fallback_content(self, strategy, products):
        """兜底策略（非常重要）"""
        if not products:
            return f"专属优惠来啦！{strategy.get('优惠类型和力度', '')}，速来选购！"

        p = products[0]
        return f"{strategy.get('优惠类型和力度', '限时优惠')}！{p['product_name']}仅${p['price']:.2f}，快来抢购！"