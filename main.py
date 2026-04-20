import config
from src.agents.user_analysis_agent import UserAnalysisAgent
from src.agents.product_recommendation_agent import ProductRecommendationAgent
from src.agents.strategy_agent import StrategyAgent
from src.agents.content_generation_agent import ContentGenerationAgent
from src.agents.evaluation_agent import EvaluationAgent  # 正确类名
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from typing import Dict, Any, List

# 定义状态（保持不变）
class State(BaseModel):
    customer_unique_id: str
    user_profile: Dict[str, Any] = {}
    recommended_products: List[Dict[str, Any]] = []
    strategy: Dict[str, Any] = {}
    content: str = ""
    evaluation_report: Dict[str, Any] = {}


# 初始化所有Agent
print("=" * 60)
print("升级版营销自动化多智能体系统启动")
print("=" * 60)

user_agent = UserAnalysisAgent()
product_agent = ProductRecommendationAgent(user_agent.order_items, user_agent.products)
strategy_agent = StrategyAgent()
content_agent = ContentGenerationAgent()
eval_agent = EvaluationAgent(user_agent.orders, user_agent.order_items) # 传入订单数据训练模型


# 定义工作流节点（完全不变）
def analyze_user(state: State):
    user_profile = user_agent.analyze_user(state.customer_unique_id)
    return {"user_profile": user_profile}


def recommend_products(state: State):
    recommended_products = product_agent.recommend_products(state.user_profile)
    return {"recommended_products": recommended_products}


def generate_strategy(state: State):
    strategy = strategy_agent.generate_strategy(state.user_profile)
    return {"strategy": strategy}


def generate_content(state: State):
    content = content_agent.generate_content(
        state.user_profile,
        state.strategy,
        state.recommended_products
    )
    return {"content": content}


def evaluate(state: State):
    evaluation_report = eval_agent.evaluate(
        state.user_profile,
        state.strategy,
        state.content
    )
    return {"evaluation_report": evaluation_report}


# 构建工作流
workflow = StateGraph(State)
workflow.add_node("analyze_user", analyze_user)
workflow.add_node("recommend_products", recommend_products)
workflow.add_node("generate_strategy", generate_strategy)
workflow.add_node("generate_content", generate_content)
workflow.add_node("evaluate", evaluate)

workflow.set_entry_point("analyze_user")
workflow.add_edge("analyze_user", "recommend_products")
workflow.add_edge("recommend_products", "generate_strategy")
workflow.add_edge("generate_strategy", "generate_content")
workflow.add_edge("generate_content", "evaluate")
workflow.add_edge("evaluate", END)

app = workflow.compile()


def main():
    print("当前数据中的用户分层：")
    print(user_agent.rfm["user_segment"].value_counts())
    print("-" * 60)

    # 获取样本用户
    sample_users = user_agent.get_sample_users(n=1)
    customer_unique_id = sample_users[0]

    print(f"\n开始处理用户：{customer_unique_id}")
    print("-" * 60)

    # 运行工作流
    result = app.invoke({
        "customer_unique_id": customer_unique_id
    })

    # ==================== 对齐评估报告字段 ====================
    report = result['evaluation_report']

    print("=" * 60)
    print("最终营销方案汇总")
    print("=" * 60)
    print(f"用户ID：{result['customer_unique_id']}")
    print(f"用户分层：{result['user_profile']['user_segment']}")
    print(f"购买次数：{result['user_profile']['frequency']}次")
    print(f"总消费：${result['user_profile']['monetary']:.2f}")
    print(f"\n营销策略：{result['strategy'].get('优惠类型和力度', '无')}")
    print(f"触达渠道：{result['strategy'].get('最佳触达渠道', '无')}")
    print(f"触达时间：{result['strategy'].get('最佳触达时间', '无')}")
    print(f"\n推荐商品：")
    for product in result['recommended_products']:
        print(f"  - {product['product_name']}，价格：${product['price']:.2f}")
    print(f"\n营销文案：{result['content']}")
    print(f"\n效果预测（工业级真实评估）：")
    print(f"  - 基础转化率(自然)：{report['基础转化率(自然)']}")
    print(f"  - 营销后预测转化率：{report['营销后预测转化率']}")
    print(f"  - 转化率提升幅度：{report['转化率提升幅度']}")
    print(f"  - 单用户营销成本：{report['单用户营销成本']}")
    print(f"  - 营销增量营收：{report['营销增量营收']}")
    print(f"  - ✅ 营销ROI：{report['✅ 营销ROI']}")
    print(f"  - 运营建议：{report['建议']}")
    print("=" * 60)


if __name__ == "__main__":
    main()