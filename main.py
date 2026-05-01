import random
from typing import Dict, Any, List

import numpy as np
import config
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

from src.agents.user_analysis_agent import UserAnalysisAgent
from src.agents.product_recommendation_agent import ProductRecommendationAgent
from src.agents.strategy_agent import StrategyAgent
from src.agents.content_generation_agent import ContentGenerationAgent
from src.agents.evaluation_agent import EvaluationAgent
from src.agents.router_agent import RouterAgent
from src.utils.bandit_suite import prepare_offline_bandit_suite
from src.utils.offer_catalog import resolve_offer_by_action


class State(BaseModel):
    round_id: int = -1
    routing_mode: str = "auto"

    customer_unique_id: str = ""
    user_profile: Dict[str, Any] = Field(default_factory=dict)
    recommended_products: List[Dict[str, Any]] = Field(default_factory=list)

    routed_tool: str = ""
    router_meta: Dict[str, Any] = Field(default_factory=dict)
    tool_scores: Dict[str, Any] = Field(default_factory=dict)
    selected_action: int = -1
    selected_offer: str = ""
    selected_offer_config: Dict[str, Any] = Field(default_factory=dict)

    strategy: Dict[str, Any] = Field(default_factory=dict)
    content: str = ""
    ope_report: Dict[str, Any] = Field(default_factory=dict)
    evaluation_report: Dict[str, Any] = Field(default_factory=dict)


def action_to_offer(action_id: int) -> str:
    offer_templates = [
        "满100减20 + 包邮",
        "限时9折 + 会员积分翻倍",
        "新人立减30元 + 首单免邮",
        "第二件8折 + 爆款加购优惠",
        "沉睡召回5折券 + 专属礼包"
    ]
    return offer_templates[action_id % len(offer_templates)]


print("=" * 60)
print("营销自动化多智能体系统启动（Router Agent + IPS/SNIPS/DR + Contextual/Uplift/RL/Rule）")
print("=" * 60)

print("[系统] 初始化原有营销Agent...")
user_agent = UserAnalysisAgent()
product_agent = ProductRecommendationAgent(user_agent.order_items, user_agent.products)
strategy_agent = StrategyAgent()
content_agent = ContentGenerationAgent()
eval_agent = EvaluationAgent(user_agent.orders, user_agent.order_items)

print("[系统] 构建离线Bandit反馈与策略评估套件...")
bandit_suite = prepare_offline_bandit_suite(
    rfm=user_agent.rfm,
    n_actions=config.BANDIT_N_ACTIONS,
    random_state=config.BANDIT_RANDOM_SEED,
    min_pscore=config.BANDIT_MIN_PSCORE,
    alpha=config.CONTEXTUAL_BANDIT_ALPHA,
    uplift_weight=config.FUSION_UPLIFT_WEIGHT,
    bandit_weight=config.FUSION_BANDIT_WEIGHT
)

bandit_feedback = bandit_suite["feedback"]
contexts = np.asarray(bandit_feedback["context"], dtype=float)
actions = np.asarray(bandit_feedback["action"], dtype=int)
rewards = np.asarray(bandit_feedback["reward"], dtype=float)
pscores = np.asarray(bandit_feedback["pscore"], dtype=float)
customer_ids = list(bandit_feedback["customer_unique_id"])
n_rounds = int(bandit_feedback["n_rounds"])

router_agent = RouterAgent(
    bandit_suite=bandit_suite,
    contexts=contexts,
    actions=actions,
    rewards=rewards,
    pscores=pscores,
)

print(f"[系统] 反馈样本数: {n_rounds}, 动作数: {config.BANDIT_N_ACTIONS}")


def build_default_profile(customer_unique_id: str) -> Dict[str, Any]:
    return {
        "customer_unique_id": customer_unique_id,
        "user_segment": "一般价值用户",
        "recency": 30,
        "frequency": 1,
        "monetary": 100.0,
        "avg_order_value": 100.0,
        "avg_items_per_order": 1.0,
        "favorite_categories": [],
        "avg_price": 100.0
    }


def build_user_profile(customer_unique_id: str, round_id: int) -> Dict[str, Any]:
    profile = user_agent.analyze_user(customer_unique_id)
    if isinstance(profile, dict) and "error" not in profile:
        user_profile = profile
    else:
        fallback = user_agent.rfm[user_agent.rfm["customer_unique_id"] == customer_unique_id]
        if len(fallback) == 0:
            user_profile = build_default_profile(customer_unique_id)
        else:
            row = fallback.iloc[0].to_dict()
            user_profile = {
                "customer_unique_id": customer_unique_id,
                "user_segment": row.get("user_segment", "一般价值用户"),
                "recency": int(row.get("recency", 30)),
                "frequency": int(row.get("frequency", 1)),
                "monetary": float(row.get("monetary", 100.0)),
                "avg_order_value": float(row.get("avg_order_value", 100.0)),
                "avg_items_per_order": float(row.get("avg_items_per_order", 1.0)),
                "favorite_categories": [],
                "avg_price": float(row.get("avg_order_value", 100.0)),
            }

    user_profile.setdefault("favorite_categories", [])
    user_profile.setdefault("avg_items_per_order", 1.0)
    user_profile.setdefault("avg_order_value", max(1.0, float(user_profile.get("monetary", 100.0))))
    user_profile.setdefault("avg_price", float(user_profile.get("avg_order_value", 100.0)))

    user_profile["obd_context"] = contexts[round_id].tolist()
    user_profile["obd_logged_action"] = int(actions[round_id])
    user_profile["obd_logged_reward"] = float(rewards[round_id])
    user_profile["obd_pscore"] = float(pscores[round_id])

    return user_profile


def load_bandit_input(state: State):
    round_id = state.round_id
    if round_id < 0 or round_id >= n_rounds:
        round_id = random.randint(0, n_rounds - 1)

    customer_unique_id = customer_ids[round_id]
    user_profile = build_user_profile(customer_unique_id, round_id)

    print(
        f"[Bandit输入节点] round_id={round_id}, "
        f"logged_action={user_profile['obd_logged_action']}, "
        f"logged_reward={user_profile['obd_logged_reward']:.4f}, "
        f"pscore={user_profile['obd_pscore']:.4f}"
    )

    return {
        "round_id": round_id,
        "customer_unique_id": customer_unique_id,
        "user_profile": user_profile
    }


def recommend_products(state: State):
    recommended_products = product_agent.recommend_products(state.user_profile)
    return {"recommended_products": recommended_products}


def route_tool(state: State):
    router_decision = router_agent.decide(
        routing_mode=state.routing_mode,
        user_profile=state.user_profile,
        round_id=state.round_id
    )
    routed_tool = router_decision["policy"]

    print(
        f"[Router节点] 选择策略: {routed_tool}, "
        f"confidence={router_decision.get('confidence', 0.0)}, "
        f"reasons={router_decision.get('reason_codes', [])}"
    )
    return {
        "routed_tool": routed_tool,
        "router_meta": router_decision
    }


def apply_policy(state: State, policy_name: str):
    policy_decision = router_agent.act(
        policy_name=policy_name,
        round_id=state.round_id,
        user_profile=state.user_profile
    )

    selected_action = int(policy_decision["selected_action"])
    router_meta = state.router_meta or {}

    try:
        offer_config = resolve_offer_by_action(
            action_id=selected_action,
            round_id=state.round_id,
            customer_unique_id=state.customer_unique_id,
            routed_tool=policy_decision["policy"],
            user_segment=str(state.user_profile.get("user_segment", "")),
        )
        selected_offer = str(offer_config.get("text") or action_to_offer(selected_action))
    except Exception:
        selected_offer = action_to_offer(selected_action)
        offer_config = {
            "offer_id": f"fallback_action_{selected_action}",
            "action_id": selected_action,
            "action_bucket": selected_action,
            "text": selected_offer,
            "template_id": "fallback",
            "ab_group": "fallback",
            "channels": [],
            "audience_segments": [],
            "frequency_cap": {},
            "estimated_discount_cost": 0.0,
            "estimated_channel_cost": 0.0,
            "estimated_total_cost": 0.0,
        }

    return {
        "selected_action": selected_action,
        "selected_offer": selected_offer,
        "selected_offer_config": offer_config,
        "tool_scores": {
            "method": policy_decision["policy"],
            "policy_score": policy_decision["policy_score"],
            "top_actions": policy_decision["top_actions"],
            "router_confidence": router_meta.get("confidence", 0.0),
            "router_reason_codes": router_meta.get("reason_codes", []),
            "router_candidate_scores": router_meta.get("candidate_scores", {}),
            "offer_id": offer_config.get("offer_id"),
            "template_id": offer_config.get("template_id"),
            "ab_group": offer_config.get("ab_group"),
            "channels": offer_config.get("channels", []),
            "frequency_cap": offer_config.get("frequency_cap", {}),
            "estimated_total_cost": offer_config.get("estimated_total_cost", 0.0),
            "logged_action": int(state.user_profile.get("obd_logged_action", -1)),
            "logged_reward": float(state.user_profile.get("obd_logged_reward", 0.0)),
            "logged_pscore": round(float(state.user_profile.get("obd_pscore", 0.0)), 6),
        }
    }


def uplift_tool(state: State):
    return apply_policy(state, "uplift")


def contextual_bandit_tool(state: State):
    return apply_policy(state, "contextual_bandit")


def fusion_tool(state: State):
    return apply_policy(state, "fusion")


def rl_policy_tool(state: State):
    return apply_policy(state, "rl_policy")


def rule_engine_tool(state: State):
    return apply_policy(state, "rule_engine")


def route_to_tool_node(state: State):
    return f"{state.routed_tool}_tool"


def generate_strategy(state: State):
    strategy = strategy_agent.generate_strategy(state.user_profile)
    offer_cfg = state.selected_offer_config or {}

    final_offer_text = offer_cfg.get("text") or state.selected_offer or strategy.get("优惠类型和力度", "专属优惠")
    strategy["优惠类型和力度"] = final_offer_text
    strategy["策略来源"] = f"{state.routed_tool} (action={state.selected_action})"

    if offer_cfg:
        channels = offer_cfg.get("channels", [])
        if channels:
            strategy["最佳触达渠道"] = " + ".join(channels)
        strategy["适用人群"] = "、".join(offer_cfg.get("audience_segments", []))
        strategy["频控配置"] = offer_cfg.get("frequency_cap", {})
        strategy["优惠成本"] = float(offer_cfg.get("estimated_discount_cost", 0.0))
        strategy["触达渠道成本"] = float(offer_cfg.get("estimated_channel_cost", 0.0))
        strategy["优惠配置ID"] = offer_cfg.get("offer_id")
        strategy["AB模板ID"] = offer_cfg.get("template_id")

    return {"strategy": strategy}


def generate_content(state: State):
    content = content_agent.generate_content(
        state.user_profile,
        state.strategy,
        state.recommended_products
    )
    return {"content": content}


def ope_estimators(state: State):
    ope_report = dict(router_agent.get_ope_report(state.routed_tool) or {})
    offer_cfg = state.selected_offer_config if isinstance(state.selected_offer_config, dict) else {}

    ope_report["当前样本动作"] = int(state.selected_action)
    ope_report["当前样本优惠"] = state.selected_offer
    ope_report["当前样本优惠ID"] = offer_cfg.get("offer_id")
    ope_report["当前样本模板ID"] = offer_cfg.get("template_id")

    feedback_update = router_agent.apply_ope_feedback(state.routed_tool, ope_report)
    ope_report["Router反馈更新"] = feedback_update

    print(
        f"[OPE节点] {state.routed_tool}: "
        f"IPS={ope_report.get('IPS策略价值估计', 'NA')}, "
        f"SNIPS={ope_report.get('SNIPS策略价值估计', 'NA')}, "
        f"DR={ope_report.get('DR策略价值估计', 'NA')}, "
        f"feedback={feedback_update.get('new_bias', 'NA') if isinstance(feedback_update, dict) else 'NA'}"
    )
    return {"ope_report": ope_report}


def evaluate(state: State):
    evaluation_report = eval_agent.evaluate(
        state.user_profile,
        state.strategy,
        state.content
    )
    evaluation_report["OPE(IPS/SNIPS/DR)"] = state.ope_report
    evaluation_report["工具路由"] = state.routed_tool
    evaluation_report["Router决策"] = state.router_meta
    evaluation_report["策略动作"] = state.selected_action
    evaluation_report["优惠配置"] = state.selected_offer_config
    evaluation_report["工具打分"] = state.tool_scores
    return {"evaluation_report": evaluation_report}


workflow = StateGraph(State)
workflow.add_node("load_bandit_input", load_bandit_input)
workflow.add_node("recommend_products", recommend_products)
workflow.add_node("route_tool", route_tool)
workflow.add_node("uplift_tool", uplift_tool)
workflow.add_node("contextual_bandit_tool", contextual_bandit_tool)
workflow.add_node("fusion_tool", fusion_tool)
workflow.add_node("rl_policy_tool", rl_policy_tool)
workflow.add_node("rule_engine_tool", rule_engine_tool)
workflow.add_node("generate_strategy", generate_strategy)
workflow.add_node("generate_content", generate_content)
workflow.add_node("ope_estimators", ope_estimators)
workflow.add_node("evaluate", evaluate)

workflow.set_entry_point("load_bandit_input")
workflow.add_edge("load_bandit_input", "recommend_products")
workflow.add_edge("recommend_products", "route_tool")

workflow.add_conditional_edges(
    "route_tool",
    route_to_tool_node,
    {
        "uplift_tool": "uplift_tool",
        "contextual_bandit_tool": "contextual_bandit_tool",
        "fusion_tool": "fusion_tool",
        "rl_policy_tool": "rl_policy_tool",
        "rule_engine_tool": "rule_engine_tool",
    }
)

workflow.add_edge("uplift_tool", "generate_strategy")
workflow.add_edge("contextual_bandit_tool", "generate_strategy")
workflow.add_edge("fusion_tool", "generate_strategy")
workflow.add_edge("rl_policy_tool", "generate_strategy")
workflow.add_edge("rule_engine_tool", "generate_strategy")
workflow.add_edge("generate_strategy", "generate_content")
workflow.add_edge("generate_content", "ope_estimators")
workflow.add_edge("ope_estimators", "evaluate")
workflow.add_edge("evaluate", END)

app = workflow.compile()


def main():
    print("-" * 60)
    print("运行方式: routing_mode 可选 auto / uplift / contextual_bandit / fusion / rl_policy / rule_engine")
    print("-" * 60)

    result = app.invoke({
        "round_id": -1,
        "routing_mode": "auto"
    })

    report = result["evaluation_report"]
    ope = report["OPE(IPS/SNIPS/DR)"]
    router_meta = report.get("Router决策", {})
    offer_cfg = report.get("优惠配置", {})

    print("=" * 60)
    print("最终营销方案汇总（Router Agent + Contextual/Uplift/Fusion/RL/Rule + IPS/SNIPS/DR）")
    print("=" * 60)
    print(f"输入用户ID：{result['customer_unique_id']}")
    print(f"路由工具：{result['routed_tool']}")
    print(f"Router置信度：{router_meta.get('confidence', 0.0)}")
    print(f"Router原因：{router_meta.get('reason_codes', [])}")
    print(f"策略动作：{result['selected_action']}")
    print(f"优惠配置ID：{offer_cfg.get('offer_id', 'N/A')}")
    print(f"AB模板ID：{offer_cfg.get('template_id', 'N/A')}")
    print(f"用户分层：{result['user_profile']['user_segment']}")
    print(f"推荐优惠：{result['strategy'].get('优惠类型和力度', '无')}")
    print(f"策略来源：{result['strategy'].get('策略来源', '无')}")

    print("\n推荐商品：")
    for product in result["recommended_products"]:
        print(f"  - {product['product_name']}，价格：${product['price']:.2f}")

    print(f"\n营销文案：{result['content']}")

    print("\nOPE结果：")
    print(f"  - 方法：{ope['方法']}")
    print(f"  - 行为策略平均回报：{ope['行为策略平均回报']}")
    print(f"  - 动作重合率：{ope['动作重合率']}")
    print(f"  - IPS策略价值估计：{ope['IPS策略价值估计']}")
    print(f"  - SNIPS策略价值估计：{ope['SNIPS策略价值估计']}")
    print(f"  - DR策略价值估计：{ope['DR策略价值估计']}")
    print(f"  - 有效样本权重和：{ope['有效样本权重和']}")

    print("\n效果预测：")
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