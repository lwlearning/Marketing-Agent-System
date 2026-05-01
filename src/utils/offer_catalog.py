from copy import deepcopy
from hashlib import md5
from typing import Dict, Any, List

OFFER_CATALOG: List[Dict[str, Any]] = [
    {
        "action_bucket": 0,
        "name": "满减包邮",
        "channels": ["APP Push", "短信"],
        "channel_costs": {"APP Push": 0.02, "短信": 0.10},
        "audience_segments": ["潜力增长用户", "一般价值用户"],
        "frequency_cap": {"window_days": 7, "max_touch": 2, "cooldown_days": 2},
        "discount_cost": 20.0,
        "templates": [
            {"template_id": "A", "ab_group": "A", "text": "满100减20 + 包邮", "discount_cost": 20.0},
            {"template_id": "B", "ab_group": "B", "text": "下单满100立减20，今日包邮", "discount_cost": 20.0},
        ],
    },
    {
        "action_bucket": 1,
        "name": "9折会员激励",
        "channels": ["APP Push", "站内信"],
        "channel_costs": {"APP Push": 0.02, "站内信": 0.01},
        "audience_segments": ["高价值忠诚用户", "潜力增长用户"],
        "frequency_cap": {"window_days": 7, "max_touch": 3, "cooldown_days": 1},
        "discount_cost": 12.0,
        "templates": [
            {"template_id": "A", "ab_group": "A", "text": "限时9折 + 会员积分翻倍", "discount_cost": 12.0},
            {"template_id": "B", "ab_group": "B", "text": "尊享9折，积分双倍返", "discount_cost": 12.0},
        ],
    },
    {
        "action_bucket": 2,
        "name": "新人首单激励",
        "channels": ["APP Push", "弹窗"],
        "channel_costs": {"APP Push": 0.02, "弹窗": 0.01},
        "audience_segments": ["新用户"],
        "frequency_cap": {"window_days": 7, "max_touch": 2, "cooldown_days": 2},
        "discount_cost": 30.0,
        "templates": [
            {"template_id": "A", "ab_group": "A", "text": "新人立减30元 + 首单免邮", "discount_cost": 30.0},
            {"template_id": "B", "ab_group": "B", "text": "欢迎礼：首单减30并包邮", "discount_cost": 30.0},
        ],
    },
    {
        "action_bucket": 3,
        "name": "连带加购激励",
        "channels": ["APP Push", "短信"],
        "channel_costs": {"APP Push": 0.02, "短信": 0.10},
        "audience_segments": ["一般价值用户", "潜力增长用户"],
        "frequency_cap": {"window_days": 7, "max_touch": 2, "cooldown_days": 2},
        "discount_cost": 15.0,
        "templates": [
            {"template_id": "A", "ab_group": "A", "text": "第二件8折 + 爆款加购优惠", "discount_cost": 15.0},
            {"template_id": "B", "ab_group": "B", "text": "加购第二件享8折，爆款立省", "discount_cost": 15.0},
        ],
    },
    {
        "action_bucket": 4,
        "name": "沉睡召回",
        "channels": ["短信", "邮件"],
        "channel_costs": {"短信": 0.10, "邮件": 0.02},
        "audience_segments": ["流失风险用户"],
        "frequency_cap": {"window_days": 14, "max_touch": 2, "cooldown_days": 5},
        "discount_cost": 40.0,
        "templates": [
            {"template_id": "A", "ab_group": "A", "text": "沉睡召回5折券 + 专属礼包", "discount_cost": 40.0},
            {"template_id": "B", "ab_group": "B", "text": "回归专享5折 + 限量礼包", "discount_cost": 40.0},
        ],
    },
]


def _stable_index(key: str, n: int) -> int:
    if n <= 0:
        return 0
    h = md5(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % n


def _get_base_offer(action_id: int) -> Dict[str, Any]:
    if not OFFER_CATALOG:
        raise ValueError("OFFER_CATALOG 为空")
    base = deepcopy(OFFER_CATALOG[action_id % len(OFFER_CATALOG)])
    base["action_id"] = int(action_id)
    base["action_bucket"] = int(action_id % len(OFFER_CATALOG))
    return base


def resolve_offer_by_action(
    action_id: int,
    round_id: int,
    customer_unique_id: str,
    routed_tool: str = "",
    user_segment: str = "",
) -> Dict[str, Any]:
    base = _get_base_offer(action_id)
    templates = base.get("templates", [])
    if not templates:
        templates = [{"template_id": "A", "ab_group": "A", "text": base.get("name", "专属优惠")}]

    key = f"{action_id}|{round_id}|{customer_unique_id}|{routed_tool}|{user_segment}"
    idx = _stable_index(key, len(templates))
    tpl = templates[idx]

    channels = tpl.get("channels") or base.get("channels", [])
    channel_costs = base.get("channel_costs", {})
    discount_cost = float(tpl.get("discount_cost", base.get("discount_cost", 0.0)))
    estimated_channel_cost = float(sum(float(channel_costs.get(ch, 0.0)) for ch in channels))

    offer_id = f"action_{base['action_bucket']}_tpl_{tpl.get('template_id', 'A')}"

    return {
        "offer_id": offer_id,
        "action_id": int(action_id),
        "action_bucket": int(base["action_bucket"]),
        "text": str(tpl.get("text", base.get("name", "专属优惠"))),
        "template_id": str(tpl.get("template_id", "A")),
        "ab_group": str(tpl.get("ab_group", "A")),
        "channels": list(channels),
        "audience_segments": list(base.get("audience_segments", [])),
        "frequency_cap": dict(base.get("frequency_cap", {})),
        "estimated_discount_cost": discount_cost,
        "estimated_channel_cost": estimated_channel_cost,
        "estimated_total_cost": discount_cost + estimated_channel_cost,
    }
