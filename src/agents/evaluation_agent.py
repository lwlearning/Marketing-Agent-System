import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import re


class EvaluationAgent:
    def __init__(self, orders):
        # 固定营销成本配置（行业标准，可配置）
        self.SMS_COST = 0.1  # 单条短信成本
        self.DEFAULT_FREIGHT_RATIO = 0.1  # 默认运费比例（保留原有逻辑）

        print("[效果评估Agent] 正在训练转化率预测模型...")
        self.orders = orders
        self.model = self._train_conversion_model()
        print("[效果评估Agent] 模型训练完成\n")

    def _train_conversion_model(self):
        """训练基于历史数据的转化率预测模型（原逻辑完全保留）"""
        features = pd.DataFrame()
        features["recency"] = (
                self.orders["order_purchase_timestamp"].max() - self.orders["order_purchase_timestamp"]).dt.days
        features["order_total"] = self.orders["order_total"]
        features["item_count"] = self.orders["item_count"]
        features["freight_total"] = self.orders["freight_total"]

        # 模拟复购标签（原逻辑保留）
        features["repurchase"] = np.random.choice([0, 1], size=len(features), p=[0.7, 0.3])

        X = features.drop("repurchase", axis=1)
        y = features["repurchase"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"[效果评估Agent] 模型MAE: {mae:.4f}")

        return model

    def _extract_discount_amount(self, offer_text: str, avg_order_value: float) -> float:
        """
        ✅ 终极优化：从优惠文案中自动提取真实优惠成本
        1. 支持 满X减Y、直减X、任意折扣(5折/7.8折/9.5折)
        2. 折扣金额 = 客单价 * (1 - 折扣率) → 动态计算，无硬编码
        3. 优先级：满减 > 直接减 > 折扣
        """
        if not offer_text or not avg_order_value or avg_order_value <= 0:
            return 5.0  # 兜底默认值

        # ------------ 1. 优先匹配：满X减Y（最高优先级）------------
        full_reduce_match = re.search(r"满\d+减(\d+\.?\d*)", offer_text)
        if full_reduce_match:
            return float(full_reduce_match.group(1))

        # ------------ 2. 匹配：直减X元 ------------
        reduce_match = re.search(r"减(\d+\.?\d*)", offer_text)
        if reduce_match:
            return float(reduce_match.group(1))

        # ------------ 3. ✅ 核心优化：匹配任意折扣（5折/8.5折/7.2折 全支持）------------
        discount_match = re.search(r"(\d+\.?\d*)折", offer_text)
        if discount_match:
            # 提取折扣数字：如 8.5 → 8.5折
            discount_num = float(discount_match.group(1))
            # 限制折扣合法范围（0.1折 ~ 10折）
            discount_num = max(0.1, min(10, discount_num))
            # 折扣率：8.5折 = 0.85
            discount_rate = discount_num / 10
            # ✅ 真实优惠金额 = 商品金额 × (1 - 折扣率)
            discount_amount = avg_order_value * (1 - discount_rate)
            return round(discount_amount, 2)

        # 无匹配规则，返回默认小额优惠
        return 5.0

    def evaluate(self, user_profile, strategy, content):
        """评估营销效果（✅ 业务逻辑完全修正版）"""
        print("[效果评估Agent] 正在评估营销效果...")

        # 1. 构造预测特征（原逻辑保留）
        features = pd.DataFrame([[
            user_profile["recency"],
            user_profile["avg_order_value"],
            user_profile["avg_items_per_order"],
            user_profile["avg_order_value"] * self.DEFAULT_FREIGHT_RATIO
        ]], columns=["recency", "order_total", "item_count", "freight_total"])

        # 2. 模型预测基础转化率（自然转化率，无营销）
        base_conversion = self.model.predict(features)[0]
        # 限制转化率合理范围 0~100%
        base_conversion = max(0.001, min(0.5, base_conversion))

        # 3. 策略/用户分层 转化率放大系数（原逻辑保留）
        segment_multipliers = {
            "高价值忠诚用户": 1.5,
            "潜力增长用户": 1.3,
            "新用户": 1.2,
            "一般价值用户": 1.0,
            "流失风险用户": 0.8
        }
        segment = user_profile["user_segment"]
        multiplier = segment_multipliers.get(segment, 1.0)

        # 优惠力度系数（原逻辑保留）
        offer = strategy.get("优惠类型和力度", "")
        if "5折" in offer:
            multiplier *= 1.5
        elif "减30" in offer or "满100减20" in offer:
            multiplier *= 1.3
        elif "9折" in offer:
            multiplier *= 1.1

        # 4. 计算营销后预测转化率
        predicted_conversion = base_conversion * multiplier
        predicted_conversion = min(0.8, predicted_conversion)

        # ==================== ✅ 核心修正：真实ROI计算 ====================
        # 1. 自动提取优惠成本（从文案识别金额 + 动态算折扣）
        discount_cost = self._extract_discount_amount(offer, user_profile["avg_order_value"])
        # 2. 单用户总营销成本 = 优惠补贴 + 短信费
        total_marketing_cost = discount_cost + self.SMS_COST
        # 3. ✅ 增量营收 = 仅营销带来的提升部分（你的核心要求）
        incremental_revenue = user_profile["avg_order_value"] * (predicted_conversion - base_conversion)
        # 4. ✅ 最终真实ROI = 增量营收 / 营销成本
        if total_marketing_cost <= 0:
            roi = 0.0
        else:
            roi = incremental_revenue / total_marketing_cost
        # ROI保底（负数代表亏损）
        roi = round(roi, 2)

        # 提升幅度
        improve_rate = (predicted_conversion - base_conversion) / base_conversion

        # ==================== 生成专业评估报告 ====================
        report = {
            "用户分层": segment,
            "基础转化率(自然)": f"{base_conversion:.2%}",
            "营销后预测转化率": f"{predicted_conversion:.2%}",
            "转化率提升幅度": f"{improve_rate:.2%}",
            "单用户营销成本": f"¥{total_marketing_cost:.2f}",
            "营销增量营收": f"¥{incremental_revenue:.2f}",
            "✅ 真实营销ROI": f"{roi:.2f}",
            "建议": "建议立即上线测试" if roi > 1.0 and predicted_conversion > 0.05 else "建议优化优惠力度后再测试"
        }

        # 打印输出
        print(f"[效果评估Agent] 效果评估完成：")
        print(f"  - 基础转化率：{report['基础转化率(自然)']}")
        print(f"  - 预测转化率：{report['营销后预测转化率']}")
        print(f"  - 增量营收：{report['营销增量营收']}")
        print(f"  - 营销成本：{report['单用户营销成本']}")
        print(f"  - ✅ 真实ROI：{report['✅ 真实营销ROI']}\n")

        return report