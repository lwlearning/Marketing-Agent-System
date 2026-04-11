import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


class EvaluationAgent:
    def __init__(self, orders):
        print("[效果评估Agent] 正在训练转化率预测模型...")

        # 准备训练数据
        self.orders = orders
        self.model = self._train_conversion_model()

        print("[效果评估Agent] 模型训练完成\n")

    def _train_conversion_model(self):
        """训练基于历史数据的转化率预测模型"""
        # 提取特征
        features = pd.DataFrame()
        features["recency"] = (
                    self.orders["order_purchase_timestamp"].max() - self.orders["order_purchase_timestamp"]).dt.days
        features["order_total"] = self.orders["order_total"]
        features["item_count"] = self.orders["item_count"]
        features["freight_total"] = self.orders["freight_total"]

        # 目标变量：是否在30天内复购
        # 这里简化处理，实际项目中可以用更复杂的标签
        features["repurchase"] = np.random.choice([0, 1], size=len(features), p=[0.7, 0.3])

        # 训练模型
        X = features.drop("repurchase", axis=1)
        y = features["repurchase"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 评估模型
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"[效果评估Agent] 模型MAE: {mae:.4f}")

        return model

    def evaluate(self, user_profile, strategy, content):
        """评估营销效果"""
        print("[效果评估Agent] 正在评估营销效果...")

        # 准备预测特征
        features = pd.DataFrame([[
            user_profile["recency"],
            user_profile["avg_order_value"],
            user_profile["avg_items_per_order"],
            user_profile["avg_order_value"] * 0.1  # 假设运费是订单金额的10%
        ]], columns=["recency", "order_total", "item_count", "freight_total"])

        # 预测基础转化率
        base_conversion = self.model.predict(features)[0]

        # 根据用户分层和策略调整转化率
        segment_multipliers = {
            "高价值忠诚用户": 1.5,
            "潜力增长用户": 1.3,
            "新用户": 1.2,
            "一般价值用户": 1.0,
            "流失风险用户": 0.8
        }

        segment = user_profile["user_segment"]
        multiplier = segment_multipliers.get(segment, 1.0)

        # 优惠力度调整
        offer = strategy.get("优惠类型和力度", "")
        if "5折" in offer:
            multiplier *= 1.5
        elif "减30" in offer or "满100减20" in offer:
            multiplier *= 1.3
        elif "9折" in offer:
            multiplier *= 1.1

        predicted_conversion = base_conversion * multiplier

        # 生成评估报告
        report = {
            "用户分层": segment,
            "基础转化率": f"{base_conversion:.2%}",
            "预测转化率": f"{predicted_conversion:.2%}",
            "提升幅度": f"{(predicted_conversion - base_conversion) / base_conversion:.2%}",
            "预期ROI": f"{predicted_conversion * user_profile['avg_order_value'] / (user_profile['avg_order_value'] * 0.2):.2f}",
            "建议": "建议立即上线测试" if predicted_conversion > 0.05 else "建议优化文案和优惠力度后再测试"
        }

        print(f"[效果评估Agent] 效果评估完成：")
        print(f"  - 预测转化率：{report['预测转化率']}")
        print(f"  - 提升幅度：{report['提升幅度']}")
        print(f"  - 预期ROI：{report['预期ROI']}\n")

        return report