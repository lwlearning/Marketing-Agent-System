import pandas as pd
import numpy as np
import os
import re
import joblib
import config
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.ensemble import HistGradientBoostingRegressor


class EvaluationAgent:
    def __init__(self, orders, order_items):
        self.SMS_COST = 0.1
        self.DEFAULT_FREIGHT_RATIO = 0.1

        # 架构标准：根目录models，与agents同级
        self.model_path = config.CONVERSION_MODEL_PATH
        os.makedirs(config.MODEL_DIR, exist_ok=True)

        print("[效果评估Agent] 正在初始化转化率预测模型...")
        self.orders = orders.dropna(subset=["order_total", "item_count", "freight_total"])
        self.order_items = order_items

        # 模型加载：存在就读取，不存在才训练
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print("[效果评估Agent] ✅ 本地模型加载成功，跳过训练")
        else:
            print("[效果评估Agent] ⚠️ 未找到模型，开始训练...")
            self.model = self._train_conversion_model()

        print("[效果评估Agent] 模型初始化完成\n")

    def _train_conversion_model(self):
        """原生回归模型，预测转化率（0~1连续值）"""
        # 构造特征
        features = self.order_items.merge(
            self.orders[["order_id", "order_purchase_timestamp", "customer_unique_id",
                         "order_total", "item_count", "freight_total"]],
            on="order_id", how="inner"
        ).dropna(subset=["order_total", "item_count", "freight_total"])

        features["recency"] = (
                features["order_purchase_timestamp"].max() - features["order_purchase_timestamp"]).dt.days

        # 你的核心逻辑：热门商品负采样
        product_sales = self.order_items["product_id"].value_counts().head(300).index.tolist()
        hot_products = set(product_sales)
        pos_samples = features.copy()
        pos_samples["label"] = 1

        neg_list = []
        user_list = pos_samples["customer_unique_id"].unique()
        user_purchased_products = pos_samples.groupby("customer_unique_id")["product_id"].apply(set).to_dict()

        for user_id in user_list:
            bought_prods = user_purchased_products.get(user_id, set())
            candidate_neg_prods = list(hot_products - bought_prods)
            if not candidate_neg_prods:
                continue
            select_neg_prods = np.random.choice(candidate_neg_prods, size=3, replace=False)
            for prod in select_neg_prods:
                user_feat = pos_samples[pos_samples["customer_unique_id"] == user_id].mean(numeric_only=True)
                neg_list.append({
                    "recency": user_feat["recency"],
                    "order_total": user_feat["order_total"],
                    "item_count": user_feat["item_count"],
                    "freight_total": user_feat["freight_total"],
                    "customer_unique_id": user_id,
                    "product_id": prod,
                    "label": 0
                })

        neg_samples = pd.DataFrame(neg_list)
        final_df = pd.concat([pos_samples, neg_samples], ignore_index=True)

        # 样本日志
        pos_count = len(final_df[final_df["label"] == 1])
        neg_count = len(final_df[final_df["label"] == 0])
        print(f"[效果评估Agent] 正样本（真实购买）：{pos_count}")
        print(f"[效果评估Agent] 负样本（热门未购买）：{neg_count}")
        print(f"[效果评估Agent] 总训练样本：{pos_count + neg_count}")

        # 训练数据
        X = final_df[["recency", "order_total", "item_count", "freight_total"]]
        y = final_df["label"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # 🔥 原生模型，零依赖、不报错、精度拉满
        model = HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )
        model.fit(X_train, y_train)

        # 双指标评估（符合你的业务需求）
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)

        print(f"[效果评估Agent] 模型 MAE: {mae:.4f}")
        print(f"[效果评估Agent] 模型 AUC: {auc:.4f}")

        # 保存模型
        joblib.dump(model, self.model_path)
        print("[效果评估Agent] ✅ 模型已保存到本地")

        return model

    def _extract_discount_amount(self, offer_text: str, avg_order_value: float) -> float:
        if not offer_text or not avg_order_value or avg_order_value <= 0:
            return 5.0
        full_reduce_match = re.search(r"满\d+减(\d+\.?\d*)", offer_text)
        if full_reduce_match:
            return float(full_reduce_match.group(1))
        reduce_match = re.search(r"减(\d+\.?\d*)", offer_text)
        if reduce_match:
            return float(reduce_match.group(1))
        discount_match = re.search(r"(\d+\.?\d*)折", offer_text)
        if discount_match:
            discount_num = float(discount_match.group(1))
            discount_num = max(0.1, min(10, discount_num))
            discount_rate = discount_num / 10
            discount_amount = avg_order_value * (1 - discount_rate)
            return round(discount_amount, 2)
        return 5.0

    def evaluate(self, user_profile, strategy, content):
        print("[效果评估Agent] 正在评估营销效果...")
        features = pd.DataFrame([[
            user_profile["recency"],
            user_profile["avg_order_value"],
            user_profile["avg_items_per_order"],
            user_profile["avg_order_value"] * self.DEFAULT_FREIGHT_RATIO
        ]], columns=["recency", "order_total", "item_count", "freight_total"])

        base_conversion = self.model.predict(features)[0]
        base_conversion = max(0.001, min(0.5, base_conversion))

        segment_multipliers = {
            "高价值忠诚用户": 1.5, "潜力增长用户": 1.3, "新用户": 1.2,
            "一般价值用户": 1.0, "流失风险用户": 0.8
        }
        segment = user_profile["user_segment"]
        multiplier = segment_multipliers.get(segment, 1.0)

        offer = strategy.get("优惠类型和力度", "")

        configured_discount_cost = strategy.get("优惠成本", None)
        if configured_discount_cost is None:
            discount_cost = self._extract_discount_amount(offer, user_profile["avg_order_value"])
        else:
            try:
                discount_cost = float(configured_discount_cost)
            except (ValueError, TypeError):
                discount_cost = self._extract_discount_amount(offer, user_profile["avg_order_value"])

        configured_channel_cost = strategy.get("触达渠道成本", self.SMS_COST)
        try:
            channel_cost = float(configured_channel_cost)
        except (ValueError, TypeError):
            channel_cost = self.SMS_COST

        discount_cost = max(0.0, discount_cost)
        channel_cost = max(0.0, channel_cost)
        total_marketing_cost = discount_cost + channel_cost

        predicted_conversion = base_conversion * multiplier
        predicted_conversion = min(0.8, predicted_conversion)
        incremental_revenue = user_profile["avg_order_value"] * (predicted_conversion - base_conversion)
        roi = incremental_revenue / total_marketing_cost if total_marketing_cost > 0 else 0.0
        roi = round(roi, 2)
        improve_rate = (predicted_conversion - base_conversion) / base_conversion

        report = {
            "用户分层": segment, "基础转化率(自然)": f"{base_conversion:.2%}",
            "营销后预测转化率": f"{predicted_conversion:.2%}", "转化率提升幅度": f"{improve_rate:.2%}",
            "单用户营销成本": f"¥{total_marketing_cost:.2f}", "营销增量营收": f"¥{incremental_revenue:.2f}",
            "✅ 营销ROI": f"{roi:.2f}",
            "建议": "建议立即上线测试" if roi >= 0.9 and predicted_conversion > 0.05 else "建议优化优惠力度后再测试"
        }

        print(f"[效果评估Agent] 效果评估完成：")
        print(f"  - 基础转化率：{report['基础转化率(自然)']}")
        print(f"  - 预测转化率：{report['营销后预测转化率']}")
        print(f"  - 增量营收：{report['营销增量营收']}")
        print(f"  - 营销成本：{report['单用户营销成本']}")
        print(f"  - ✅ 营销ROI：{report['✅ 营销ROI']}\n")
        return report