from src.utils.data_loader import load_olist_data
from src.utils.feature_engineering import calculate_rfm_features, kmeans_user_segmentation, get_user_preferences


class UserAnalysisAgent:
    def __init__(self):
        print("[用户分析Agent] 正在加载数据集...")
        # 直接调用，无需传任何路径！全自动用config绝对路径
        self.data = load_olist_data()
        self.orders = self.data["orders"]
        self.order_items = self.data["order_items"]
        self.products = self.data["products"]

        print("[用户分析Agent] 正在计算RFM特征并进行用户分层...")
        self.rfm = calculate_rfm_features(self.orders)
        self.rfm, self.scaler, self.kmeans = kmeans_user_segmentation(self.rfm)

        # 打印分层统计
        segment_counts = self.rfm["user_segment"].value_counts()
        print(f"[用户分析Agent] 用户分层完成：\n{segment_counts}\n")

    def analyze_user(self, customer_unique_id):
        """分析单个用户"""
        print(f"[用户分析Agent] 正在分析用户 {customer_unique_id}...")

        # 获取用户RFM信息
        user_rfm = self.rfm[self.rfm["customer_unique_id"] == customer_unique_id]

        if len(user_rfm) == 0:
            return {"error": "用户不存在"}

        user_rfm = user_rfm.iloc[0].to_dict()

        # 获取用户商品偏好
        user_preferences = get_user_preferences(
            customer_unique_id,
            self.order_items.merge(self.orders[["order_id", "customer_unique_id"]], on="order_id"),
            self.products
        )

        # 合并用户信息
        user_profile = {**user_rfm, **user_preferences}

        print(f"[用户分析Agent] 用户分析完成：")
        print(f"  - 用户分层：{user_profile['user_segment']}")
        print(f"  - 最近购买：{user_profile['recency']}天前")
        print(f"  - 购买次数：{user_profile['frequency']}次")
        print(f"  - 总消费：${user_profile['monetary']:.2f}")
        print(f"  - 最喜欢的品类：{user_profile['favorite_categories']}\n")

        return user_profile

    def get_sample_users(self, segment=None, n=5):
        """获取样本用户ID（容错版：指定分层不存在时自动回退）"""
        if segment:
            # 先检查指定的分层是否存在
            segment_users = self.rfm[self.rfm["user_segment"] == segment]
            if len(segment_users) >= n:
                sample = segment_users.sample(n)
            else:
                # 分层不存在或数量不够，回退到随机采样所有用户
                print(f"[用户分析Agent] 分层「{segment}」不存在或数量不足，回退到随机采样")
                sample = self.rfm.sample(n)
        else:
            sample = self.rfm.sample(n)

        return sample["customer_unique_id"].tolist()