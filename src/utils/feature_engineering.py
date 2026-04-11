import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def calculate_rfm_features(orders):
    """计算完整的RFM特征"""
    # 计算参考日期（数据集中最后一个订单日期+1天）
    reference_date = orders["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

    # 按用户聚合计算RFM
    rfm = orders.groupby("customer_unique_id").agg(
        recency=("order_purchase_timestamp", lambda x: (reference_date - x.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("order_total", "sum"),
        avg_order_value=("order_total", "mean"),
        total_items=("item_count", "sum"),
        avg_items_per_order=("item_count", "mean"),
        first_purchase=("order_purchase_timestamp", "min"),
        last_purchase=("order_purchase_timestamp", "max")
    ).reset_index()

    # 计算客户生命周期（天）
    rfm["customer_lifetime"] = (rfm["last_purchase"] - rfm["first_purchase"]).dt.days

    return rfm


def kmeans_user_segmentation(rfm, n_clusters=5):
    """使用K-Means进行用户分层"""
    # 选择用于聚类的特征
    features = ["recency", "frequency", "monetary", "avg_order_value"]

    # 标准化
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[features])

    # K-Means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

    # 给聚类命名（基于特征均值）
    cluster_means = rfm.groupby("cluster")[features].mean()

    # 定义用户分层名称
    def name_cluster(row):
        if row["recency"] < 30 and row["frequency"] > 2 and row["monetary"] > 200:
            return "高价值忠诚用户"
        elif row["recency"] < 60 and row["frequency"] > 1 and row["monetary"] > 100:
            return "潜力增长用户"
        elif row["recency"] < 90 and row["frequency"] == 1:
            return "新用户"
        elif row["recency"] < 180 and row["monetary"] > 50:
            return "一般价值用户"
        else:
            return "流失风险用户"

    cluster_names = cluster_means.apply(name_cluster, axis=1).to_dict()
    rfm["user_segment"] = rfm["cluster"].map(cluster_names)

    return rfm, scaler, kmeans


def get_user_preferences(customer_unique_id, order_items, products):
    """获取用户的商品偏好"""
    # 获取该用户的所有订单
    user_orders = order_items[order_items["customer_unique_id"] == customer_unique_id]

    if len(user_orders) == 0:
        return {"favorite_categories": [], "avg_price": 0}

    # 关联商品信息
    user_orders = user_orders.merge(products, on="product_id", how="left")

    # 计算最喜欢的品类
    category_counts = user_orders["product_category_name_english"].value_counts()
    favorite_categories = category_counts.head(3).index.tolist()

    # 计算平均购买价格
    avg_price = user_orders["price"].mean()

    return {
        "favorite_categories": favorite_categories,
        "avg_price": avg_price,
        "total_spent": user_orders["price"].sum(),
        "purchase_count": len(user_orders)
    }