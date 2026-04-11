import pandas as pd
import os
import config  # 导入配置


def load_olist_data():
    """加载并关联Olist所有核心表（使用config绝对路径）"""
    # 直接用config里的绝对路径！！！
    data_dir = config.DATA_DIR

    # 加载所有表
    orders = pd.read_csv(os.path.join(data_dir, "olist_orders_dataset.csv"))
    order_items = pd.read_csv(os.path.join(data_dir, "olist_order_items_dataset.csv"))
    order_payments = pd.read_csv(os.path.join(data_dir, "olist_order_payments_dataset.csv"))
    customers = pd.read_csv(os.path.join(data_dir, "olist_customers_dataset.csv"))
    products = pd.read_csv(os.path.join(data_dir, "olist_products_dataset.csv"))
    category_translation = pd.read_csv(os.path.join(data_dir, "product_category_name_translation.csv"))

    # 关联商品品类名称
    products = products.merge(category_translation, on="product_category_name", how="left")

    # 计算订单总金额
    order_totals = order_items.groupby("order_id").agg(
        order_total=("price", "sum"),
        freight_total=("freight_value", "sum"),
        item_count=("order_item_id", "count")
    ).reset_index()

    # 关联订单信息
    orders = orders.merge(order_totals, on="order_id", how="left")
    orders = orders.merge(customers, on="customer_id", how="left")

    # 过滤已完成订单
    completed_orders = orders[orders["order_status"] == "delivered"].copy()

    # 转换时间格式
    time_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_customer_date"
    ]
    for col in time_columns:
        completed_orders[col] = pd.to_datetime(completed_orders[col])

    return {
        "orders": completed_orders,
        "order_items": order_items,
        "products": products,
        "customers": customers
    }