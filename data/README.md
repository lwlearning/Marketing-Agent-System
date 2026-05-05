# Olist 电商数据集说明

本项目使用 **Olist 巴西电商数据集**，来源：[Kaggle Olist Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

## 数据集概述

Olist 是巴西最大的电商平台之一，提供来自约 10 万家商户的真实电商数据。数据集包含 2016 年至 2018 年的订单信息，覆盖用户行为、商品信息、支付信息、物流信息和评价信息等多个维度。

## 文件清单

| 文件名 | 描述 | 记录数（近似） |
|-------|------|--------------|
| `olist_orders_dataset.csv` | 订单主表 | 99,441 |
| `olist_order_items_dataset.csv` | 订单商品明细 | 112,650 |
| `olist_order_payments_dataset.csv` | 订单支付信息 | 103,886 |
| `olist_customers_dataset.csv` | 客户信息 | 99,441 |
| `olist_products_dataset.csv` | 商品信息 | 32,951 |
| `olist_geolocation_dataset.csv` | 地理信息 | 1,000,000 |
| `olist_order_reviews_dataset.csv` | 评价信息 | 100,000 |
| `olist_sellers_dataset.csv` | 卖家信息 | 3,095 |
| `product_category_name_translation.csv` | 品类名称翻译（葡英） | 71 |

## 核心字段说明

### orders
- `order_id`: 订单唯一标识
- `customer_id`: 客户ID
- `order_status`: 订单状态（delivered、shipped、canceled等）
- `order_purchase_timestamp`: 购买时间
- `order_delivered_customer_date`: 实际交付时间

### order_items
- `order_id`: 订单ID
- `product_id`: 商品ID
- `price`: 商品价格
- `freight_value`: 运费

### customers
- `customer_id`: 客户ID
- `customer_unique_id`: 客户唯一标识（用于用户分析）
- `customer_state`: 客户所在州

### products
- `product_id`: 商品ID
- `product_category_name`: 商品品类名称（葡萄牙语）

## 数据获取

从 Kaggle 下载 Olist 数据集：
1. 访问 https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
2. 注册/登录 Kaggle 账号
3. 下载数据集 ZIP 文件
4. 解压并将 CSV 文件放入 `data/olist/` 目录

## 预处理说明

本项目 `src/utils/data_loader.py` 中的 `load_olist_data()` 函数会自动：
- 读取所有 CSV 文件
- 进行基础数据清洗
- 处理日期格式
- 合并相关数据集

## 数据隐私说明

Olist 数据集已对敏感信息进行脱敏处理，包括：
- 客户姓名已移除
- 信用卡号已移除
- 地理坐标已近似到市级
