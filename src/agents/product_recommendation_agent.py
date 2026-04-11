import pandas as pd


class ProductRecommendationAgent:
    def __init__(self, order_items, products):
        self.order_items = order_items
        self.products = products

        # -------------------------- 修复代码开始：提前关联表并计算价格 --------------------------
        # 1. 关联 order_items 和 products，获取商品完整信息+价格
        items_with_products = order_items.merge(
            products[["product_id", "product_category_name_english"]],
            on="product_id",
            how="left"
        )

        # 2. 计算每个商品的平均价格（同一个商品可能在不同订单里价格不同）
        self.product_avg_prices = items_with_products.groupby("product_id").agg(
            avg_price=("price", "mean"),
            sales_count=("order_item_id", "count")
        ).reset_index()

        # 3. 关联回 products 表，得到完整商品信息
        self.products_with_price = self.product_avg_prices.merge(
            products,
            on="product_id",
            how="left"
        )

        # 4. 计算品类热销商品
        self.category_bestsellers = self._calculate_category_bestsellers()
        # -------------------------- 修复代码结束 --------------------------

    def _calculate_category_bestsellers(self):
        """计算每个品类的热销商品（基于销量）"""
        category_bestsellers = {}
        for category in self.products_with_price["product_category_name_english"].unique():
            if pd.isna(category):
                continue
            category_products = self.products_with_price[
                self.products_with_price["product_category_name_english"] == category
                ]
            # 按销量排序，取前5
            bestsellers = category_products.sort_values("sales_count", ascending=False).head(5)
            category_bestsellers[category] = bestsellers["product_id"].tolist()
        return category_bestsellers

    def recommend_products(self, user_profile, n=3):
        """基于用户偏好推荐商品（修复版：使用正确的价格）"""
        print("[商品推荐Agent] 正在生成个性化商品推荐...")

        favorite_categories = user_profile["favorite_categories"]
        # 容错：如果用户没有平均消费，用全局平均
        avg_price = user_profile.get("avg_price", 0)
        if pd.isna(avg_price) or avg_price == 0:
            avg_price = self.products_with_price["avg_price"].mean()

        recommended_products = []

        # 从用户喜欢的品类中推荐
        for category in favorite_categories:
            if category in self.category_bestsellers:
                for product_id in self.category_bestsellers[category]:
                    # 从预计算好的表里取商品信息
                    product = self.products_with_price[
                        self.products_with_price["product_id"] == product_id
                        ].iloc[0]

                    # 价格过滤（推荐与用户平均购买价格相近的商品）
                    product_price = product["avg_price"]
                    if 0.5 * avg_price <= product_price <= 1.5 * avg_price:
                        recommended_products.append({
                            "product_id": product_id,
                            "category": category,
                            "price": product_price,
                            "product_name": f"{category} 商品 {product_id[:8]}"
                        })

                    if len(recommended_products) >= n:
                        break

            if len(recommended_products) >= n:
                break

        # 如果推荐数量不够，从热销商品中补充
        if len(recommended_products) < n:
            all_bestsellers = self.products_with_price.sort_values("sales_count", ascending=False).head(20)
            for _, product in all_bestsellers.iterrows():
                product_id = product["product_id"]
                if product_id not in [p["product_id"] for p in recommended_products]:
                    recommended_products.append({
                        "product_id": product_id,
                        "category": product["product_category_name_english"],
                        "price": product["avg_price"],
                        "product_name": f"{product['product_category_name_english']} 商品 {product_id[:8]}"
                    })

                    if len(recommended_products) >= n:
                        break

        print(f"[商品推荐Agent] 推荐了 {len(recommended_products)} 个商品：")
        for product in recommended_products:
            print(f"  - {product['product_name']}，价格：${product['price']:.2f}")
        print()

        return recommended_products