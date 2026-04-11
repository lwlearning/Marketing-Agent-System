import pandas as pd


class ProductRecommendationAgent:
    def __init__(self, order_items, products):
        self.order_items = order_items
        self.products = products

        # 计算品类热销商品
        self.category_bestsellers = self._calculate_category_bestsellers()

    def _calculate_category_bestsellers(self):
        """计算每个品类的热销商品"""
        # 关联商品信息
        order_items_with_category = self.order_items.merge(
            self.products[["product_id", "product_category_name_english"]],
            on="product_id",
            how="left"
        )

        # 计算每个商品的销量
        product_sales = order_items_with_category.groupby(
            ["product_category_name_english", "product_id"]
        ).size().reset_index(name="sales_count")

        # 每个品类取销量前5的商品
        category_bestsellers = {}
        for category in product_sales["product_category_name_english"].unique():
            if pd.isna(category):
                continue
            category_products = product_sales[product_sales["product_category_name_english"] == category]
            bestsellers = category_products.sort_values("sales_count", ascending=False).head(5)
            category_bestsellers[category] = bestsellers["product_id"].tolist()

        return category_bestsellers

    def recommend_products(self, user_profile, n=3):
        """基于用户偏好推荐商品"""
        print("[商品推荐Agent] 正在生成个性化商品推荐...")

        favorite_categories = user_profile["favorite_categories"]
        avg_price = user_profile["avg_price"]

        recommended_products = []

        # 从用户喜欢的品类中推荐
        for category in favorite_categories:
            if category in self.category_bestsellers:
                for product_id in self.category_bestsellers[category]:
                    product = self.products[self.products["product_id"] == product_id].iloc[0]

                    # 价格过滤（推荐与用户平均购买价格相近的商品）
                    if 0.5 * avg_price <= product["price"] <= 1.5 * avg_price:
                        recommended_products.append({
                            "product_id": product_id,
                            "category": category,
                            "price": product["price"],
                            "product_name": f"{category} 商品 {product_id[:8]}"  # 简化名称
                        })

                    if len(recommended_products) >= n:
                        break

            if len(recommended_products) >= n:
                break

        # 如果推荐数量不够，从热销商品中补充
        if len(recommended_products) < n:
            all_bestsellers = []
            for category in self.category_bestsellers:
                all_bestsellers.extend(self.category_bestsellers[category][:2])

            for product_id in all_bestsellers:
                if product_id not in [p["product_id"] for p in recommended_products]:
                    product = self.products[self.products["product_id"] == product_id].iloc[0]
                    recommended_products.append({
                        "product_id": product_id,
                        "category": product["product_category_name_english"],
                        "price": product["price"],
                        "product_name": f"{product['product_category_name_english']} 商品 {product_id[:8]}"
                    })

                    if len(recommended_products) >= n:
                        break

        print(f"[商品推荐Agent] 推荐了 {len(recommended_products)} 个商品：")
        for product in recommended_products:
            print(f"  - {product['product_name']}，价格：${product['price']:.2f}")
        print()

        return recommended_products