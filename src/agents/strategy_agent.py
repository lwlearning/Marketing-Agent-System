from src.utils.rag_helper import load_marketing_rules


class StrategyAgent:
    def __init__(self, knowledge_base_path):
        self.knowledge_base = load_marketing_rules(knowledge_base_path)

    def generate_strategy(self, user_profile):
        """根据用户画像生成策略"""
        print("[策略Agent] 正在生成营销策略...")

        user_segment = user_profile['user_segment']

        # 简单规则匹配
        if user_segment == '高价值用户':
            strategy = {
                'segment': '高价值用户',
                'offer': '专属9折优惠券 + 新品优先体验',
                'channel': '短信 + APP Push',
                'time': '工作日晚上8点',
                'tone': '尊贵、专属'
            }
        elif user_segment == '中价值用户':
            strategy = {
                'segment': '中价值用户',
                'offer': '满199减50优惠券',
                'channel': 'APP Push',
                'time': '周末下午3点',
                'tone': '实惠、推荐'
            }
        else:
            strategy = {
                'segment': '低价值用户',
                'offer': '限时5折秒杀券',
                'channel': '短信',
                'time': '上午10点',
                'tone': '紧迫感、福利'
            }

        print(f"[策略Agent] 生成的策略：\n{strategy}\n")
        return strategy