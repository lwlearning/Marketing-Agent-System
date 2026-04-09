import random


class EvaluationAgent:
    def __init__(self):
        pass

    def evaluate(self, strategy, content):
        """模拟效果评估"""
        print("[效果评估Agent] 正在模拟A/B测试评估...")

        # 模拟不同分层的转化率
        segment = strategy['segment']
        if segment == '高价值用户':
            base_ctr = 0.15
            improvement = random.uniform(0.05, 0.15)
        elif segment == '中价值用户':
            base_ctr = 0.08
            improvement = random.uniform(0.03, 0.10)
        else:
            base_ctr = 0.03
            improvement = random.uniform(0.02, 0.08)

        predicted_ctr = base_ctr + improvement

        report = {
            'segment': segment,
            'base_ctr': f"{base_ctr:.2%}",
            'predicted_ctr': f"{predicted_ctr:.2%}",
            'improvement': f"{improvement:.2%}",
            'recommendation': "建议立即上线测试" if improvement > 0.05 else "建议优化文案后再测试"
        }

        print(f"[效果评估Agent] 评估报告：\n{report}\n")
        return report