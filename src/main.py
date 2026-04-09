import config
from src.agents.user_analysis_agent import UserAnalysisAgent
from src.agents.strategy_agent import StrategyAgent
from src.agents.content_generation_agent import ContentGenerationAgent
from src.agents.evaluation_agent import EvaluationAgent


def main():
    print("=" * 50)
    print("营销自动化多智能体系统启动")
    print("=" * 50)

    # 1. 初始化各个Agent
    user_agent = UserAnalysisAgent(config.DATA_PATH)
    strategy_agent = StrategyAgent(config.KNOWLEDGE_BASE_PATH)
    content_agent = ContentGenerationAgent()
    eval_agent = EvaluationAgent()

    # 2. 随机选一个用户ID作为示例
    sample_user_id = 10001  # 示例用户ID

    # 3. 执行多智能体流程
    user_profile = user_agent.analyze(sample_user_id)
    strategy = strategy_agent.generate_strategy(user_profile)
    content = content_agent.generate_content(user_profile, strategy)
    report = eval_agent.evaluate(strategy, content)

    # 4. 输出最终结果
    print("=" * 50)
    print("最终营销方案汇总")
    print("=" * 50)
    print(f"用户分层：{strategy['segment']}")
    print(f"营销策略：{strategy['offer']}")
    print(f"营销文案：{content}")
    print(f"预期效果：点击率提升 {report['improvement']}")
    print("=" * 50)


if __name__ == "__main__":
    main()