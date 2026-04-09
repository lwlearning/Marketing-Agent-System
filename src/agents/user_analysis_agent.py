from src.utils.data_processor import load_and_process_data, simple_user_segmentation


class UserAnalysisAgent:
    def __init__(self, data_path):
        self.data_path = data_path

    def analyze(self, sample_user_id=None):
        """执行用户分析"""
        print("[用户分析Agent] 正在分析用户数据...")

        # 加载数据
        user_features = load_and_process_data(self.data_path)

        # 用户分层
        user_features = simple_user_segmentation(user_features)

        # 展示整体统计
        segment_counts = user_features['user_segment'].value_counts()
        print(f"[用户分析Agent] 用户分层结果：\n{segment_counts}\n")

        # 如果指定了用户，返回该用户的画像
        if sample_user_id:
            user_profile = user_features[user_features['user_id'] == sample_user_id].iloc[0]
            print(f"[用户分析Agent] 目标用户画像：\n{user_profile}\n")
            return user_profile.to_dict()

        return user_features