import pandas as pd
from datetime import datetime


def load_and_process_data(file_path):
    """加载并预处理用户行为数据"""
    # 读取数据
    df = pd.read_csv(file_path, header=None,
                     names=['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp'])

    # 时间转换
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['date'] = df['datetime'].dt.date

    # 计算用户特征 (RFM简化版)
    user_features = df.groupby('user_id').agg(
        last_purchase=(
        'datetime', lambda x: x[df['behavior_type'] == 'buy'].max() if any(df['behavior_type'] == 'buy') else pd.NaT),
        total_purchases=('behavior_type', lambda x: (x == 'buy').sum()),
        total_views=('behavior_type', lambda x: (x == 'pv').sum()),
        total_carts=('behavior_type', lambda x: (x == 'cart').sum())
    ).reset_index()

    # 计算最近购买天数
    user_features['days_since_last_purchase'] = (datetime.now() - user_features['last_purchase']).dt.days
    user_features['days_since_last_purchase'] = user_features['days_since_last_purchase'].fillna(999)

    return user_features


def simple_user_segmentation(user_features):
    """简单用户分层"""

    def segment(row):
        if row['days_since_last_purchase'] <= 7 and row['total_purchases'] >= 3:
            return '高价值用户'
        elif row['days_since_last_purchase'] <= 30 and row['total_purchases'] >= 1:
            return '中价值用户'
        else:
            return '低价值用户'

    user_features['user_segment'] = user_features.apply(segment, axis=1)
    return user_features