def load_marketing_rules(file_path):
    """简单加载营销知识库"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()