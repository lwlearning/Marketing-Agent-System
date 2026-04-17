# src/utils/llm_utils.py

def parse_llm_output(result, schema_cls, strict=True):
    """
    工业级 LLM 输出解析函数
    """

    try:
        if isinstance(result, schema_cls):
            return result.model_dump()

        if hasattr(result, "parsed") and result.parsed:
            return result.parsed.model_dump()

        if not strict and hasattr(result, "content"):
            import json
            return json.loads(result.content)

    except Exception as e:
        print(f"[LLM解析失败] {e}")

    return None