"""
Search Knowledge Skill
搜索医学知识库 Skill（自包含，无需依赖tools）
"""
from typing import Dict, Any
from loguru import logger
import threading

# 全局知识库实例（避免重复加载模型）
_kb_instance = None
# Skill 现由 SkillRegistry 丢进线程池并发执行，懒加载必须加锁；
# 否则多线程会重复创建实例（重复加载向量模型 + 重复打开 Milvus Lite 文件句柄）
_kb_lock = threading.RLock()

# 类型代码 → 中文标签。来源头是给人看、也给模型引用的话，显示中文更清楚；
# 未收录的代码原样输出，不做隐式转换
_TYPE_LABELS = {
    "clinical_guideline": "临床指南",
    "lifestyle": "生活方式",
    "disease_classification": "疾病分类",
    "general": "通用咨询",
}


def get_knowledge_base():
    """获取知识库单例（线程安全，双重检查锁定）"""
    global _kb_instance
    if _kb_instance is None:
        with _kb_lock:
            if _kb_instance is None:
                from knowledge.milvus_kb import MedicalKnowledgeBase
                _kb_instance = MedicalKnowledgeBase()
    return _kb_instance


# 注意：函数体全为同步阻塞调用（Milvus 检索 + 向量模型推理），没有任何 await。
# 声明为同步函数后，SkillRegistry 会自动丢进线程池执行，不阻塞 event loop；
# 切勿改回 async def，否则阻塞会串行化整个 Swarm。
def search_knowledge(query: str, max_results: int = 8) -> Dict[str, Any]:
    """
    搜索医学知识库

    Args:
        query: 查询内容
        max_results: 最多返回结果数（默认8，提高咨询/症状召回覆盖）

    Returns:
        {
            "answer": "格式化的知识库检索结果",
            "total_found": 检索到的结果数,
            "query": "原始查询"
        }
    """
    logger.info(f"Searching knowledge base: query={query}, max_results={max_results}")

    # 获取知识库单例（避免重复加载模型）
    kb = get_knowledge_base()

    # 使用 Milvus 进行语义检索
    results = kb.search(
        query=query,
        top_k=max_results,
        filter_type=None
    )

    # 格式化结果
    #
    # 注意：format_results 读的是每个结果里的 `metadata` 字典（与 kb.search 的输出同构）。
    # 这里曾经把 source/type 拍平到顶层，于是 format_results 拿到空 metadata，
    # 来源头在**真实 Agent 链路**里退化成「来源：医学知识库｜类型：-」——
    # 模型看不到出处，接地（P1-3）等于白做。必须原样把 metadata 传下去。
    formatted_results = []
    for doc in results:
        md = doc.get("metadata") or {}
        formatted_results.append({
            "content": doc["content"],
            "score": doc["score"],
            "metadata": {
                "source": md.get("source"),
                "filename": md.get("filename"),
                "type": md.get("type"),
                # 页码：来源头会拼成 p.8-9，便于人工翻原文核对
                "pages": md.get("pages"),
            },
        })

    # Skill 的格式化输出
    if formatted_results:
        return {
            "answer": format_results(formatted_results),
            "total_found": len(formatted_results),
            "query": query
        }
    else:
        return {
            "answer": f"未找到关于'{query}'的相关医学知识，请尝试更具体的查询。",
            "total_found": 0,
            "query": query
        }


def format_results(results: list) -> str:
    """
    格式化知识库检索结果

    每条结果都带上来源头，让模型能引用并溯源：
        【资料 1｜来源：中国高血压防治指南（2024年修订版） p.1｜类型：临床指南】

    三条规则：
    1. 来源优先取 source（资料全称），缺失才回退到 filename
    2. 有 pages 时拼在来源后（p.8-9），便于人工翻原文核对
    3. type 转成中文标签，未收录的代码原样显示

    Args:
        results: 检索结果列表

    Returns:
        格式化的字符串
    """
    if not results:
        return "未找到相关信息。"

    output = []
    for i, doc in enumerate(results, 1):
        metadata = doc.get("metadata") or {}
        source = metadata.get("source") or metadata.get("filename") or "医学知识库"
        pages = str(metadata.get("pages") or "").strip()
        if pages and pages != "-":
            source = f"{source} p.{pages}"
        doc_type = _TYPE_LABELS.get(str(metadata.get("type") or ""), str(metadata.get("type") or "-"))
        output.append(f"【资料 {i}｜来源：{source}｜类型：{doc_type}】")
        output.append(doc.get("content", "无内容"))

        # 显示相关度分数（如果有）
        score = doc.get("score", 0)
        if score > 0:
            output.append(f"相关度: {score:.2%}")

        output.append("")  # 空行分隔

    return "\n".join(output)


# 同步版本（函数本身已是同步，直接透传）
def search_knowledge_sync(query: str, max_results: int = 5) -> Dict[str, Any]:
    """同步版本的搜索知识库"""
    return search_knowledge(query, max_results)


if __name__ == "__main__":
    # 测试
    test_query = "高血压的治疗方法"
    result = search_knowledge(test_query)

    print("=" * 70)
    print(f"查询: {test_query}")
    print("=" * 70)
    print(result["answer"])
    print("=" * 70)
    print(f"找到结果数: {result['total_found']}")
