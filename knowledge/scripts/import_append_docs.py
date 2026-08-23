"""仅追加导入指定新文档，不删库、不重导旧文档。"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from knowledge.milvus_kb import MedicalKnowledgeBase

NEW_FILES = [
    "30_consult_common_conditions.txt",
    "31_symptom_triage_common.txt",
]


def main():
    doc_dir = Path(__file__).parent.parent / "data" / "documents"
    docs = []
    for name in NEW_FILES:
        path = doc_dir / name
        if not path.exists():
            logger.error(f"missing: {path}")
            continue
        content = path.read_text(encoding="utf-8")
        stem = path.stem
        docs.append(
            {
                "id": f"general_{stem}",
                "content": content,
                "metadata": {
                    "type": "general",
                    "disease": content.split("\n", 1)[0].strip()[:80],
                    "source": "咨询与症状高频补充",
                    "filename": name,
                },
            }
        )
    if not docs:
        logger.error("no docs to import")
        return
    kb = MedicalKnowledgeBase()
    n = kb.add_documents(docs)
    logger.info(f"appended chunks={n} files={len(docs)}")
    for q in ("痔疮坐浴 马应龙", "小儿支气管炎 阿奇霉素", "糖尿病 蜜饯 罐头"):
        hits = kb.search(q, top_k=3)
        logger.info(f"probe '{q}' -> {len(hits)} hits, top_score={hits[0]['score'] if hits else None}")


if __name__ == "__main__":
    main()
