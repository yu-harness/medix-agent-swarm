"""
医学知识库（Milvus）

功能：
1. 文档向量化和存储
2. 语义检索
3. 知识库管理

参考实现：/Users/saintgeo/Desktop/self-learn/shanglv
"""
import json
import os
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False

try:
    from sentence_transformers import CrossEncoder
    _HAS_RERANKER = True
except ImportError:
    _HAS_RERANKER = False


class MedicalKnowledgeBase:
    """医学知识库"""

    _instance = None
    # Skill 现由线程池并发执行，单例的创建与初始化必须加锁，
    # 否则并发首次调用会重复加载向量模型 / 重复创建 Milvus 连接
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """实现单例模式（线程安全，双重检查锁定）"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        db_path: str = "./knowledge/data/milvus_lite.db",
        collection_name: str = "medical_knowledge",
        embedding_model: str = "BAAI/bge-small-zh-v1.5"
    ):
        """
        初始化医学知识库

        Args:
            db_path: Milvus Lite 数据库文件路径
            collection_name: Collection 名称
            embedding_model: Embedding 模型名称或本地路径
        """
        # 防止重复初始化：锁覆盖整个初始化过程，
        # 保证并发首次调用只有一个线程真正执行 _initialize()
        with self._instance_lock:
            if getattr(self, '_initialized', False):
                return
            self._initialize(db_path, collection_name, embedding_model)
            self._initialized = True

    def _initialize(
        self,
        db_path: str,
        collection_name: str,
        embedding_model: str,
    ):
        """真正的初始化逻辑（由 __init__ 在锁内调用，请勿直接调用）"""
        target = Path(__file__).resolve().parent.parent / 'knowledge' / 'data' / 'milvus_lite.db'
        target.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(target)
        self.collection_name = collection_name

        # 初始化 Embedding 模型（支持 HF_HOME / 默认缓存本地路径）
        roots = []
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            roots.append(Path(hf_home) / "hub" / "models--BAAI--bge-small-zh-v1.5" / "snapshots")
        roots.append(Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-small-zh-v1.5" / "snapshots")
        model_path = None
        for local_model_path in roots:
            if not local_model_path.exists():
                continue
            snapshots = sorted(local_model_path.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if snapshots:
                model_path = str(snapshots[0])
                break
        if model_path:
            logger.info(f"Loading embedding model from local cache: {model_path}")
            self.embedding_model = SentenceTransformer(model_path, device='cpu')
        else:
            logger.info(f"Loading embedding model: {embedding_model}")
            self.embedding_model = SentenceTransformer(embedding_model, device='cpu')

        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded (dimension={self.embedding_dim})")

        # 初始化 Milvus Lite（调大 keepalive，避免默认 10s ping 触发 too_many_pings）
        logger.info(f"Connecting to Milvus Lite: {self.db_path}")
        # pymilvus 默认 keepalive=10s，Milvus Lite/gRPC 服务端会 GOAWAY too_many_pings
        os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
        self.milvus_client = MilvusClient(
            uri=self.db_path,
            db_name="default",
            grpc_options={
                "grpc.keepalive_time_ms": 300000,
                "grpc.keepalive_timeout_ms": 20000,
                "grpc.keepalive_permit_without_calls": False,
            },
        )

        # 创建 collection（如果不存在）
        if not self.milvus_client.has_collection(collection_name):
            logger.info(f"Creating collection: {collection_name}")
            self.milvus_client.create_collection(
                collection_name=collection_name,
                dimension=self.embedding_dim,
                metric_type="COSINE",  # 余弦相似度
                auto_id=True  # 自动生成整数ID
            )
        else:
            logger.info(f"Collection already exists: {collection_name}")

        # 加载 collection 到内存（MilvusLite 重启后需要重新 load）
        try:
            self.milvus_client.load_collection(collection_name)
            logger.info(f"Collection loaded into memory: {collection_name}")
        except Exception as e:
            logger.warning(f"Failed to load collection: {e}")

        self._reranker = None

        # BM25 索引缓存：语料构建代价高（全表拉取 + 全量分词 + 建索引），
        # 不能每次检索都重做；知识库写入时由 invalidate_bm25_cache() 失效
        self._bm25_cache: Optional[Dict[str, Any]] = None
        self._bm25_lock = threading.RLock()

        # reranker 懒加载锁：Skill 现由线程池并发执行，若不加锁，
        # 并发首次检索会重复加载 CrossEncoder（模型约 1GB，代价极高）
        self._reranker_lock = threading.Lock()

    def _get_reranker(self):
        """懒加载 reranker（cross-encoder），按需加载，避免拖慢启动（线程安全）"""
        if not _HAS_RERANKER:
            return None

        # 快路径：已加载（或已标记失败）则直接返回
        if self._reranker is not None:
            return self._reranker if self._reranker else None

        with self._reranker_lock:
            # 双重检查：可能已被其它线程加载完成
            if self._reranker is not None:
                return self._reranker if self._reranker else None
            try:
                import os

                # 优先本地缓存，其次按名称加载
                reranker_name = os.environ.get(
                    "MEDIX_RERANKER_MODEL", "BAAI/bge-reranker-base"
                )
                logger.info(f"Loading reranker: {reranker_name}")
                self._reranker = CrossEncoder(reranker_name)
            except Exception as e:
                logger.error(f"Failed to load reranker: {e}")
                self._reranker = False  # 标记失败，避免反复尝试

        return self._reranker if self._reranker else None

    def _chunk_text(self, text: str, chunk_size: int = 1024, overlap: int = 100) -> List[str]:
        """
        分块文本

        Args:
            text: 原始文本
            chunk_size: 块大小（字符数）
            overlap: 重叠字符数

        Returns:
            文本块列表
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap  # 重叠

        return chunks

    def add_documents(self, documents: List[Dict[str, Any]], chunk_size: int = 1024) -> int:
        """
        添加文档到知识库（支持分块）

        Args:
            documents: 文档列表，每个文档包含 id, content, metadata
            chunk_size: 分块大小（字符数），默认 1024

        Returns:
            成功添加的文档块数量
        """
        if not documents:
            logger.warning("No documents to add")
            return 0

        logger.info(f"Adding {len(documents)} documents to knowledge base (chunk_size={chunk_size})...")

        # 分块并向量化
        all_chunks = []
        for doc in documents:
            chunks = self._chunk_text(doc["content"], chunk_size=chunk_size)
            for i, chunk in enumerate(chunks):
                metadata = doc.get("metadata", {}).copy()
                metadata["doc_id"] = doc["id"]
                metadata["chunk_id"] = i
                metadata["total_chunks"] = len(chunks)

                all_chunks.append({
                    "content": chunk,
                    "metadata": metadata
                })

        logger.info(f"Split into {len(all_chunks)} chunks")

        # 向量化
        contents = [chunk["content"] for chunk in all_chunks]
        vectors = self.embedding_model.encode(contents, show_progress_bar=True)

        # 准备数据
        data = []
        for i, chunk in enumerate(all_chunks):
            data.append({
                "vector": vectors[i].tolist(),
                "content": chunk["content"],
                "metadata": json.dumps(chunk["metadata"], ensure_ascii=False)
            })

        # 插入
        self.milvus_client.insert(self.collection_name, data)
        logger.info(f"Successfully added {len(data)} chunks")

        # 语料已变化，BM25 索引必须失效，否则后续检索会命中过期语料
        self.invalidate_bm25_cache()

        return len(data)

    def _fetch_all_chunks(self, limit: int = 16384) -> List[Dict[str, Any]]:
        """
        拉取知识库全部 chunk（按内容去重），供 BM25 混合检索使用。

        Milvus Lite 没有流式遍历接口，这里用 query + 内存去重。
        chunk 总量通常很小（几十~几百条），内存开销可接受。
        """
        try:
            data = self.milvus_client.query(
                self.collection_name,
                filter="",
                output_fields=["content", "metadata"],
                limit=limit,
            )
        except Exception as e:
            logger.error(f"_fetch_all_chunks failed: {e}")
            return []

        # 按 content 去重（之前重复导入导致同一 chunk 存在多份）
        seen: Dict[str, Dict[str, Any]] = {}
        for d in data:
            content = d.get("content") or ""
            if not content or content in seen:
                continue
            meta = d.get("metadata")
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            seen[content] = {
                "id": d.get("id"),
                "content": content,
                "metadata": meta,
            }
        return list(seen.values())

    def invalidate_bm25_cache(self) -> None:
        """让 BM25 索引缓存失效。

        知识库发生任何写入后都必须调用，否则检索会命中过期语料。
        """
        with self._bm25_lock:
            if self._bm25_cache is not None:
                logger.debug("BM25 index cache invalidated")
            self._bm25_cache = None

    def _get_bm25_index(self) -> Optional[Dict[str, Any]]:
        """获取 BM25 索引，首次调用或缓存失效后重建。

        缓存内容：去重后的 chunk 列表 + 分词后的语料 + BM25Okapi 实例。
        只在首次检索或知识库写入后重建，避免每次查询都全表扫描并重建索引（O(N)）。
        """
        if not _HAS_BM25:
            return None

        # 快路径：缓存命中直接返回（单个属性赋值为原子操作，无锁读安全）
        cache = self._bm25_cache
        if cache is not None:
            return cache

        with self._bm25_lock:
            # 双重检查：可能已被其它线程抢先构建
            if self._bm25_cache is not None:
                return self._bm25_cache

            chunks = self._fetch_all_chunks()
            if not chunks:
                # 知识库为空：不缓存（空集合查询很快），下次仍会重试
                return None

            corpus = [self._tokenize(c["content"]) for c in chunks]
            self._bm25_cache = {
                "chunks": chunks,
                "index": BM25Okapi(corpus),
            }
            logger.info(f"BM25 index built/rebuilt (chunks={len(chunks)})")
            return self._bm25_cache

    def _bm25_retrieve(
        self, query: str, top_k: int = 5, filter_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """BM25 关键词检索（在去重后的全量 chunk 上，索引带缓存）。

        filter_type: 按 metadata.type 过滤候选。必须与向量检索的过滤保持一致，
        否则 BM25 会把其它类型的文档通过 RRF 融合"漏"进最终结果，导致 filter_type
        约束被绕过（例如查 ICD-10 编码却返回生活方式文档，给出错误编码）。
        """
        if not _HAS_BM25:
            return []
        try:
            cache = self._get_bm25_index()
        except Exception as e:
            logger.error(f"_bm25_retrieve build/fetch failed: {e}")
            return []
        if not cache:
            return []

        chunks = cache["chunks"]
        # 索引覆盖全量 chunk，下标与 chunks 一一对应，因此先算全量分数再过滤
        scores = cache["index"].get_scores(self._tokenize(query))

        ranked = sorted(
            range(len(chunks)), key=lambda i: scores[i], reverse=True
        )
        if filter_type:
            ranked = [
                i for i in ranked
                if (chunks[i].get("metadata") or {}).get("type") == filter_type
            ]

        results = []
        for i in ranked[:top_k]:
            if scores[i] <= 0.0:
                continue
            results.append(
                {
                    "id": chunks[i]["id"],
                    "content": chunks[i]["content"],
                    "metadata": chunks[i]["metadata"],
                    "score": float(scores[i]),
                }
            )
        return results

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """BM25 中文分词（简单实现：按非汉字/非字母数字切分 + 汉字单字）"""
        if not text:
            return []
        # 拉丁字母/数字 token
        import re as _re

        tokens = _re.findall(r"[a-zA-Z0-9]+", text.lower())
        # 汉字：按连续汉字串再切成单字（对医学术语召回更稳）
        for han in _re.findall(r"[一-鿿]+", text):
            tokens.extend(list(han))
        return tokens

    @staticmethod
    def _rrf_fusion(
        ranked_lists: List[List[Dict[str, Any]]], k: int = 60
    ) -> Dict[str, float]:
        """Reciprocal Rank Fusion：合并多路检索结果"""
        scores: Dict[str, float] = {}
        doc_refs: Dict[str, Dict[str, Any]] = {}
        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked):
                key = doc.get("id")
                if key is None:
                    key = doc.get("content", "")[:120]
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
                doc_refs.setdefault(key, doc)
        return doc_refs, scores

    def search(
        self,
        query: str,
        top_k: int = 8,
        filter_type: Optional[str] = None,
        use_hybrid: bool = True,
        min_vector_score: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """
        检索相关文档

        Args:
            query: 查询文本
            top_k: 返回top K个结果
            filter_type: 可选的类型过滤（如 "lifestyle", "disease_classification"）
            use_hybrid: 是否使用混合检索（向量 + BM25 + RRF 融合）
            min_vector_score: 向量相似度下限（过低则丢弃噪声命中）

        Returns:
            文档列表，每个文档包含 id, content, metadata, score
        """
        logger.debug(f"Searching for: {query} (top_k={top_k}, filter_type={filter_type}, use_hybrid={use_hybrid})")

        # 向量化查询
        query_vector = self.embedding_model.encode([query])[0]

        # 构建过滤条件
        filter_expr = None
        if filter_type:
            filter_expr = f'metadata like "%\\"type\\": \\"{filter_type}\\"%"'

        # 检索
        try:
            results = self.milvus_client.search(
                collection_name=self.collection_name,
                data=[query_vector.tolist()],
                limit=top_k * 2,  # 多取一些，供融合后取 top_k
                filter=filter_expr,
                output_fields=["content", "metadata"]
            )
        except Exception as e:
            # 如果 collection 被 released，尝试重新 load 并重试一次
            if "released" in str(e).lower():
                logger.warning(f"Collection released, attempting to reload...")
                try:
                    self.milvus_client.load_collection(self.collection_name)
                    results = self.milvus_client.search(
                        collection_name=self.collection_name,
                        data=[query_vector.tolist()],
                        limit=top_k * 2,
                        filter=filter_expr,
                        output_fields=["content", "metadata"]
                    )
                except Exception as e2:
                    logger.error(f"Search retry also failed: {e2}")
                    return []
            else:
                logger.error(f"Search failed: {e}")
                return []

        # 格式化向量检索结果（按内容去重）
        vector_docs: Dict[str, Dict[str, Any]] = {}
        for hits in results:
            for hit in hits:
                try:
                    content = hit["entity"]["content"]
                    if not content or content in vector_docs:
                        continue
                    vector_docs[content] = {
                        "id": hit["id"],
                        "content": content,
                        "metadata": json.loads(hit["entity"]["metadata"]),
                        "score": 1 - hit["distance"],  # 转换为相似度分数
                    }
                except Exception as e:
                    logger.warning(f"Failed to parse result: {e}")
                    continue
        vector_ranked = [
            d for d in vector_docs.values() if d.get("score", 0) >= min_vector_score
        ]
        vector_ranked = sorted(
            vector_ranked, key=lambda d: d.get("score", 0), reverse=True
        )[: top_k * 2]

        # 若关闭混合检索或没有 BM25，直接返回向量结果
        if not use_hybrid or not _HAS_BM25:
            return vector_ranked[:top_k]

        # 混合检索：向量 + BM25，RRF 融合
        try:
            # filter_type 必须一并传给 BM25，否则融合结果会混入其它类型的文档
            bm25_ranked = self._bm25_retrieve(
                query, top_k=top_k * 2, filter_type=filter_type
            )
        except Exception as e:
            logger.error(f"BM25 retrieve failed, fallback to vector only: {e}")
            return vector_ranked[:top_k]

        if not bm25_ranked:
            return vector_ranked[:top_k]

        doc_refs, fused = self._rrf_fusion([vector_ranked, bm25_ranked])
        # 按融合分数降序取 top_k
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        documents = []
        for key, score in ordered:
            doc = doc_refs[key]
            documents.append(
                {
                    "id": doc["id"],
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": round(score, 4),  # RRF 融合分
                }
            )

        # 可选：cross-encoder 重排（rerank），把真正相关的排在前面
        reranker = self._get_reranker()
        if reranker is not None and documents:
            try:
                pairs = [[query, d["content"]] for d in documents]
                rerank_scores = reranker.predict(pairs)
                for d, s in zip(documents, rerank_scores):
                    d["score"] = round(float(s), 4)  # 保留 rerank 分
                documents.sort(key=lambda d: d["score"], reverse=True)
            except Exception as e:
                logger.error(f"Rerank failed, keep RRF order: {e}")

        logger.debug(f"Hybrid search returned {len(documents)} documents")
        return documents

    def delete_collection(self):
        """删除 collection（用于测试）"""
        if self.milvus_client.has_collection(self.collection_name):
            self.milvus_client.drop_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        self.invalidate_bm25_cache()

    def count_documents(self) -> int:
        """统计文档数量"""
        try:
            stats = self.milvus_client.describe_collection(self.collection_name)
            # Note: Milvus Lite may not return accurate count, this is a best-effort
            return stats.get("num_entities", 0)
        except Exception as e:
            logger.warning(f"Failed to count documents: {e}")
            return 0
