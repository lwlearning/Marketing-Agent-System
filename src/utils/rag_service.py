import os
import faiss
import numpy as np
import logging
from typing import List, Optional
import dashscope
from dashscope import TextEmbedding
from langchain_core.embeddings import Embeddings
import config
from retry import retry  #  pip install retry

# -------------------------- 日志配置（替代print，生产环境标准） --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RAGService")

# -------------------------- Embedding 优化 --------------------------
class QwenEmbeddings(Embeddings):
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-v2",
        timeout: int = 30,
        batch_size: int = 10  # 可配置批处理大小
    ):
        dashscope.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.batch_size = batch_size

    @retry(tries=2, delay=1)  # 网络波动自动重试
    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        resp = TextEmbedding.call(
            model=self.model,
            input=batch,
            timeout=self.timeout
        )
        if resp.status_code != 200:
            raise Exception(f"Embedding API error: {resp.status_code}")
        return [e['embedding'] for e in resp.output['embeddings']]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 严格预处理，保留长度对应关系
        valid_texts = [str(t).strip() for t in texts]
        embeddings = [[] for _ in valid_texts]

        for i in range(0, len(valid_texts), self.batch_size):
            batch_idx = list(range(i, min(i + self.batch_size, len(valid_texts))))
            batch = [valid_texts[j] for j in batch_idx if valid_texts[j]]

            if not batch:
                continue

            try:
                batch_emb = self._embed_batch(batch)
                ptr = 0
                for j in batch_idx:
                    if valid_texts[j]:
                        embeddings[j] = batch_emb[ptr]
                        ptr += 1
            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")

        # 空向量填充零向量，保证长度一致（核心修复）
        dim = 1536  # 通义v2固定维度
        return [vec if vec else [0.0]*dim for vec in embeddings]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

# -------------------------- 文本加载（优化编码、日志） --------------------------
def load_documents(dir_path: str) -> List[dict]:
    if not os.path.exists(dir_path):
        logger.warning(f"知识库目录不存在: {dir_path}")
        return []

    docs = []
    for filename in os.listdir(dir_path):
        if filename.lower().endswith((".txt", ".md")):
            path = os.path.join(dir_path, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        docs.append({
                            "content": content,
                            "metadata": {"source": filename}
                        })
            except Exception as e:
                logger.error(f"读取文件失败 {filename}: {e}")
    return docs

# -------------------------- 智能文本切分（替代硬切割，保留语义） --------------------------
def split_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 100
) -> List[str]:
    """按标点/换行切分，不硬断句，提升RAG效果"""
    separators = ["\n\n", "\n", "。", "！", "？", "；", ".", " "]
    chunks = []
    current_chunk = []
    current_length = 0

    for sep in separators:
        if sep in text:
            sentences = text.split(sep)
            break
    else:
        sentences = [text]

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        sent_len = len(sent)

        if current_length + sent_len > chunk_size and current_chunk:
            chunks.append(sep.join(current_chunk))
            # 重叠部分
            overlap = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) > chunk_overlap:
                    break
                overlap.insert(0, s)
                overlap_len += len(s)
            current_chunk = overlap
            current_length = overlap_len

        current_chunk.append(sent)
        current_length += sent_len

    if current_chunk:
        chunks.append(sep.join(current_chunk))
    return [c for c in chunks if c.strip()]

# -------------------------- RAG Service（终极优化版） --------------------------
class RAGService:
    def __init__(self):
        # 1. 路径配置
        self.knowledge_base_dir = config.KNOWLEDGE_BASE_DIR
        self.faiss_index_path = "./faiss.index"
        self.meta_path = "./faiss_meta.npy"

        # 2. RAG核心超参数（统一管理，禁止魔法数字）
        self.chunk_size = 512
        self.chunk_overlap = 100
        self.embed_batch_size = 10
        self.top_k = 3
        self.nprobe = 10
        self.min_nlist = 1
        self.max_nlist = 1000

        # 3. 初始化嵌入模型
        self.embeddings = QwenEmbeddings(
            api_key=config.QWEN_API_KEY,
            model=config.QWEN_EMBEDDING_MODEL,
            batch_size=self.embed_batch_size
        )

        logger.info("[RAG] 初始化中...")
        self._init_index()

    def _init_index(self):
        """统一初始化入口"""
        if os.path.exists(self.faiss_index_path):
            self._load()
        else:
            self._build()

    def _build(self):
        """构建索引（自动适配大小数据）"""
        # 1. 加载并切分文档
        docs = load_documents(self.knowledge_base_dir)
        if not docs:
            logger.error("无有效文档，索引构建失败")
            return

        all_chunks = []
        for doc in docs:
            chunks = split_text(
                doc["content"],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            for c in chunks:
                all_chunks.append({
                    "text": c,
                    "metadata": doc["metadata"]
                })

        texts = [c["text"] for c in all_chunks]
        logger.info(f"[RAG] 有效分块数量: {len(texts)}")

        # 2. 向量化
        vectors = self.embeddings.embed_documents(texts)
        vectors = np.array(vectors, dtype="float32")

        # 3. 归一化（余弦相似度）
        faiss.normalize_L2(vectors)
        dim = vectors.shape[1]
        num_vectors = len(vectors)

        # ===================== 【自动适配：小数据Flat，大数据IVF】 =====================
        threshold = 100  # 小于100条用Flat，大于用IVF
        if num_vectors < threshold:
            logger.info(f"[RAG] 数据量过小({num_vectors})，使用Flat索引")
            index = faiss.IndexFlatIP(dim)
        else:
            # IVF 索引配置
            nlist = int(np.sqrt(num_vectors))
            nlist = max(self.min_nlist, min(nlist, self.max_nlist))
            logger.info(f"[RAG] 数据量较大({num_vectors})，使用IVF索引，nlist={nlist}, nprobe={self.nprobe}")

            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(
                quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT
            )
            # 训练IVF索引
            index.train(vectors)
            index.nprobe = self.nprobe
        # ============================================================================

        # 添加向量
        index.add(vectors)

        # 保存索引
        faiss.write_index(index, self.faiss_index_path)
        np.save(self.meta_path, all_chunks, allow_pickle=True)

        self.index = index
        self.meta = all_chunks
        logger.info("[RAG] 索引构建完成")

    def _load(self):
        """加载索引（自动判断索引类型）"""
        try:
            self.index = faiss.read_index(self.faiss_index_path)
            self.meta = np.load(self.meta_path, allow_pickle=True).tolist()

            # ✅ 只有 IVF 索引才设置 nprobe，Flat 不设置
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = self.nprobe
                logger.info(f"[RAG] IVF索引加载完成，nprobe={self.nprobe}")
            else:
                logger.info("[RAG] Flat索引加载完成")

        except Exception as e:
            logger.error(f"索引加载失败: {e}")
            raise

    def retrieve(self, query: str) -> List[dict]:
        """检索优化（高效去重、鲁棒性增强）"""
        if not query.strip():
            return []

        # 1. 向量化 + 归一化
        q_vec = np.array([self.embeddings.embed_query(query)], dtype="float32")
        faiss.normalize_L2(q_vec)

        # 2. 检索
        scores, indices = self.index.search(q_vec, self.top_k * 2)  # 多取防止重复

        # 3. 结果处理（用索引去重，效率更高）
        results = []
        seen_idx = set()

        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.meta):
                continue
            if idx in seen_idx:
                continue

            chunk = self.meta[idx]
            results.append({
                "text": chunk["text"],
                "score": float(score),
                "metadata": chunk["metadata"]
            })
            seen_idx.add(idx)

            if len(results) >= self.top_k:
                break

        return results