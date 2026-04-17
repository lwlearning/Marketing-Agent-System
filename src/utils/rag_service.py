import os
import faiss
import numpy as np
from typing import List
import dashscope
from dashscope import TextEmbedding
from langchain_core.embeddings import Embeddings
import config


# -------------------------- Embedding --------------------------
class QwenEmbeddings(Embeddings):
    def __init__(self, api_key, model="text-embedding-v2"):
        dashscope.api_key = api_key
        self.model = model
        self.timeout = 30

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        valid_texts = [str(t).strip() for t in texts if str(t).strip()]
        if not valid_texts:
            return []

        embeddings = []
        batch_size = 10

        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i + batch_size]
            try:
                resp = TextEmbedding.call(
                    model=self.model,
                    input=batch,
                    timeout=self.timeout
                )
                if resp.status_code == 200:
                    embeddings.extend([e['embedding'] for e in resp.output['embeddings']])
                else:
                    embeddings.extend([[] for _ in batch])
            except Exception:
                embeddings.extend([[] for _ in batch])

        return [vec for vec in embeddings if vec]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


# -------------------------- 文本加载（替代 TextLoader） --------------------------
def load_documents(dir_path):
    docs = []
    for filename in os.listdir(dir_path):
        if filename.endswith((".txt", ".md")):
            path = os.path.join(dir_path, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    docs.append({
                        "content": f.read(),
                        "metadata": {"source": filename}
                    })
            except:
                continue
    return docs


# -------------------------- 文本切分（替代 RecursiveCharacterTextSplitter） --------------------------
def split_text(text, chunk_size=512, overlap=100):
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# -------------------------- RAG Service --------------------------
class RAGService:
    def __init__(self):
        self.knowledge_base_dir = config.KNOWLEDGE_BASE_DIR
        self.faiss_index_path = "./faiss.index"
        self.meta_path = "./faiss_meta.npy"

        self.embeddings = QwenEmbeddings(
            api_key=config.QWEN_API_KEY,
            model=config.QWEN_EMBEDDING_MODEL
        )

        print("[RAG] loading...")

        if os.path.exists(self.faiss_index_path):
            self._load()
        else:
            self._build()

        self.top_k = 3

    # -------------------------- build --------------------------
    def _build(self):
        docs = load_documents(self.knowledge_base_dir)

        all_chunks = []
        for doc in docs:
            chunks = split_text(doc["content"])
            for c in chunks:
                if c.strip():
                    all_chunks.append({
                        "text": c,
                        "metadata": doc["metadata"]
                    })

        texts = [c["text"] for c in all_chunks]

        print(f"[RAG] chunks: {len(texts)}")

        vectors = self.embeddings.embed_documents(texts)
        vectors = np.array(vectors).astype("float32")

        # ✅ 关键：归一化 → 用 cosine
        faiss.normalize_L2(vectors)

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine

        index.add(vectors)

        faiss.write_index(index, self.faiss_index_path)
        np.save(self.meta_path, all_chunks)

        self.index = index
        self.meta = all_chunks

        print("[RAG] build done")

    # -------------------------- load --------------------------
    def _load(self):
        self.index = faiss.read_index(self.faiss_index_path)
        self.meta = np.load(self.meta_path, allow_pickle=True).tolist()
        print("[RAG] load done")

    # -------------------------- retrieve --------------------------
    def retrieve(self, query):
        q_vec = np.array([self.embeddings.embed_query(query)]).astype("float32")
        faiss.normalize_L2(q_vec)

        scores, indices = self.index.search(q_vec, 10)

        results = []
        seen = set()

        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue

            text = self.meta[idx]["text"]

            if text not in seen:
                results.append({
                    "text": text,
                    "score": float(score)
                })
                seen.add(text)

            if len(results) >= self.top_k:
                break

        return results