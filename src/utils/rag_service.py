import dashscope
from dashscope import TextEmbedding
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import config
from typing import List


# -------------------------- 自定义通义千问 Embedding 类（完全官方） --------------------------
class QwenEmbeddings(Embeddings):
    def __init__(self, api_key, model="text-embedding-v2"):
        dashscope.api_key = api_key
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档（官方API调用）"""
        # 通义千问官方要求：必须是list[str]，且不能有空字符串
        valid_texts = [str(t).strip() for t in texts if str(t).strip() != ""]

        embeddings = []
        # 分批处理（避免单次请求过大）
        batch_size = 10
        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i + batch_size]
            resp = TextEmbedding.call(
                model=self.model,
                input=batch
            )
            if resp.status_code == 200:
                batch_embeddings = [e['embedding'] for e in resp.output['embeddings']]
                embeddings.extend(batch_embeddings)
            else:
                raise Exception(f"Embedding调用失败：{resp}")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        return self.embed_documents([text])[0]


# -------------------------- RAG 服务（使用官方 Embedding） --------------------------
class RAGService:
    def __init__(self):
        knowledge_base_dir = config.KNOWLEDGE_BASE_DIR
        print("[RAG服务] 正在加载知识库...")

        documents = []
        import os
        for filename in os.listdir(knowledge_base_dir):
            if filename.endswith(".md"):
                file_path = os.path.join(knowledge_base_dir, filename)
                loader = TextLoader(file_path, encoding="utf-8")
                documents.extend(loader.load())

        # 1. 过滤空文档
        valid_documents = []
        for doc in documents:
            if doc.page_content and len(doc.page_content.strip()) > 0:
                valid_documents.append(doc)

        # 2. 切分文档
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            length_function=len
        )
        splits = text_splitter.split_documents(valid_documents)

        # 3. 提取纯文本和元数据
        texts = []
        metadatas = []
        for s in splits:
            text = str(s.page_content).strip()
            if text and len(text) > 0:
                texts.append(text)
                metadatas.append(s.metadata)

        print(f"[RAG服务] 有效文本块数量：{len(texts)}")
        if len(texts) == 0:
            raise ValueError("知识库没有有效文本！请检查 knowledge_base/ 目录")

        # 4. 使用通义千问官方 Embedding
        print("[RAG服务] 正在调用通义千问官方 Embedding API...")
        embeddings = QwenEmbeddings(
            api_key=config.QWEN_API_KEY,
            model=config.QWEN_EMBEDDING_MODEL
        )

        # 5. 创建 FAISS 向量库
        self.vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas
        )
        self.retriever = self.vectorstore.as_retriever(k=3)

        print(f"[RAG服务] 知识库加载完成\n")

    def retrieve(self, query):
        docs = self.retriever.invoke(query)
        return [doc.page_content for doc in docs]