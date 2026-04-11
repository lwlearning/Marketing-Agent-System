from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import config  # 导入配置


class RAGService:
    def __init__(self):
        # ✅ 强制使用config的绝对路径，彻底抛弃相对路径！
        knowledge_base_dir = config.KNOWLEDGE_BASE_DIR
        print("[RAG服务] 正在加载知识库...")

        # 加载所有markdown文档
        documents = []
        import os
        # 用配置的绝对路径遍历文件
        for filename in os.listdir(knowledge_base_dir):
            if filename.endswith(".md"):
                file_path = os.path.join(knowledge_base_dir, filename)
                loader = TextLoader(file_path, encoding="utf-8")
                documents.extend(loader.load())

        # 文档分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len
        )
        splits = text_splitter.split_documents(documents)

        # 通义千问嵌入模型（兼容接口）
        embeddings = OpenAIEmbeddings(
            model=config.QWEN_EMBEDDING_MODEL,
            api_key=config.QWEN_API_KEY,
            base_url=config.QWEN_BASE_URL
        )

        # 向量库路径也用config
        self.vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=embeddings
        )
        self.retriever = self.vectorstore.as_retriever(k=3)

        print(f"[RAG服务] 知识库加载完成，共 {len(splits)} 个文档块\n")

    def retrieve(self, query):
        """检索相关知识"""
        docs = self.retriever.invoke(query)
        return [doc.page_content for doc in docs]