from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from modules.config import OPENROUTER_API_KEY, EMBEDDING_MODEL_NAME
from typing import List, Optional
import os

# Đường dẫn đến thư mục lưu trữ vector store
VECTOR_STORE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "vector_store")

# Khởi tạo OpenAIEmbeddings với OpenRouter
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    openai_api_key=OPENROUTER_API_KEY,  # type: ignore
    openai_api_base="https://openrouter.ai/api/v1"  # type: ignore
)

def get_embeddings():
    """
    Trả về instance của OpenAIEmbeddings đã được cấu hình
    """
    return embeddings

def embed_text(text: str) -> List[float]:
    """
    Tạo embedding cho một đoạn text
    
    Args:
        text: Đoạn text cần tạo embedding
        
    Returns:
        List các giá trị embedding
    """
    return embeddings.embed_query(text)

def embed_documents(texts: List[str]) -> List[List[float]]:
    """
    Tạo embeddings cho nhiều documents
    
    Args:
        texts: List các đoạn text cần tạo embedding
        
    Returns:
        List các embeddings
    """
    return embeddings.embed_documents(texts)

def get_chroma_vectorstore(persist_directory: Optional[str] = None, collection_name: str = "documents") -> Chroma:
    """
    Tạo hoặc load Chroma vector store
    
    Args:
        persist_directory: Đường dẫn thư mục lưu trữ vector store. 
                          Mặc định sử dụng VECTOR_STORE_PATH
        collection_name: Tên collection trong Chroma
        
    Returns:
        Chroma vector store instance
    """
    if persist_directory is None:
        persist_directory = VECTOR_STORE_PATH
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(persist_directory, exist_ok=True)
    
    # Tạo hoặc load Chroma vector store
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    
    return vectorstore

def create_chroma_vectorstore(texts: List[str], metadatas: Optional[List[dict]] = None, 
                              ids: Optional[List[str]] = None,
                              persist_directory: Optional[str] = None, 
                              collection_name: str = "documents") -> Chroma:
    """
    Tạo Chroma vector store mới từ danh sách texts
    
    Args:
        texts: List các đoạn text cần lưu vào vector store
        metadatas: List các metadata tương ứng với mỗi text (optional)
        ids: List các ID tương ứng với mỗi text (optional)
        persist_directory: Đường dẫn thư mục lưu trữ vector store
        collection_name: Tên collection trong Chroma
        
    Returns:
        Chroma vector store instance đã được lưu
    """
    if persist_directory is None:
        persist_directory = VECTOR_STORE_PATH
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(persist_directory, exist_ok=True)
    
    # Tạo Chroma vector store từ texts
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    
    return vectorstore

