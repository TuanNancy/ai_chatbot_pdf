from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.retrievers import BaseRetriever
from modules.config import EMBEDDING_MODEL_NAME, OPENROUTER_API_KEY, VECTOR_STORE_DIR
from typing import Optional, Dict, Any
import os


def load_retriever(persist_directory: Optional[str] = None, 
                   collection_name: str = "documents",
                   search_kwargs: Optional[Dict[str, Any]] = None) -> BaseRetriever:
    """
    Load retriever từ Chroma vector store
    
    Args:
        persist_directory: Đường dẫn thư mục lưu trữ vector store.
                          Mặc định sử dụng VECTOR_STORE_DIR từ config
        collection_name: Tên collection trong Chroma
        search_kwargs: Các tham số tìm kiếm cho retriever (ví dụ: {"k": 4})
        
    Returns:
        BaseRetriever instance từ Chroma vector store
    """
    if persist_directory is None:
        persist_directory = VECTOR_STORE_DIR
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(persist_directory, exist_ok=True)
    
    # Khởi tạo embedding function
    embedding_function = OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        openai_api_key=OPENROUTER_API_KEY,  # type: ignore
        openai_api_base="https://openrouter.ai/api/v1"  # type: ignore
    )
    
    # Load Chroma vector store
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function,
        collection_name=collection_name
    )
    
    # Tạo retriever với search_kwargs nếu có
    if search_kwargs:
        return vectorstore.as_retriever(search_kwargs=search_kwargs)
    else:
        return vectorstore.as_retriever()

