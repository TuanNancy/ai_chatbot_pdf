"""
Unit tests cho modules/retriever.py
"""
import pytest
import os
from modules.retriever import load_retriever
from langchain_core.retrievers import BaseRetriever


class TestRetriever:
    """Test các functions trong retriever module"""
    
    @pytest.mark.skipif(
        os.getenv("OPENROUTER_API_KEY") is None,
        reason="Cần OPENROUTER_API_KEY để test retriever"
    )
    def test_load_retriever_returns_base_retriever(self, temp_dir, sample_chunks):
        """Test load_retriever trả về BaseRetriever instance"""
        # Tạo vector store trước
        from modules.embedding import create_chroma_vectorstore
        create_chroma_vectorstore(
            texts=sample_chunks,
            persist_directory=temp_dir
        )
        
        # Load retriever
        retriever = load_retriever(persist_directory=temp_dir)
        
        assert isinstance(retriever, BaseRetriever)
    
    @pytest.mark.skipif(
        os.getenv("OPENROUTER_API_KEY") is None,
        reason="Cần OPENROUTER_API_KEY để test retriever"
    )
    def test_load_retriever_with_search_kwargs(self, temp_dir, sample_chunks):
        """Test load_retriever với search_kwargs"""
        # Tạo vector store trước
        from modules.embedding import create_chroma_vectorstore
        create_chroma_vectorstore(
            texts=sample_chunks,
            persist_directory=temp_dir
        )
        
        # Load retriever với search_kwargs
        retriever = load_retriever(
            persist_directory=temp_dir,
            search_kwargs={"k": 2}
        )
        
        assert isinstance(retriever, BaseRetriever)
        
        # Test retriever có thể retrieve documents
        docs = retriever.invoke("Chunk")
        assert isinstance(docs, list)
        assert len(docs) <= 2  # k=2
    
    @pytest.mark.skipif(
        os.getenv("OPENROUTER_API_KEY") is None,
        reason="Cần OPENROUTER_API_KEY để test retriever"
    )
    def test_load_retriever_with_custom_collection(self, temp_dir, sample_chunks):
        """Test load_retriever với collection_name tùy chỉnh"""
        # Tạo vector store với collection name tùy chỉnh
        from modules.embedding import create_chroma_vectorstore
        create_chroma_vectorstore(
            texts=sample_chunks,
            persist_directory=temp_dir,
            collection_name="test_collection"
        )
        
        # Load retriever với cùng collection name
        retriever = load_retriever(
            persist_directory=temp_dir,
            collection_name="test_collection"
        )
        
        assert isinstance(retriever, BaseRetriever)

