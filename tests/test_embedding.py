"""
Unit tests cho modules/embedding.py
"""
import pytest
import os
from modules.embedding import (
    get_embeddings,
    embed_text,
    embed_documents,
    get_chroma_vectorstore,
    create_chroma_vectorstore
)
from langchain_openai import OpenAIEmbeddings


class TestEmbedding:
    """Test các functions trong embedding module"""
    
    def test_get_embeddings_returns_instance(self):
        """Test get_embeddings trả về OpenAIEmbeddings instance"""
        embeddings = get_embeddings()
        assert isinstance(embeddings, OpenAIEmbeddings)
    
    @pytest.mark.skipif(
        os.getenv("OPENROUTER_API_KEY") is None,
        reason="Cần OPENROUTER_API_KEY để test embedding"
    )
    def test_embed_text_returns_list(self):
        """Test embed_text trả về list các float"""
        text = "Test text for embedding"
        result = embed_text(text)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(x, (int, float)) for x in result)
    
    @pytest.mark.skipif(
        os.getenv("OPENROUTER_API_KEY") is None,
        reason="Cần OPENROUTER_API_KEY để test embedding"
    )
    def test_embed_documents_returns_list_of_lists(self, sample_chunks):
        """Test embed_documents trả về list các list"""
        result = embed_documents(sample_chunks)
        
        assert isinstance(result, list)
        assert len(result) == len(sample_chunks)
        assert all(isinstance(x, list) for x in result)
    
    def test_get_chroma_vectorstore_creates_instance(self, temp_dir):
        """Test get_chroma_vectorstore tạo Chroma instance"""
        vectorstore = get_chroma_vectorstore(persist_directory=temp_dir)
        
        from langchain_community.vectorstores import Chroma
        assert isinstance(vectorstore, Chroma)
    
    @pytest.mark.skipif(
        os.getenv("OPENROUTER_API_KEY") is None,
        reason="Cần OPENROUTER_API_KEY để test vector store"
    )
    def test_create_chroma_vectorstore_creates_instance(self, temp_dir, sample_chunks):
        """Test create_chroma_vectorstore tạo Chroma instance với texts"""
        vectorstore = create_chroma_vectorstore(
            texts=sample_chunks,
            persist_directory=temp_dir
        )
        
        from langchain_community.vectorstores import Chroma
        assert isinstance(vectorstore, Chroma)
        
        # Kiểm tra vectorstore có thể search
        results = vectorstore.similarity_search("Chunk", k=1)
        assert len(results) > 0
    
    def test_create_chroma_vectorstore_with_metadata(self, temp_dir, sample_chunks):
        """Test create_chroma_vectorstore với metadata"""
        metadatas = [{"source": f"test_{i}"} for i in range(len(sample_chunks))]
        
        vectorstore = create_chroma_vectorstore(
            texts=sample_chunks,
            metadatas=metadatas,
            persist_directory=temp_dir
        )
        
        from langchain_community.vectorstores import Chroma
        assert isinstance(vectorstore, Chroma)

