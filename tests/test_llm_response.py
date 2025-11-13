"""
Unit tests cho modules/llm_response.py
"""
import pytest
import os
from modules.llm_response import format_documents, get_llm_response
from langchain_core.documents import Document


class TestLLMResponse:
    """Test các functions trong llm_response module"""
    
    def test_format_documents_empty_list(self):
        """Test format_documents với list rỗng"""
        result = format_documents([])
        assert isinstance(result, str)
        assert result == ""
    
    def test_format_documents_single_doc(self):
        """Test format_documents với một document"""
        docs = [Document(page_content="Test content")]
        result = format_documents(docs)
        
        assert isinstance(result, str)
        assert result == "Test content"
    
    def test_format_documents_multiple_docs(self):
        """Test format_documents với nhiều documents"""
        docs = [
            Document(page_content="Content 1"),
            Document(page_content="Content 2"),
            Document(page_content="Content 3")
        ]
        result = format_documents(docs)
        
        assert isinstance(result, str)
        assert "Content 1" in result
        assert "Content 2" in result
        assert "Content 3" in result
        # Kiểm tra format đúng (join bằng "\n\n")
        assert result.count("\n\n") == 2
    
    @pytest.mark.skipif(
        os.getenv("OPENROUTER_API_KEY") is None,
        reason="Cần OPENROUTER_API_KEY để test LLM response"
    )
    def test_get_llm_response_requires_model_name(self, temp_dir, sample_chunks):
        """Test get_llm_response yêu cầu model_name"""
        # Tạo vector store và retriever
        from modules.embedding import create_chroma_vectorstore
        from modules.retriever import load_retriever
        
        create_chroma_vectorstore(
            texts=sample_chunks,
            persist_directory=temp_dir
        )
        retriever = load_retriever(persist_directory=temp_dir)
        
        # Test với model_name
        response = get_llm_response(
            retriever=retriever,
            question="What is this about?",
            model_name="Qwen/Qwen3"
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_format_documents_preserves_content(self):
        """Test format_documents giữ nguyên nội dung"""
        content1 = "First document content"
        content2 = "Second document content"
        docs = [
            Document(page_content=content1),
            Document(page_content=content2)
        ]
        
        result = format_documents(docs)
        
        assert content1 in result
        assert content2 in result

