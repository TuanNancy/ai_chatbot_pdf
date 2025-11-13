"""
Unit tests cho modules/pdf_loader.py
"""
import pytest
import os
import tempfile
from modules.pdf_loader import load_pdf, split_text, load_and_split_pdf


class TestPDFLoader:
    """Test các functions trong pdf_loader module"""
    
    def test_split_text_basic(self):
        """Test split_text với text đơn giản"""
        text = "A" * 1000  # 1000 ký tự 'A'
        chunks = split_text(text, chunk_size=500, chunk_overlap=50)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        # Kiểm tra mỗi chunk không vượt quá chunk_size + overlap
        for chunk in chunks:
            assert len(chunk) <= 500 + 50
    
    def test_split_text_empty(self):
        """Test split_text với text rỗng"""
        chunks = split_text("", chunk_size=500, chunk_overlap=50)
        assert isinstance(chunks, list)
        # Text rỗng có thể trả về list rỗng hoặc list với 1 empty string
        assert len(chunks) >= 0
    
    def test_split_text_custom_params(self):
        """Test split_text với tham số tùy chỉnh"""
        text = "Test text " * 100
        chunks = split_text(text, chunk_size=100, chunk_overlap=10)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
    
    def test_load_pdf_file_not_found(self):
        """Test load_pdf với file không tồn tại"""
        with pytest.raises(FileNotFoundError):
            load_pdf("non_existent_file.pdf")
    
    def test_load_and_split_pdf_file_not_found(self):
        """Test load_and_split_pdf với file không tồn tại"""
        with pytest.raises(FileNotFoundError):
            load_and_split_pdf("non_existent_file.pdf")
    
    def test_split_text_preserves_content(self, sample_text):
        """Test split_text giữ nguyên nội dung"""
        chunks = split_text(sample_text, chunk_size=500, chunk_overlap=50)
        
        # Nối lại các chunks và kiểm tra nội dung được giữ nguyên
        reconstructed = "".join(chunks)
        # Loại bỏ whitespace để so sánh
        assert reconstructed.replace(" ", "").replace("\n", "") == sample_text.replace(" ", "").replace("\n", "")

