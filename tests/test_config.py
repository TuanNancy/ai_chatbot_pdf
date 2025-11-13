"""
Unit tests cho modules/config.py
"""
import pytest
import os
from modules import config


class TestConfig:
    """Test các cấu hình trong config module"""
    
    def test_data_dir_exists(self):
        """Test DATA_DIR được định nghĩa"""
        assert hasattr(config, 'DATA_DIR')
        assert isinstance(config.DATA_DIR, str)
        assert config.DATA_DIR == "data"
    
    def test_upload_dir_exists(self):
        """Test UPLOAD_DIR được định nghĩa"""
        assert hasattr(config, 'UPLOAD_DIR')
        assert isinstance(config.UPLOAD_DIR, str)
        assert config.UPLOAD_DIR == os.path.join("data", "uploaded_docs")
    
    def test_vector_store_dir_exists(self):
        """Test VECTOR_STORE_DIR được định nghĩa"""
        assert hasattr(config, 'VECTOR_STORE_DIR')
        assert isinstance(config.VECTOR_STORE_DIR, str)
        assert config.VECTOR_STORE_DIR == os.path.join("data", "vector_store")
    
    def test_embedding_model_name_exists(self):
        """Test EMBEDDING_MODEL_NAME được định nghĩa"""
        assert hasattr(config, 'EMBEDDING_MODEL_NAME')
        assert isinstance(config.EMBEDDING_MODEL_NAME, str)
        assert len(config.EMBEDDING_MODEL_NAME) > 0
    
    def test_openrouter_api_key_exists(self):
        """Test OPENROUTER_API_KEY được định nghĩa (có thể None nếu chưa set env)"""
        assert hasattr(config, 'OPENROUTER_API_KEY')
        # API key có thể None nếu chưa được set trong .env
        assert config.OPENROUTER_API_KEY is None or isinstance(config.OPENROUTER_API_KEY, str)

