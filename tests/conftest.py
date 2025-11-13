"""
Shared fixtures và utilities cho tests
"""
import pytest
import os
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Tạo thư mục tạm thời cho tests"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    # Cleanup sau khi test xong
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)


@pytest.fixture
def sample_text():
    """Sample text để test text splitting"""
    return """
    Đây là một đoạn văn bản mẫu để test các chức năng của PDF loader.
    Văn bản này sẽ được chia thành các chunks nhỏ hơn để test RecursiveCharacterTextSplitter.
    Mỗi chunk sẽ có kích thước khoảng 500 ký tự với overlap 50 ký tự.
    Điều này giúp đảm bảo rằng không có thông tin quan trọng nào bị mất khi chia nhỏ văn bản.
    """ * 3  # Nhân lên để có đủ text cho nhiều chunks


@pytest.fixture
def sample_chunks():
    """Sample chunks để test embedding và vector store"""
    return [
        "Chunk 1: Đây là nội dung của chunk đầu tiên.",
        "Chunk 2: Đây là nội dung của chunk thứ hai.",
        "Chunk 3: Đây là nội dung của chunk thứ ba."
    ]


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables cho tests"""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_api_key_12345")
    # Note: Config sẽ tự động load env vars khi import

