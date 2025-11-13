import os
from dotenv import load_dotenv
from pathlib import Path

# Tìm đường dẫn đến thư mục gốc project (nơi có file .env)
# Đi từ modules/config.py lên 1 level để đến thư mục gốc
PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# Load biến môi trường từ file .env
# Sử dụng đường dẫn tuyệt đối để đảm bảo tìm đúng file
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE, override=True)
else:
    # Nếu không tìm thấy .env ở thư mục gốc, thử load từ thư mục hiện tại
    load_dotenv(override=True)

# API Keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Embedding model từ OpenRouter
# Các model khả dụng trên OpenRouter:
# - "qwen/qwen3-embedding-0.6b" (Qwen - đang sử dụng)
# - "text-embedding-ada-002" (OpenAI)
# - "text-embedding-3-small" (OpenAI)
# - "text-embedding-3-large" (OpenAI)
# - "nomic-embed-text-v1" (Nomic)
EMBEDDING_MODEL_NAME = "qwen/qwen3-embedding-0.6b"  # Model đang sử dụng

# Cấu hình chung
DATA_DIR = "data"
UPLOAD_DIR = os.path.join(DATA_DIR, "uploaded_docs")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")

