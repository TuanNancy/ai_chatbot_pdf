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

# Debug: Kiểm tra xem API key có được load không (chỉ hiển thị trong development)
if not OPENROUTER_API_KEY:
    import sys
    if not sys.argv[0].endswith('pytest'):
        print(f"⚠️ Cảnh báo: OPENROUTER_API_KEY không được tìm thấy!")
        print(f"   Đường dẫn file .env mong đợi: {ENV_FILE}")
        print(f"   File .env tồn tại: {ENV_FILE.exists()}")
        if ENV_FILE.exists():
            print(f"   Nội dung file .env (ẩn API key):")
            try:
                with open(ENV_FILE, 'r', encoding='utf-8') as f:
                    for line in f:
                        if 'OPENROUTER_API_KEY' in line:
                            # Ẩn giá trị API key
                            parts = line.split('=', 1)
                            if len(parts) == 2:
                                print(f"   {parts[0]}=***")
                            else:
                                print(f"   {line.strip()}")
                        else:
                            print(f"   {line.strip()}")
            except Exception as e:
                print(f"   Không thể đọc file .env: {e}")

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

