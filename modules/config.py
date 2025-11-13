import os
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()

# API Keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Embedding model từ OpenRouter
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

# Cấu hình chung
DATA_DIR = "data"
UPLOAD_DIR = os.path.join(DATA_DIR, "uploaded_docs")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")

