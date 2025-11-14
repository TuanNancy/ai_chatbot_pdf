from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from modules.config import OPENROUTER_API_KEY, EMBEDDING_MODEL_NAME, VECTOR_STORE_DIR
from typing import List, Optional
import os
from openai import OpenAI

# Custom Embeddings class để đảm bảo headers được truyền đúng cách cho OpenRouter
# Sử dụng OpenAI client trực tiếp như trong sample code của OpenRouter
class OpenRouterEmbeddings(Embeddings):
    """Custom Embeddings class cho OpenRouter API với Qwen embedding model"""
    
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key
        
        # Khởi tạo OpenAI client với OpenRouter base URL
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Tạo embeddings cho danh sách documents"""
        try:
            response = self.client.embeddings.create(
                extra_headers={
                    "HTTP-Referer": "",  # Optional: Site URL for rankings
                    "X-Title": "AI Chatbot PDF"  # Optional: Site title for rankings
                },
                model=self.model_name,
                input=texts,
                encoding_format="float"
            )
            # Trả về list các embeddings
            return [item.embedding for item in response.data]
        except Exception as e:
            error_msg = str(e)
            if "No embedding data received" in error_msg or "401" in error_msg or "403" in error_msg:
                raise ValueError(
                    f"❌ Lỗi xác thực API: {error_msg}\n\n"
                    "🔍 Các bước khắc phục:\n"
                    "1. Kiểm tra OPENROUTER_API_KEY trong file .env có đúng không\n"
                    "2. Vào https://openrouter.ai/dashboard/settings/api-keys và:\n"
                    "   - Chọn API key của bạn → Edit\n"
                    "   - Bật 'Allow Free Models' và 'Allow Paid Models'\n"
                    "   - Chọn provider phù hợp (Qwen cho qwen/qwen3-embedding-0.6b)\n"
                    "   - Save changes\n"
                    f"3. Kiểm tra model '{self.model_name}' có khả dụng trên OpenRouter không"
                ) from e
            else:
                raise
    
    def embed_query(self, text: str) -> List[float]:
        """Tạo embedding cho một query"""
        try:
            response = self.client.embeddings.create(
                extra_headers={
                    "HTTP-Referer": "",  # Optional: Site URL for rankings
                    "X-Title": "AI Chatbot PDF"  # Optional: Site title for rankings
                },
                model=self.model_name,
                input=text,
                encoding_format="float"
            )
            # Trả về embedding đầu tiên
            if response.data and len(response.data) > 0:
                return response.data[0].embedding
            else:
                raise ValueError("API trả về embedding rỗng")
        except Exception as e:
            error_msg = str(e)
            if "No embedding data received" in error_msg or "401" in error_msg or "403" in error_msg:
                raise ValueError(
                    f"❌ Lỗi xác thực API: {error_msg}\n\n"
                    "🔍 Các bước khắc phục:\n"
                    "1. Kiểm tra OPENROUTER_API_KEY trong file .env có đúng không\n"
                    "2. Vào https://openrouter.ai/dashboard/settings/api-keys và:\n"
                    "   - Chọn API key của bạn → Edit\n"
                    "   - Bật 'Allow Free Models' và 'Allow Paid Models'\n"
                    "   - Chọn provider phù hợp (Qwen cho qwen/qwen3-embedding-0.6b)\n"
                    "   - Save changes\n"
                    f"3. Kiểm tra model '{self.model_name}' có khả dụng trên OpenRouter không"
                ) from e
            else:
                raise

# Khởi tạo embeddings instance
embeddings = None
if OPENROUTER_API_KEY:
    try:
        embeddings = OpenRouterEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            api_key=OPENROUTER_API_KEY
        )
    except Exception:
        # Silently fail khi khởi tạo để có thể test mà không cần API key
        pass

def get_embeddings():
    """
    Trả về instance của OpenRouterEmbeddings đã được cấu hình
    
    Raises:
        ValueError: Nếu API key chưa được cấu hình
    """
    if embeddings is None:
        raise ValueError(
            "OPENROUTER_API_KEY không được tìm thấy. "

        )
    return embeddings

def embed_text(text: str) -> List[float]:
    """
    Tạo embedding cho một đoạn text
    
    Args:
        text: Đoạn text cần tạo embedding
        
    Returns:
        List các giá trị embedding
        
    Raises:
        ValueError: Nếu API key chưa được cấu hình
    """
    if embeddings is None:
        raise ValueError(
            "OPENROUTER_API_KEY không được tìm thấy. "
            
        )
    return embeddings.embed_query(text)

def embed_documents(texts: List[str]) -> List[List[float]]:
    """
    Tạo embeddings cho nhiều documents
    
    Args:
        texts: List các đoạn text cần tạo embedding
        
    Returns:
        List các embeddings
        
    Raises:
        ValueError: Nếu API key chưa được cấu hình
    """
    if embeddings is None:
        raise ValueError(
            "OPENROUTER_API_KEY không được tìm thấy. "
            
        )
    return embeddings.embed_documents(texts)

def get_chroma_vectorstore(persist_directory: Optional[str] = None, collection_name: str = "documents") -> Chroma:
    """
    Tạo hoặc load Chroma vector store
    
    Args:
        persist_directory: Đường dẫn thư mục lưu trữ vector store. 
                          Mặc định sử dụng VECTOR_STORE_DIR từ config
        collection_name: Tên collection trong Chroma
        
    Returns:
        Chroma vector store instance
        
    Raises:
        ValueError: Nếu API key chưa được cấu hình
    """
    if persist_directory is None:
        persist_directory = VECTOR_STORE_DIR
    
    # Lấy embeddings instance (sẽ raise ValueError nếu API key chưa được set)
    embedding_instance = get_embeddings()
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(persist_directory, exist_ok=True)
    
    # Tạo hoặc load Chroma vector store
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_instance,
        collection_name=collection_name
    )
    
    return vectorstore

def create_chroma_vectorstore(texts: List[str], metadatas: Optional[List[dict]] = None, 
                              ids: Optional[List[str]] = None,
                              persist_directory: Optional[str] = None, 
                              collection_name: str = "documents") -> Chroma:
    """
    Tạo Chroma vector store mới từ danh sách texts
    
    Args:
        texts: List các đoạn text cần lưu vào vector store
        metadatas: List các metadata tương ứng với mỗi text (optional)
        ids: List các ID tương ứng với mỗi text (optional)
        persist_directory: Đường dẫn thư mục lưu trữ vector store
        collection_name: Tên collection trong Chroma
        
    Returns:
        Chroma vector store instance đã được lưu
        
    Raises:
        ValueError: Nếu API key không hợp lệ hoặc không có
        Exception: Nếu có lỗi khi tạo embeddings từ API
    """
    if persist_directory is None:
        persist_directory = VECTOR_STORE_DIR
    
    # Kiểm tra API key
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY không được tìm thấy. "
            "Vui lòng kiểm tra file .env và đảm bảo OPENROUTER_API_KEY đã được set."
        )
    
    # Kiểm tra texts không rỗng
    if not texts or len(texts) == 0:
        raise ValueError("Danh sách texts không được rỗng")
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(persist_directory, exist_ok=True)
    
    try:
        # Lấy embeddings instance (sẽ raise ValueError nếu API key chưa được set)
        embedding_instance = get_embeddings()
        
        # Test embedding với một text nhỏ trước để verify API hoạt động
        try:
            test_result = embedding_instance.embed_query("test")
            if not test_result or len(test_result) == 0:
                raise ValueError("API trả về embedding rỗng")
        except Exception as test_error:
            error_msg = str(test_error)
            if "No embedding data received" in error_msg or "401" in error_msg or "403" in error_msg:
                raise ValueError(
                    f"❌ Lỗi xác thực API: {error_msg}\n\n"
                    "🔍 Các bước khắc phục:\n"
                    "1. Kiểm tra OPENROUTER_API_KEY trong file .env có đúng không\n"
                    "2. Vào https://openrouter.ai/dashboard/settings/api-keys và:\n"
                    "   - Chọn API key của bạn → Edit\n"
                    "   - Bật 'Allow Free Models' và 'Allow Paid Models'\n"
                    "   - Chọn provider phù hợp (Qwen cho qwen/qwen3-embedding-0.6b)\n"
                    "   - Save changes\n"
                    f"3. Kiểm tra model '{EMBEDDING_MODEL_NAME}' có khả dụng trên OpenRouter không\n"
                    "4. Thử model khác trong modules/config.py:\n"
                    "   - qwen/qwen3-embedding-0.6b (Qwen - đang sử dụng)\n"
                    "   - text-embedding-ada-002 (OpenAI)\n"
                    "   - text-embedding-3-small (OpenAI)\n"
                    "   - nomic-embed-text-v1 (Nomic)"
                ) from test_error
            else:
                raise
        
        # Xóa collection cũ nếu tồn tại để đảm bảo tạo mới hoàn toàn
        try:
            import chromadb
            client = chromadb.PersistentClient(path=persist_directory)
            try:
                client.delete_collection(name=collection_name)
            except Exception:
                # Collection không tồn tại, bỏ qua
                pass
        except Exception:
            # Không thể xóa collection, tiếp tục tạo mới (ChromaDB sẽ xử lý)
            pass
        
        # Tạo Chroma vector store từ texts
        try:
            vectorstore = Chroma.from_texts(
                texts=texts,
                embedding=embedding_instance,
                metadatas=metadatas,
                ids=ids,
                persist_directory=persist_directory,
                collection_name=collection_name
            )
            return vectorstore
        except Exception as create_error:
            error_msg = str(create_error)
            # Xử lý lỗi dimension mismatch - collection cũ có dimension khác
            if "dimension" in error_msg.lower() or "expecting embedding" in error_msg.lower():
                # Xóa collection cũ và tạo lại
                try:
                    import chromadb
                    client = chromadb.PersistentClient(path=persist_directory)
                    try:
                        client.delete_collection(name=collection_name)
                    except Exception:
                        # Collection có thể không tồn tại, bỏ qua
                        pass
                    
                    # Tạo lại vector store với collection mới
                    vectorstore = Chroma.from_texts(
                        texts=texts,
                        embedding=embedding_instance,
                        metadatas=metadatas,
                        ids=ids,
                        persist_directory=persist_directory,
                        collection_name=collection_name
                    )
                    return vectorstore
                except Exception as retry_error:
                    raise Exception(
                        f"❌ Lỗi khi tạo vector store (dimension mismatch): {error_msg}\n\n"
                        f"Đã thử xóa collection cũ nhưng vẫn lỗi: {str(retry_error)}\n\n"
                        "💡 Giải pháp:\n"
                        f"1. Xóa thủ công thư mục vector store: {persist_directory}\n"
                        "2. Hoặc đổi collection_name trong code để tạo collection mới"
                    ) from retry_error
            else:
                # Re-raise các lỗi khác
                raise
        
    except ValueError:
        # Re-raise ValueError để giữ nguyên thông báo lỗi chi tiết
        raise
    except Exception as e:
        error_msg = str(e)
        # Cải thiện thông báo lỗi
        if "No embedding data received" in error_msg or "401" in error_msg or "403" in error_msg:
            raise ValueError(
                f"❌ Lỗi xác thực API: {error_msg}\n\n"
                "🔍 Các bước khắc phục:\n"
                "1. Kiểm tra OPENROUTER_API_KEY trong file .env\n"
                "2. Vào https://openrouter.ai/dashboard/settings/api-keys:\n"
                "   - Edit API key → Bật 'Allow Free Models' và 'Allow Paid Models'\n"
                "   - Chọn provider phù hợp\n"
                f"3. Kiểm tra model '{EMBEDDING_MODEL_NAME}' có khả dụng không"
            ) from e
        elif "404" in error_msg or "model" in error_msg.lower() or "not found" in error_msg.lower():
            raise ValueError(
                f"❌ Model không tìm thấy: {error_msg}\n\n"
                f"Model hiện tại: '{EMBEDDING_MODEL_NAME}'\n\n"
                "💡 Thử đổi model trong modules/config.py thành:\n"
                "   - qwen/qwen3-embedding-0.6b (Qwen - đang sử dụng)\n"
                "   - text-embedding-ada-002 (OpenAI)\n"
                "   - text-embedding-3-small (OpenAI)\n"
                "   - nomic-embed-text-v1 (Nomic)"
            ) from e
        elif "dimension" in error_msg.lower() or "expecting embedding" in error_msg.lower():
            # Lỗi này đã được xử lý ở trên, nhưng nếu vẫn xảy ra thì báo lỗi rõ ràng
            raise Exception(
                f"❌ Lỗi dimension không khớp: {error_msg}\n\n"
                "💡 Collection cũ có dimension khác với model hiện tại.\n"
                f"Đã thử xóa collection cũ nhưng vẫn lỗi.\n\n"
                "Giải pháp:\n"
                f"1. Xóa thủ công thư mục: {persist_directory}\n"
                "2. Hoặc đổi collection_name để tạo collection mới"
            ) from e
        else:
            raise Exception(
                f"❌ Lỗi khi tạo vector store: {error_msg}\n\n"
                "Vui lòng kiểm tra:\n"
                "- Kết nối mạng\n"
                "- Cấu hình API\n"
                "- Xem thêm chi tiết trong ENV_SETUP.md"
            ) from e

