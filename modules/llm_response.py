from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from modules.config import OPENROUTER_API_KEY
from typing import List
import os


def format_documents(docs: List[Document]) -> str:
    """
    Format danh sách Document objects thành string để sử dụng trong prompt
    
    Args:
        docs: List các Document objects từ retriever
        
    Returns:
        String chứa nội dung của tất cả documents, được join bằng "\n\n"
    """
    return "\n\n".join(doc.page_content for doc in docs)


def get_llm_response(retriever: BaseRetriever, question: str, 
                     model_name: str) -> str:
    """
    Tạo phản hồi từ LLM dựa trên retriever và câu hỏi
    
    Args:
        retriever: BaseRetriever instance để tìm kiếm context
        question: Câu hỏi cần trả lời
        model_name: Tên model LLM (bắt buộc)
        
    Returns:
        Câu trả lời từ LLM dựa trên context được retrieve
        
    Raises:
        ValueError: Nếu API key chưa được cấu hình, retriever không tìm thấy context, hoặc model không hợp lệ
        Exception: Nếu có lỗi khác khi gọi LLM
    """
    # Kiểm tra API key
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "❌ OPENROUTER_API_KEY không được tìm thấy.\n"
            "Vui lòng kiểm tra file .env và đảm bảo OPENROUTER_API_KEY đã được set."
        )
    
    # Kiểm tra retriever có tìm thấy documents không
    # Trong LangChain mới, retriever là Runnable và dùng invoke() thay vì get_relevant_documents()
    try:
        docs = retriever.invoke(question)
        if not docs or len(docs) == 0:
            raise ValueError(
                "❌ Không tìm thấy nội dung liên quan trong tài liệu.\n\n"
                "💡 Có thể:\n"
                "   - Câu hỏi không khớp với nội dung PDF\n"
                "   - PDF chưa được xử lý đúng cách\n"
                "   - Thử đặt câu hỏi khác hoặc upload lại PDF"
            )
    except Exception as retriever_error:
        error_msg = str(retriever_error)
        if "not found" in error_msg.lower() or "empty" in error_msg.lower():
            raise ValueError(
                "❌ Không tìm thấy tài liệu trong vector store.\n\n"
                "💡 Vui lòng upload PDF lại để tạo vector store mới."
            ) from retriever_error
        else:
            raise ValueError(
                f"❌ Lỗi khi tìm kiếm trong tài liệu: {error_msg}"
            ) from retriever_error
    
    # Đọc prompt template từ file
    prompt_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "rag_prompt_universal.txt")
    with open(prompt_file_path, "r", encoding="utf-8") as f:
        template = f.read()
    
    # Tạo prompt template
    prompt = ChatPromptTemplate.from_template(template)
    
    # Khởi tạo ChatOpenAI model với headers cho OpenRouter
    try:
        model = ChatOpenAI(
            model=model_name,
            openai_api_key=OPENROUTER_API_KEY,  # type: ignore
            openai_api_base="https://openrouter.ai/api/v1",  # type: ignore
            default_headers={
                "HTTP-Referer": "",  # Optional: Site URL for rankings
                "X-Title": "AI Chatbot PDF"  # Optional: Site title for rankings
            }
        )
    except Exception as e:
        raise ValueError(
            f"❌ Lỗi khi khởi tạo model: {str(e)}\n"
            f"Model '{model_name}' có thể không hợp lệ hoặc không khả dụng trên OpenRouter."
        ) from e
    
    # Tạo chain: retriever -> format documents -> prompt -> model -> output parser
    # Retriever trả về list Document objects, cần format thành string cho prompt
    chain = (
        {
            "context": retriever | RunnableLambda(format_documents),
            "question": RunnablePassthrough()
        }
        | prompt
        | model
        | StrOutputParser()
    )
    
    # Invoke chain với question
    try:
        response = chain.invoke(question)
        
        # Kiểm tra response có rỗng không
        if not response or (isinstance(response, str) and response.strip() == ""):
            raise ValueError(
                f"❌ LLM trả về response rỗng.\n\n"
                f"💡 Có thể:\n"
                f"   - Model '{model_name}' không được phép truy cập trên OpenRouter\n"
                f"   - API key chưa được cấu hình đúng quyền\n"
                f"   - Model không hỗ trợ hoặc đã bị gỡ\n\n"
                f"🔍 Kiểm tra:\n"
                f"1. Vào https://openrouter.ai/dashboard/settings/api-keys\n"
                f"2. Chọn API key → Edit → Bật 'Allow Paid Models'\n"
                f"3. Thử model khác: qwen/qwen-2.5-32b-instruct hoặc openai/gpt-4o"
            )
        
        return response
    except ValueError:
        # Re-raise ValueError để giữ nguyên thông báo lỗi
        raise
    except Exception as e:
        error_msg = str(e)
        # Xử lý lỗi model không hợp lệ
        if "not a valid model" in error_msg.lower() or "400" in error_msg or ("model" in error_msg.lower() and "not found" in error_msg.lower()):
            raise ValueError(
                f"❌ Model không hợp lệ: {error_msg}\n\n"
                f"Model '{model_name}' không khả dụng trên OpenRouter.\n\n"
                "💡 Thử các model sau:\n"
                "   - qwen/qwen-2.5-72b-instruct (Qwen - khuyến nghị)\n"
                "   - qwen/qwen-2.5-32b-instruct (Qwen)\n"
                "   - openai/gpt-4o (OpenAI)\n"
                "   - openai/gpt-4-turbo (OpenAI)\n"
                "   - anthropic/claude-3.5-sonnet (Anthropic)\n\n"
                "Xem danh sách đầy đủ tại: https://openrouter.ai/models"
            ) from e
        elif "401" in error_msg or "403" in error_msg or "unauthorized" in error_msg.lower():
            raise ValueError(
                f"❌ Lỗi xác thực API: {error_msg}\n\n"
                "🔍 Các bước khắc phục:\n"
                "1. Kiểm tra OPENROUTER_API_KEY trong file .env có đúng không\n"
                "2. Vào https://openrouter.ai/dashboard/settings/api-keys và:\n"
                "   - Chọn API key của bạn → Edit\n"
                "   - Bật 'Allow Free Models' và 'Allow Paid Models'\n"
                "   - Chọn provider phù hợp\n"
                "   - Save changes\n"
                f"3. Kiểm tra model '{model_name}' có khả dụng trên OpenRouter không"
            ) from e
        else:
            raise Exception(
                f"❌ Lỗi khi gọi LLM: {error_msg}\n\n"
                "Vui lòng kiểm tra:\n"
                "- Kết nối mạng\n"
                "- Cấu hình API key\n"
                "- Model có khả dụng không\n"
                f"- Chi tiết lỗi: {error_msg}"
            ) from e

