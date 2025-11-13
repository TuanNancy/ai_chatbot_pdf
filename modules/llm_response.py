from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from modules.config import OPENROUTER_API_KEY
import os


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
    """
    # Đọc prompt template từ file
    prompt_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "rag_prompt_universal.txt")
    with open(prompt_file_path, "r", encoding="utf-8") as f:
        template = f.read()
    
    # Tạo prompt template
    prompt = ChatPromptTemplate.from_template(template)
    
    # Khởi tạo ChatOpenAI model
    model = ChatOpenAI(
        model=model_name,
        openai_api_key=OPENROUTER_API_KEY,  # type: ignore
        openai_api_base="https://openrouter.ai/api/v1"  # type: ignore
    )
    
    # Tạo chain: retriever -> prompt -> model -> output parser
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    
    # Invoke chain với question
    return chain.invoke(question)

