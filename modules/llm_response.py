from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from modules.config import OPENROUTER_API_KEY
from typing import Optional


def get_llm_response(retriever: BaseRetriever, question: str, 
                     model_name: str,
                     template: Optional[str] = None) -> str:
    """
    Tạo phản hồi từ LLM dựa trên retriever và câu hỏi
    
    Args:
        retriever: BaseRetriever instance để tìm kiếm context
        question: Câu hỏi cần trả lời
        model_name: Tên model LLM (bắt buộc)
        template: Template cho prompt (optional, có template mặc định)
        
    Returns:
        Câu trả lời từ LLM dựa trên context được retrieve
    """
    
    # Template mặc định nếu không được cung cấp
    if template is None:
        template = """
Answer the question based only on the following context:

{context}

Question: {question}
"""
    
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

