import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
import os


def load_pdf(file_path: str) -> str:
    """
    Load và trích xuất text từ file PDF
    
    Args:
        file_path: Đường dẫn đến file PDF
        
    Returns:
        Text đã được trích xuất từ PDF
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File không tồn tại: {file_path}")
    
    # Mở file PDF
    doc = fitz.open(file_path)
    text_parts: List[str] = []
    
    # Duyệt qua từng trang và trích xuất text
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text()
        if isinstance(page_text, str):
            text_parts.append(page_text)
    
    # Đóng document
    doc.close()
    
    # Kết hợp tất cả text
    text = "\n".join(text_parts)
    return text


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Chia text thành các chunks sử dụng RecursiveCharacterTextSplitter
    
    Args:
        text: Text cần chia nhỏ
        chunk_size: Kích thước mỗi chunk (mặc định: 500)
        chunk_overlap: Số ký tự overlap giữa các chunks (mặc định: 50)
        
    Returns:
        List các text chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_text(text)
    return chunks


def load_and_split_pdf(file_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Load PDF và chia thành các chunks
    
    Args:
        file_path: Đường dẫn đến file PDF
        chunk_size: Kích thước mỗi chunk (mặc định: 500)
        chunk_overlap: Số ký tự overlap giữa các chunks (mặc định: 50)
        
    Returns:
        List các text chunks từ PDF
    """
    # Load text từ PDF
    text = load_pdf(file_path)
    
    # Chia text thành chunks
    chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    return chunks

