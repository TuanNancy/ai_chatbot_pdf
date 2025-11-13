import streamlit as st
import os
import tempfile
from modules.pdf_loader import load_and_split_pdf
from modules.embedding import create_chroma_vectorstore
from modules.retriever import load_retriever
from modules.llm_response import get_llm_response
from modules.config import UPLOAD_DIR, VECTOR_STORE_DIR

# Cấu hình trang
st.set_page_config(
    page_title="AI Chatbot PDF",
    page_icon="📚",
    layout="wide"
)

# Khởi tạo session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore_loaded" not in st.session_state:
    st.session_state.vectorstore_loaded = False

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Model LLM mặc định (có thể thay đổi)
DEFAULT_LLM_MODEL = "Qwen/Qwen3"

# Tạo thư mục upload nếu chưa tồn tại
os.makedirs(UPLOAD_DIR, exist_ok=True)


def process_pdf(uploaded_file):
    """Xử lý file PDF: lưu, load, split và tạo vector store"""
    tmp_file_path = None
    try:
        # Lưu file tạm thời
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=UPLOAD_DIR) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        # Hiển thị progress
        with st.spinner("Đang xử lý PDF..."):
            # Load và split PDF thành chunks
            st.info("Đang trích xuất text từ PDF...")
            chunks = load_and_split_pdf(tmp_file_path)
            
            if not chunks:
                st.error("Không thể trích xuất text từ PDF. Vui lòng kiểm tra lại file.")
                return False
            
            st.info(f"Đã chia PDF thành {len(chunks)} chunks. Đang tạo vector store...")
            
            # Tạo metadata cho mỗi chunk
            metadatas = [{"source": uploaded_file.name, "chunk_index": i} for i in range(len(chunks))]
            
            # Tạo vector store từ chunks
            vectorstore = create_chroma_vectorstore(
                texts=chunks,
                metadatas=metadatas,
                persist_directory=VECTOR_STORE_DIR
            )
            
            # Load retriever từ vector store
            st.session_state.retriever = load_retriever(
                persist_directory=VECTOR_STORE_DIR,
                search_kwargs={"k": 4}
            )
            
            st.session_state.vectorstore_loaded = True
            
            st.success(f"✅ Đã tải PDF thành công! ({len(chunks)} chunks)")
            return True
            
    except Exception as e:
        st.error(f"Lỗi khi xử lý PDF: {str(e)}")
        return False
    finally:
        # File được giữ lại trong UPLOAD_DIR để có thể sử dụng lại
        # Không cần xóa file tạm
        pass


# Sidebar cho upload PDF
with st.sidebar:
    st.header("📤 Upload PDF")
    
    uploaded_file = st.file_uploader(
        "Chọn file PDF",
        type=["pdf"],
        help="Kéo thả file PDF vào đây hoặc click để chọn file"
    )
    
    if uploaded_file is not None:
        if st.button("📥 Tải PDF lên", type="primary"):
            if process_pdf(uploaded_file):
                # Xóa lịch sử chat cũ khi upload PDF mới
                st.session_state.messages = []
                st.rerun()
    
    st.divider()
    
    # Cấu hình model LLM
    st.header("⚙️ Cấu hình")
    llm_model = st.text_input(
        "Model LLM",
        value=DEFAULT_LLM_MODEL,
        help="Nhập tên model LLM từ OpenRouter"
    )
    
    st.divider()
    
    # Hiển thị trạng thái
    st.header("📊 Trạng thái")
    if st.session_state.vectorstore_loaded:
        st.success("✅ PDF đã được tải")
    else:
        st.warning("⚠️ Chưa có PDF nào được tải")
    
    if st.button("🗑️ Xóa lịch sử chat"):
        st.session_state.messages = []
        st.rerun()


# Main content - Chat interface
st.title("📚 AI Chatbot PDF")
st.markdown("---")

# Hiển thị lịch sử chat
chat_container = st.container()

with chat_container:
    # Hiển thị các tin nhắn đã lưu
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Input chat
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    # Kiểm tra xem đã có PDF chưa
    if not st.session_state.vectorstore_loaded:
        st.warning("⚠️ Vui lòng upload PDF trước khi đặt câu hỏi!")
        st.stop()
    
    # Thêm câu hỏi của người dùng vào lịch sử
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Hiển thị câu hỏi
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Hiển thị phản hồi đang xử lý
    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ..."):
            try:
                # Lấy phản hồi từ LLM
                # Sửa lỗi: nếu chưa upload PDF (chưa có retriever), báo lỗi rõ ràng
                if not hasattr(st.session_state, "retriever") or st.session_state.retriever is None:
                    raise RuntimeError("❌ Không tìm thấy tài liệu hoặc retriever. Vui lòng tải PDF trước.")
                response = get_llm_response(
                    retriever=st.session_state.retriever,
                    question=prompt,
                    model_name=llm_model
                )
                # Hiển thị phản hồi
                st.markdown(response)
                
                # Lưu phản hồi vào lịch sử
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"❌ Lỗi: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.caption("💡 Tip: Upload PDF và đặt câu hỏi về nội dung trong tài liệu")

