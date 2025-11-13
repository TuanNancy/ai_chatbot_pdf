# 📚 RAG Chatbot - Trả lời câu hỏi từ PDF

> **Một chatbot thông minh sử dụng RAG (Retrieval-Augmented Generation) để trả lời câu hỏi chỉ dựa trên nội dung PDF được upload, không bịa thông tin, không dùng kiến thức bên ngoài.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://www.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Mục tiêu dự án

Xây dựng một chatbot thông minh có thể:

- ✅ **Upload bất kỳ file PDF** (giáo trình, báo cáo tài chính, sổ tay sản phẩm, tài liệu kỹ thuật...)
- ✅ **Đọc hiểu nội dung** từ PDF bằng PyMuPDF (fitz)
- ✅ **Trả lời câu hỏi chỉ dựa trên tài liệu** đó, không bịa thông tin
- ✅ **Không dùng kiến thức bên ngoài** → RAG thực sự, đảm bảo tính chính xác
- ✅ **Giao diện web thân thiện** với Streamlit
- ✅ **Hỗ trợ nhiều LLM models** qua OpenRouter API

## 🏗️ Kiến trúc hệ thống

### RAG Pipeline Flow

```
┌─────────────┐
│   PDF File  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐      ┌──────────────────┐
│  PDF Loader     │─────▶│  Text Splitter   │
│  (PyMuPDF/fitz) │      │  (Recursive)     │
└─────────────────┘      └────────┬─────────┘
                                   │
                                   ▼
                          ┌──────────────────┐
                          │   Embeddings     │
                          │  (OpenRouter API)│
                          │  Qwen3-Embedding │
                          └────────┬─────────┘
                                   │
                                   ▼
                          ┌──────────────────┐
                          │  Vector Store    │
                          │  (Chroma DB)     │
                          └────────┬─────────┘
                                   │
                                   ▼
┌──────────────┐          ┌──────────────────┐
│ User Query   │─────────▶│    Retriever     │
└──────────────┘          │  (Similarity)     │
                           └────────┬─────────┘
                                    │
                                    ▼
                           ┌──────────────────┐
                           │   LLM Chain      │
                           │  (OpenRouter)    │
                           │  + RAG Prompt    │
                           └────────┬─────────┘
                                    │
                                    ▼
                           ┌──────────────────┐
                           │   Answer         │
                           └──────────────────┘
```

### Components

1. **PDF Processing**: PyMuPDF (fitz) để trích xuất text từ PDF
2. **Text Chunking**: RecursiveCharacterTextSplitter với overlap để giữ context
3. **Embedding**: OpenRouter API với Qwen3-Embedding-0.6B model
4. **Vector Store**: Chroma DB để lưu trữ và tìm kiếm vectors
5. **Retrieval**: Semantic search để tìm context liên quan
6. **Generation**: LLM qua OpenRouter (hỗ trợ nhiều models: Qwen, GPT-4, Claude...)

## 🛠️ Công nghệ sử dụng

### Core Stack

| Công nghệ | Phiên bản | Mục đích |
|-----------|-----------|----------|
| **Python** | 3.11+ | Ngôn ngữ lập trình chính |
| **Streamlit** | Latest | Web UI framework |
| **LangChain** | 0.1+ | RAG framework và orchestration |
| **Chroma** | Latest | Vector database |
| **PyMuPDF (fitz)** | Latest | PDF text extraction |
| **OpenRouter API** | v1 | Unified API cho embeddings và LLMs |

### Tại sao chọn các công nghệ này?

- **OpenRouter**: 
  - Unified API cho nhiều LLM providers (OpenAI, Anthropic, Qwen, Meta...)
  - Hỗ trợ cả embedding và chat models
  - Dễ dàng switch models mà không cần đổi code
  - Cost-effective với nhiều free models

- **Chroma**: 
  - Lightweight, embedded vector database
  - Không cần server riêng, persist local
  - Tích hợp tốt với LangChain
  - Performance tốt cho small-medium datasets

- **PyMuPDF (fitz)**: 
  - Fast và accurate PDF text extraction
  - Hỗ trợ tốt các loại PDF (text-based, scanned với OCR)
  - Python-native, dễ tích hợp

- **Streamlit**: 
  - Rapid prototyping và deployment
  - Built-in components cho file upload, chat interface
  - Không cần frontend knowledge

## ✨ Tính năng chính

### 🎨 User Interface

- **Upload PDF**: Drag & drop hoặc click để chọn file
- **Chat Interface**: Giao diện chat giống ChatGPT
- **Model Selection**: Chọn LLM model trong sidebar
- **Status Display**: Hiển thị trạng thái PDF và API key
- **Chat History**: Lưu lịch sử chat trong session

### 🔧 Technical Features

- **Custom Embeddings**: Custom class để tích hợp OpenRouter embeddings
- **Error Handling**: Comprehensive error handling với thông báo rõ ràng
- **Vector Store Persistence**: Lưu vector store để tái sử dụng
- **Configurable**: Dễ dàng đổi models, chunk size, search parameters
- **Type Safety**: Type hints đầy đủ cho code maintainability

### 🧪 Testing

- Unit tests cho tất cả modules
- Test fixtures và utilities
- Skip tests khi không có API key
- Coverage reporting

## 📦 Cài đặt

### Yêu cầu

- Python 3.11 hoặc cao hơn
- pip package manager
- OpenRouter API key ([Lấy tại đây](https://openrouter.ai/))

### Bước 1: Clone repository

```bash
git clone <repository-url>
cd ai_chatbot_pdf
```

### Bước 2: Tạo virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Bước 4: Cấu hình API key

Tạo file `.env` trong thư mục gốc project:

```bash
OPENROUTER_API_KEY=your_api_key_here
```

**Lưu ý**: 
- Lấy API key tại [OpenRouter Dashboard](https://openrouter.ai/dashboard/settings/api-keys)
- Bật "Allow Free Models" và "Allow Paid Models" trong API key settings
- Chọn providers phù hợp (Qwen cho embedding model)

### Bước 5: Chạy ứng dụng

```bash
streamlit run app.py
```

Ứng dụng sẽ mở tại `http://localhost:8501`

## 🚀 Cách sử dụng

### 1. Upload PDF

1. Click vào sidebar → "Upload PDF"
2. Chọn file PDF từ máy tính
3. Click "📥 Tải PDF lên"
4. Đợi hệ thống xử lý (extract text → chunking → embedding → vector store)

### 2. Đặt câu hỏi

1. Nhập câu hỏi vào ô chat ở cuối trang
2. Nhấn Enter hoặc click gửi
3. Hệ thống sẽ:
   - Tìm context liên quan trong PDF
   - Gửi context + question đến LLM
   - Trả về câu trả lời dựa trên PDF

### 3. Đổi LLM Model

1. Vào sidebar → "⚙️ Cấu hình"
2. Nhập tên model mới vào ô "Model LLM"
3. Các model khả dụng:
   - `qwen/qwen-2.5-72b-instruct` (Qwen - mạnh)
   - `qwen/qwen-2.5-32b-instruct` (Qwen - nhỏ hơn)
   - `openai/gpt-4o` (OpenAI)
   - `anthropic/claude-3.5-sonnet` (Anthropic)
   - Xem thêm tại [OpenRouter Models](https://openrouter.ai/models)

### Ví dụ câu hỏi

- "Tóm tắt nội dung chính của tài liệu này"
- "Giải thích khái niệm X trong PDF"
- "Liệt kê các bước thực hiện Y"
- "So sánh A và B trong tài liệu"

## 📁 Cấu trúc dự án

```
ai_chatbot_pdf/
│
├── app.py                      # Streamlit app chính
├── requirements.txt            # Python dependencies
├── README.md                   # Tài liệu này
├── .env                        # Environment variables (tạo mới)
│
├── modules/                    # Core modules
│   ├── config.py              # Configuration và environment variables
│   ├── pdf_loader.py          # PDF loading và text splitting
│   ├── embedding.py           # Embedding với OpenRouter API
│   ├── retriever.py           # Vector store retrieval
│   └── llm_response.py        # LLM response generation
│
├── prompts/                    # Prompt templates
│   └── rag_prompt_universal.txt  # RAG prompt template
│
├── data/                       # Data storage
│   ├── uploaded_docs/         # PDF files đã upload
│   └── vector_store/          # Chroma vector database
│
└── tests/                      # Unit tests
    ├── conftest.py            # Test fixtures
    ├── test_config.py         # Tests cho config
    ├── test_pdf_loader.py     # Tests cho PDF loader
    ├── test_embedding.py      # Tests cho embedding
    ├── test_retriever.py      # Tests cho retriever
    └── test_llm_response.py   # Tests cho LLM response
```

## 🔍 Kiến trúc code chi tiết

### Module: `pdf_loader.py`

**Chức năng**: Trích xuất text từ PDF và chia thành chunks

**Key Functions**:
- `load_pdf(file_path)`: Trích xuất text từ PDF bằng PyMuPDF
- `split_text(text, chunk_size, chunk_overlap)`: Chia text thành chunks với overlap
- `load_and_split_pdf(file_path)`: Wrapper function kết hợp cả 2

**Kỹ thuật**:
- Sử dụng `RecursiveCharacterTextSplitter` để giữ context
- Chunk overlap để không mất thông tin ở ranh giới
- Xử lý lỗi khi PDF không đọc được

### Module: `embedding.py`

**Chức năng**: Tạo embeddings cho text chunks

**Key Classes/Functions**:
- `OpenRouterEmbeddings`: Custom embeddings class cho OpenRouter API
- `create_chroma_vectorstore()`: Tạo và lưu vector store
- `get_chroma_vectorstore()`: Load vector store đã có

**Kỹ thuật**:
- Custom `Embeddings` class implement LangChain interface
- Sử dụng OpenAI client với OpenRouter base URL
- Error handling chi tiết với troubleshooting guide
- Auto-recovery khi dimension mismatch (xóa collection cũ)

### Module: `retriever.py`

**Chức năng**: Load retriever từ vector store

**Key Functions**:
- `load_retriever()`: Tạo retriever với configurable search parameters

**Kỹ thuật**:
- Sử dụng `as_retriever()` với `search_kwargs`
- Default `k=4` documents được retrieve
- Tự động tạo thư mục nếu chưa tồn tại

### Module: `llm_response.py`

**Chức năng**: Tạo câu trả lời từ LLM dựa trên retrieved context

**Key Functions**:
- `get_llm_response()`: Main function tạo response
- `format_documents()`: Format documents thành string cho prompt

**Kỹ thuật**:
- LangChain chain: `retriever → format → prompt → model → parser`
- Custom RAG prompt để đảm bảo chỉ trả lời dựa trên context
- Comprehensive error handling cho API errors
- Validate retriever trước khi gọi LLM

### Module: `config.py`

**Chức năng**: Centralized configuration

**Key Features**:
- Load environment variables từ `.env`
- Path resolution cho cross-platform compatibility
- Debug mode để kiểm tra API key loading

## 🧪 Testing

### Chạy tests

```bash
# Tất cả tests
pytest

# Với coverage report
pytest --cov=modules --cov-report=html

# Một test file cụ thể
pytest tests/test_embedding.py

# Một test cụ thể
pytest tests/test_config.py::TestConfig::test_data_dir_exists
```

### Test Structure

- **Unit tests** cho mỗi module
- **Fixtures** trong `conftest.py` để tái sử dụng
- **Skip tests** tự động nếu không có API key
- **Temporary directories** để tránh ảnh hưởng dữ liệu thực

Xem chi tiết tại [tests/README.md](tests/README.md)

## 🐛 Troubleshooting

### Lỗi: "OPENROUTER_API_KEY không được tìm thấy"

**Giải pháp**:
1. Kiểm tra file `.env` có trong thư mục gốc project
2. Format đúng: `OPENROUTER_API_KEY=your_key_here` (không có dấu cách quanh `=`)
3. Restart Streamlit app sau khi sửa `.env`

### Lỗi: "Model không hợp lệ" hoặc "401/403"

**Giải pháp**:
1. Vào [OpenRouter API Keys](https://openrouter.ai/dashboard/settings/api-keys)
2. Chọn API key → Edit
3. Bật "Allow Free Models" và "Allow Paid Models"
4. Chọn providers phù hợp:
   - Qwen cho `qwen/qwen3-embedding-0.6b`
   - OpenAI cho `text-embedding-ada-002`
5. Save changes và thử lại

### Lỗi: "Không tìm thấy nội dung liên quan"

**Nguyên nhân có thể**:
- Câu hỏi không khớp với nội dung PDF
- PDF chưa được xử lý đúng (thử upload lại)
- Vector store bị lỗi (xóa `data/vector_store/` và upload lại)

**Giải pháp**:
- Thử câu hỏi khác, cụ thể hơn
- Upload lại PDF
- Xóa vector store và tạo lại

### Lỗi: "LLM trả về response rỗng"

**Nguyên nhân**:
- Model không được phép truy cập
- API key chưa có quyền cho model đó
- Model không khả dụng

**Giải pháp**:
- Kiểm tra API key settings (xem trên)
- Thử model khác (ví dụ: `qwen/qwen-2.5-32b-instruct`)
- Kiểm tra model có khả dụng tại [OpenRouter Models](https://openrouter.ai/models)

### PDF không đọc được

**Nguyên nhân**:
- PDF bị mã hóa/password protected
- PDF là scanned images (cần OCR)
- PDF corrupted

**Giải pháp**:
- Unlock PDF nếu có password
- Sử dụng PDF có text layer (không phải scanned)
- Thử PDF khác

## 🎓 Best Practices

### 1. Chunk Size và Overlap

- **Chunk size**: 500-1000 ký tự tùy loại tài liệu
  - Tài liệu kỹ thuật: 500-700
  - Tài liệu văn bản: 800-1000
- **Overlap**: 10-20% của chunk size để giữ context

### 2. Model Selection

- **Embedding**: `qwen/qwen3-embedding-0.6b` (free, tốt cho tiếng Việt)
- **LLM**: 
  - Free tier: `qwen/qwen-2.5-32b-instruct`
  - Paid: `qwen/qwen-2.5-72b-instruct` hoặc `openai/gpt-4o`

### 3. Retrieval Parameters

- **k (number of documents)**: 3-5 documents thường đủ
- Tăng k nếu câu hỏi cần nhiều context
- Giảm k nếu muốn response nhanh hơn

### 4. Prompt Engineering

- Prompt trong `prompts/rag_prompt_universal.txt` được thiết kế để:
  - Chỉ trả lời dựa trên context
  - Không bịa thông tin
  - Trả lời tự nhiên, không nhắc đến "theo PDF"

## 🚧 Roadmap / Future Improvements

### Phase 1: Core Features ✅
- [x] PDF upload và processing
- [x] Vector store với Chroma
- [x] RAG pipeline với OpenRouter
- [x] Streamlit UI
- [x] Error handling

### Phase 2: Enhancements (Planned)
- [ ] Support multiple PDFs trong một session
- [ ] PDF preview trong UI
- [ ] Export chat history
- [ ] Advanced search với filters
- [ ] Support cho các file types khác (DOCX, TXT)

### Phase 3: Advanced Features (Future)
- [ ] Multi-language support
- [ ] Citation/source tracking
- [ ] Fine-tuning embedding model
- [ ] Batch processing
- [ ] API endpoint cho integration

## 📄 License

MIT License - Xem [LICENSE](LICENSE) file để biết chi tiết.

## 👏 Acknowledgments

- [LangChain](https://www.langchain.com/) - RAG framework
- [OpenRouter](https://openrouter.ai/) - Unified LLM API
- [Chroma](https://www.trychroma.com/) - Vector database
- [Streamlit](https://streamlit.io/) - Web framework
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF processing

## 📧 Contact

Nếu có câu hỏi hoặc góp ý, vui lòng tạo issue hoặc pull request.

---

**Made with ❤️ using Python, LangChain, OpenRouter, and Streamlit**

