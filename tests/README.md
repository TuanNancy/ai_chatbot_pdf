# Unit Tests

Thư mục này chứa các unit tests cho project AI Chatbot PDF.

## Cấu trúc

- `conftest.py`: Shared fixtures và utilities
- `test_config.py`: Tests cho `modules/config.py`
- `test_pdf_loader.py`: Tests cho `modules/pdf_loader.py`
- `test_embedding.py`: Tests cho `modules/embedding.py`
- `test_retriever.py`: Tests cho `modules/retriever.py`
- `test_llm_response.py`: Tests cho `modules/llm_response.py`

## Chạy tests

### Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Chạy tất cả tests

```bash
pytest
```

### Chạy tests với coverage

```bash
pytest --cov=modules --cov-report=html
```

### Chạy một test file cụ thể

```bash
pytest tests/test_config.py
```

### Chạy một test cụ thể

```bash
pytest tests/test_config.py::TestConfig::test_data_dir_exists
```

### Chạy tests không cần API key (skip các tests cần API)

```bash
pytest -m "not slow"
```

## Lưu ý

- Một số tests yêu cầu `OPENROUTER_API_KEY` trong environment variables
- Các tests này sẽ được skip tự động nếu không có API key
- Tests sử dụng temporary directories để tránh ảnh hưởng đến dữ liệu thực
