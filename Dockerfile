FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY dubai_rag ./dubai_rag
COPY data/corpus ./data/corpus
RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "dubai_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]

