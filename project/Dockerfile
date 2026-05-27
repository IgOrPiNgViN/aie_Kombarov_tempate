FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src/ /app/src/
COPY configs/ /app/configs/
COPY data/ /app/data/
COPY artifacts/ /app/artifacts/

ENV PYTHONPATH=/app/src \
    AIE_RAG_ARTIFACTS_DIR=artifacts \
    AIE_RAG_KB_PATH=data/knowledge_base.json

EXPOSE 8000

CMD ["uvicorn", "aie_rag.service.app:app", "--host", "0.0.0.0", "--port", "8000"]

