FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY data/knowledge_base/ ./data/knowledge_base/

RUN useradd --create-home --uid 10001 appuser && \
    mkdir -p /app/data/generated /app/docs/assets && \
    chown -R appuser:appuser /app

USER appuser

RUN python -m finguard.cli run-all

EXPOSE 8050

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8050/api/health', timeout=3)" || exit 1

CMD ["python", "-m", "finguard.cli", "serve", "--host", "0.0.0.0", "--port", "8050"]