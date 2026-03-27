FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY adapters/ adapters/
COPY models/ models/
COPY training/ training/
COPY evaluation/ evaluation/
COPY scripts/ scripts/
COPY tests/ tests/

RUN pip install --no-cache-dir -e ".[dev]"

# Cache model weights at build time
RUN python -c "from transformers import DistilBertModel, DistilBertTokenizer; \
    DistilBertModel.from_pretrained('distilbert-base-uncased'); \
    DistilBertTokenizer.from_pretrained('distilbert-base-uncased')"

CMD ["pytest", "tests/", "-v", "--tb=short"]
