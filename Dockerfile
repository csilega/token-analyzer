FROM python:3.12-slim

# Unprivileged runtime user. Created early because it never changes,
# so this layer is always cached.
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin appuser

# Where tokenizer data lives inside the image. Set BEFORE the prefetch so
# downloads land here, and kept at runtime so the app reads from here.
ENV TIKTOKEN_CACHE_DIR=/opt/tokenizers/tiktoken \
    HF_HOME=/opt/tokenizers/hf \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 1. Third-party dependencies, pinned (change rarely)
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

# 2. Our library (changes sometimes)
COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir --no-deps .

# 3. Bake in tokenizer data using the library's own loader
COPY scripts/prefetch_tokenizers.py scripts/
RUN python scripts/prefetch_tokenizers.py

# From here on, the container must never fetch from the Hub
ENV HF_HUB_OFFLINE=1

# 4. Web layer (changes most often)
COPY api/ api/

# Everything above stays root-owned: readable by appuser, not writable.
USER appuser

CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]