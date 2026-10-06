# Filings Q&A web app (Day 9): FastAPI on :8000 (internal) + Streamlit UI on :7860.
# Hugging Face Spaces (Docker SDK) serves port 7860 and runs containers as user 1000.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/user/.cache/huggingface

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt requirements-index.txt requirements-app.txt ./
RUN pip install -r requirements-app.txt

USER user
# Bake the embedding model into the image so the first question does not download it.
COPY --chown=user config.yaml ./
COPY --chown=user src/__init__.py src/config.py ./src/
RUN python -c "from sentence_transformers import SentenceTransformer; \
from src.config import load_config; \
SentenceTransformer(load_config()['retrieval']['embedding_model'])"

COPY --chown=user .streamlit ./.streamlit
COPY --chown=user src ./src
COPY --chown=user app ./app
COPY --chown=user start.sh ./

EXPOSE 7860 8000
# Run through sh: files uploaded to the Space do not keep the executable bit.
CMD ["sh", "start.sh"]
