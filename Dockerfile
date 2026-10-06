# Filings Q&A web app (Day 9): FastAPI on :8000 + Streamlit UI on :7860, for any container
# host. (The hosted demo runs on Streamlit Community Cloud instead; see README "Deploy".)
FROM python:3.11-slim

ENV API_URL=http://localhost:8000 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/user/.cache/huggingface

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY app/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt "uvicorn>=0.30"

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
# Run through sh, so the image works even where the executable bit is lost.
CMD ["sh", "start.sh"]
