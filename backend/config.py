"""Central configuration loaded from environment / .env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
BRONZE_DIR = DATA_DIR / "lake" / "bronze"
SILVER_DIR = DATA_DIR / "lake" / "silver"
GOLD_DIR = DATA_DIR / "lake" / "gold"
SQLITE_PATH = DATA_DIR / "factloom.db"

# Storage backend: "postgres" (primary) or "sqlite" (local/tests fallback)
STORE_BACKEND = os.environ.get("STORE_BACKEND", "postgres").lower()
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://factloom:factloom@localhost:5432/factloom",
)

# Object storage (MinIO / S3)
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://localhost:9000")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "minioadmin")
S3_BUCKET = os.environ.get("S3_BUCKET", "factloom-raw")
S3_REGION = os.environ.get("S3_REGION", "us-east-1")
S3_ENABLED = os.environ.get("S3_ENABLED", "true").lower() == "true"

# Kafka async pipeline
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_TOPIC_UPLOADS = os.environ.get("KAFKA_TOPIC_UPLOADS", "factloom.uploads")
KAFKA_ENABLED = os.environ.get("KAFKA_ENABLED", "true").lower() == "true"
# When Kafka is down, process inline (sync) instead of failing uploads
KAFKA_FALLBACK_SYNC = os.environ.get("KAFKA_FALLBACK_SYNC", "true").lower() == "true"

# Processing / versioning
MODEL_PROMPT_VERSION = os.environ.get("MODEL_PROMPT_VERSION", "extract-v2+compare-v2")
PIPELINE_VERSION = os.environ.get("PIPELINE_VERSION", "2.0.0")

# Embeddings
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "384"))
EMBEDDING_OFFLINE_FALLBACK = os.environ.get("EMBEDDING_OFFLINE_FALLBACK", "false").lower() == "true"

# Retrieval
RETRIEVAL_VECTOR_K = int(os.environ.get("RETRIEVAL_VECTOR_K", "20"))
RETRIEVAL_LEXICAL_K = int(os.environ.get("RETRIEVAL_LEXICAL_K", "20"))
RETRIEVAL_FINAL_K = int(os.environ.get("RETRIEVAL_FINAL_K", "5"))
RETRIEVAL_SIM_THRESHOLD = float(os.environ.get("RETRIEVAL_SIM_THRESHOLD", "0.45"))


def ensure_data_dirs() -> None:
    for d in (UPLOAD_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR):
        d.mkdir(parents=True, exist_ok=True)
