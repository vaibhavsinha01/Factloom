"""Object storage (MinIO / S3) for raw PDFs."""
from __future__ import annotations

import logging
from pathlib import Path

from backend.config import (
    S3_ACCESS_KEY,
    S3_BUCKET,
    S3_ENABLED,
    S3_ENDPOINT,
    S3_REGION,
    S3_SECRET_KEY,
)

logger = logging.getLogger("factloom.object_store")

_client = None


def _get_client():
    global _client
    if _client is None:
        import boto3
        from botocore.client import Config

        _client = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            region_name=S3_REGION,
            config=Config(signature_version="s3v4"),
        )
    return _client


def ensure_bucket() -> None:
    if not S3_ENABLED:
        return
    try:
        client = _get_client()
        buckets = [b["Name"] for b in client.list_buckets().get("Buckets", [])]
        if S3_BUCKET not in buckets:
            client.create_bucket(Bucket=S3_BUCKET)
            logger.info("created S3 bucket %s", S3_BUCKET)
    except Exception as e:
        logger.warning("ensure_bucket failed: %s", e)


def upload_pdf(local_path: str, key: str | None = None) -> str | None:
    """Upload a PDF to MinIO/S3. Returns s3:// URI or None if disabled/failed."""
    if not S3_ENABLED:
        return None
    path = Path(local_path)
    object_key = key or f"raw/{path.name}"
    try:
        ensure_bucket()
        _get_client().upload_file(str(path), S3_BUCKET, object_key)
        uri = f"s3://{S3_BUCKET}/{object_key}"
        logger.info("uploaded %s -> %s", path.name, uri)
        return uri
    except Exception as e:
        logger.warning("S3 upload failed for %s: %s", path, e)
        return None


def download_pdf(s3_uri: str, dest_path: str) -> str:
    """Download s3://bucket/key to dest_path."""
    assert s3_uri.startswith("s3://")
    _, rest = s3_uri.split("s3://", 1)
    bucket, key = rest.split("/", 1)
    _get_client().download_file(bucket, key, dest_path)
    return dest_path
