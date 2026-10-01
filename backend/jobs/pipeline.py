"""Async job pipeline via Kafka (with sync fallback)."""
from __future__ import annotations

import json
import logging
import threading
from typing import Callable

from backend import store
from backend.config import (
    KAFKA_BOOTSTRAP,
    KAFKA_ENABLED,
    KAFKA_FALLBACK_SYNC,
    KAFKA_TOPIC_UPLOADS,
)

logger = logging.getLogger("factloom.jobs")

_consumer_thread: threading.Thread | None = None
_stop_event = threading.Event()


def enqueue_upload(pdf_path: str, filename: str) -> dict:
    """Create a job row and publish to Kafka. Falls back to sync processing."""
    job_id = store.create_job("process_pdf", {"pdf_path": pdf_path, "filename": filename})
    payload = {"job_id": job_id, "pdf_path": pdf_path, "filename": filename}

    if KAFKA_ENABLED:
        try:
            from kafka import KafkaProducer

            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP.split(","),
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                request_timeout_ms=5000,
            )
            producer.send(KAFKA_TOPIC_UPLOADS, payload)
            producer.flush(timeout=5)
            producer.close()
            logger.info("enqueued job %s to Kafka topic %s", job_id, KAFKA_TOPIC_UPLOADS)
            return {"job_id": job_id, "status": "queued", "mode": "kafka"}
        except Exception as e:
            logger.warning("Kafka enqueue failed: %s", e)
            if not KAFKA_FALLBACK_SYNC:
                store.update_job(job_id, "failed", str(e))
                return {"job_id": job_id, "status": "failed", "error": str(e)}

    result = process_job(payload)
    return {"job_id": job_id, "status": result.get("status", "complete"), "mode": "sync", **result}


def process_job(payload: dict) -> dict:
    from backend import embed as embed_mod
    from backend.compare import compare_new_facts
    from backend.extract import process_pdf

    job_id = payload.get("job_id")
    pdf_path = payload["pdf_path"]
    if job_id:
        store.update_job(job_id, "running")
    try:
        new_facts = process_pdf(pdf_path)
        for fact in new_facts:
            if fact.get("embedding_status") == "success" and fact.get("embedding"):
                try:
                    vec = (
                        json.loads(fact["embedding"])
                        if isinstance(fact["embedding"], str)
                        else fact["embedding"]
                    )
                    if vec:
                        embed_mod.get_faiss_index().add(fact["id"], vec)
                except Exception as e:
                    logger.warning("FAISS add failed for fact %s: %s", fact.get("id"), e)
        compare_new_facts(new_facts)
        if job_id:
            store.update_job(job_id, "complete")
        return {
            "status": "complete",
            "document_id": new_facts[0]["document_id"] if new_facts else None,
            "facts_extracted": len(new_facts),
        }
    except Exception as e:
        logger.exception("job failed: %s", e)
        if job_id:
            store.update_job(job_id, "failed", str(e))
        return {"status": "failed", "error": str(e)}


def start_consumer(handler: Callable[[dict], None] | None = None) -> None:
    global _consumer_thread
    if not KAFKA_ENABLED:
        logger.info("Kafka disabled — consumer not started")
        return
    if _consumer_thread and _consumer_thread.is_alive():
        return

    def _run():
        try:
            from kafka import KafkaConsumer
        except ImportError:
            logger.warning("kafka-python not installed — consumer not started")
            return

        try:
            consumer = KafkaConsumer(
                KAFKA_TOPIC_UPLOADS,
                bootstrap_servers=KAFKA_BOOTSTRAP.split(","),
                value_deserializer=lambda b: json.loads(b.decode("utf-8")),
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                group_id="factloom-workers",
                consumer_timeout_ms=1000,
            )
        except Exception as e:
            logger.warning("Kafka consumer connect failed: %s", e)
            return

        logger.info("Kafka consumer listening on %s", KAFKA_TOPIC_UPLOADS)
        fn = handler or (lambda p: process_job(p))
        while not _stop_event.is_set():
            try:
                for msg in consumer:
                    if _stop_event.is_set():
                        break
                    try:
                        fn(msg.value)
                    except Exception as e:
                        logger.exception("consumer handler error: %s", e)
            except Exception:
                pass
        consumer.close()

    _stop_event.clear()
    _consumer_thread = threading.Thread(target=_run, name="factloom-kafka-consumer", daemon=True)
    _consumer_thread.start()


def stop_consumer():
    _stop_event.set()
