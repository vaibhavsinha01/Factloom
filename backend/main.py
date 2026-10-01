import logging
import os
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend import embed as embed_mod
from backend import llm, store
from backend.compare import compare_new_facts
from backend.config import UPLOAD_DIR, ensure_data_dirs
from backend.extract import process_pdf
from backend.jobs import enqueue_upload, start_consumer
from backend.rag import answer_question

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("factloom.main")

ensure_data_dirs()
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="FactLoom")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

store.init_db()


@app.on_event("startup")
def startup():
    store.init_db()
    facts = store.get_all_facts(active_only=True)
    if facts:
        embed_mod.get_faiss_index().rebuild_from_facts(facts)
        logger.info("startup: FAISS index rebuilt with %d facts", embed_mod.get_faiss_index().size)
    try:
        start_consumer()
    except Exception as e:
        logger.warning("Kafka consumer not started: %s", e)


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), sync: bool = False):
    """Upload a PDF. By default enqueues async processing (Kafka with sync fallback).

    Pass ?sync=true to process inline (useful for demos / eval).
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    dest = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    if sync:
        llm.reset_metrics()
        new_facts = process_pdf(dest)
        for fact in new_facts:
            if fact.get("embedding_status") == "success" and fact.get("embedding"):
                import json
                try:
                    vec = json.loads(fact["embedding"]) if isinstance(fact["embedding"], str) else fact["embedding"]
                    if vec:
                        embed_mod.get_faiss_index().add(fact["id"], vec)
                except Exception as e:
                    logger.warning("could not add fact %s to FAISS: %s", fact.get("id"), e)
        compare_new_facts(new_facts)
        metrics = llm.get_metrics()
        return {
            "document_id": new_facts[0]["document_id"] if new_facts else None,
            "facts_extracted": len(new_facts),
            "metrics": metrics,
            "mode": "sync",
        }

    result = enqueue_upload(dest, file.filename)
    return result


@app.get("/jobs/{job_id}")
def get_job(job_id: int):
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.post("/reprocess")
def reprocess_all():
    """Reprocess all uploaded PDFs (sync). Versioning preserves historical facts."""
    llm.reset_metrics()
    results = {}
    pdf_files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf")]
    for fname in sorted(pdf_files):
        path = os.path.join(UPLOAD_DIR, fname)
        new_facts = process_pdf(path)
        for fact in new_facts:
            if fact.get("embedding_status") == "success" and fact.get("embedding"):
                import json
                try:
                    vec = json.loads(fact["embedding"]) if isinstance(fact["embedding"], str) else fact["embedding"]
                    if vec:
                        embed_mod.get_faiss_index().add(fact["id"], vec)
                except Exception as e:
                    logger.warning("could not add fact %s to FAISS: %s", fact.get("id"), e)
        compare_new_facts(new_facts)
        results[fname] = len(new_facts)

    return {
        "reprocessed": results,
        "total_facts": len(store.get_all_facts()),
        "total_relations": len(store.get_all_relations()),
        "metrics": llm.get_metrics(),
    }


@app.get("/documents")
def list_documents():
    return store.list_documents()


@app.get("/facts")
def get_facts(document_id: str | None = None):
    facts = store.get_facts_by_document(document_id) if document_id else store.get_all_facts()
    for f in facts:
        f.pop("embedding", None)
    return facts


@app.get("/facts/{fact_id}/relations")
def get_fact_relations(fact_id: int):
    fact = store.get_fact(fact_id)
    if not fact:
        raise HTTPException(404, "Fact not found")
    fact.pop("embedding", None)
    relations = store.get_relations_for_fact(fact_id)
    enriched = []
    for r in relations:
        other_id = r["fact_b_id"] if r["fact_a_id"] == fact_id else r["fact_a_id"]
        other_fact = store.get_fact(other_id)
        if other_fact:
            other_fact.pop("embedding", None)
        # Always include both sides clearly for the A→relation→B viewer
        fact_a = fact if r["fact_a_id"] == fact_id else other_fact
        fact_b = other_fact if r["fact_a_id"] == fact_id else fact
        if r["fact_a_id"] != fact_id:
            fact_a = other_fact
            fact_b = fact
        enriched.append({
            **r,
            "other_fact": other_fact,
            "fact_a": fact_a,
            "fact_b": fact_b,
        })
    return enriched


@app.get("/relations")
def get_all_relations():
    return store.get_all_relations()


@app.get("/metrics")
def get_metrics():
    return {
        **llm.get_metrics(),
        "faiss_index_size": embed_mod.get_faiss_index().size,
    }


@app.get("/metrics/history")
def metric_history(entity: str | None = None, metric: str | None = None):
    return store.get_metric_history(entity=entity, metric=metric)


@app.post("/ask")
def ask(payload: dict):
    """Grounded RAG endpoint."""
    question = (payload or {}).get("question", "").strip()
    if not question:
        raise HTTPException(400, "question is required")
    k = int((payload or {}).get("k", 8))
    return answer_question(question, k=k)


FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def root():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
