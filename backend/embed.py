"""
Local embedding model + FAISS index for semantic retrieval.

Architecture:
    text
     ↓
    SentenceTransformer (local, no API key)
     ↓
    L2-normalized vector
     ↓
    FAISS IndexFlatIP   (inner-product on normalized vectors ≈ cosine similarity)
     ↓
    top-k nearest neighbors

Embedding is entirely decoupled from LLM providers:
  - Works when Groq is unavailable
  - Works when Gemini is unavailable
  - No external API key required

FAISS inner-product note:
  cosine_similarity(a, b) == inner_product(a/|a|, b/|b|)
  Since we L2-normalize before insertion and search, IP search gives cosine similarity.
"""
import os
import json
import logging
import numpy as np

from backend.models import EmbeddingRecord, EmbeddingStatus

logger = logging.getLogger("factloom.embed")

EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EXPECTED_DIM = int(os.environ.get("EMBEDDING_DIM", "384"))  # bge-small-en-v1.5 → 384

# Set EMBEDDING_OFFLINE_FALLBACK=true in sandboxes without HuggingFace internet access.
OFFLINE_FALLBACK = os.environ.get("EMBEDDING_OFFLINE_FALLBACK", "false").lower() == "true"

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("loading local embedding model: %s", EMBEDDING_MODEL_NAME)
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("embedding model loaded (dim=%d)", EXPECTED_DIM)
    return _model


def _offline_fallback_vector(text: str) -> list[float]:
    """Hashed bag-of-words vector — only used when EMBEDDING_OFFLINE_FALLBACK=true.
    NOT a substitute for a real sentence embedding model."""
    import hashlib
    import re
    vec = np.zeros(EXPECTED_DIM)
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    for tok in tokens:
        h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % EXPECTED_DIM
        sign = 1.0 if (h // EXPECTED_DIM) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec.tolist()
    return (vec / norm).tolist()


def _validate_vector(vector: list[float], context: str = "") -> str | None:
    """Return an error string if vector is invalid, else None."""
    if not vector:
        return f"empty vector {context}"
    if len(vector) != EXPECTED_DIM:
        return f"wrong dimensions: got {len(vector)}, expected {EXPECTED_DIM} {context}"
    arr = np.array(vector, dtype=float)
    if not np.all(np.isfinite(arr)):
        return f"vector contains NaN or Inf values {context}"
    if np.all(arr == 0):
        return f"all-zero vector {context}"
    return None


def embed_text(text: str, fact_id: int | None = None) -> EmbeddingRecord:
    """Embed a single string, always returning an EmbeddingRecord (never a bare list).

    Steps:
    1. Validate input text is non-empty
    2. Generate embedding from local model (no API key needed)
    3. Validate vector: dimensions, finite values, non-zero
    4. Return structured EmbeddingRecord with status
    """
    fid = fact_id or 0

    # 1. Validate input
    if not text or not text.strip():
        return EmbeddingRecord(fact_id=fid, status=EmbeddingStatus.FAILED, error="empty input text")

    # 2. Generate embedding
    if OFFLINE_FALLBACK:
        vector = _offline_fallback_vector(text)
        model_name = f"{EMBEDDING_MODEL_NAME}(offline-fallback)"
    else:
        try:
            model = _get_model()
            raw = model.encode(text, normalize_embeddings=True)
            vector = raw.tolist()
            model_name = EMBEDDING_MODEL_NAME
        except Exception as e:
            logger.warning("embedding model error for fact_id=%s: %s", fid, e)
            return EmbeddingRecord(fact_id=fid, status=EmbeddingStatus.FAILED, error=str(e))

    # 3. Validate vector
    err = _validate_vector(vector, f"(fact_id={fid})")
    if err:
        logger.warning("invalid embedding: %s", err)
        return EmbeddingRecord(fact_id=fid, status=EmbeddingStatus.FAILED, error=err)

    return EmbeddingRecord(
        fact_id=fid, status=EmbeddingStatus.SUCCESS,
        vector=vector, model=model_name, dim=len(vector),
    )


# ---------- FAISS Index ----------

class FaissIndex:
    """FAISS inner-product index over L2-normalized embeddings.

    cosine_similarity(a, b) ≈ inner_product(a, b) when both are L2-normalized.
    We use IndexFlatIP for exact nearest-neighbor search.
    """

    def __init__(self, dim: int = EXPECTED_DIM):
        self._dim = dim
        self._index = None     # lazy-init so faiss import error doesn't break startup
        self._id_map: list[int] = []   # position → fact_id mapping

    def _ensure_index(self):
        if self._index is None:
            try:
                import faiss
                self._index = faiss.IndexFlatIP(self._dim)
                logger.info("FAISS IndexFlatIP initialized (dim=%d)", self._dim)
            except ImportError:
                logger.error("faiss-cpu not installed — install it with: pip install faiss-cpu")
                raise

    def add(self, fact_id: int, vector: list[float]):
        """L2-normalize and insert a vector. Rejects invalid vectors."""
        if not vector:
            logger.warning("FAISS: empty vector for fact_id=%s", fact_id)
            return
        if len(vector) != self._dim:
            logger.warning("FAISS: dim mismatch for fact_id=%s: got %d expected %d", fact_id, len(vector), self._dim)
            return
        arr = np.array(vector, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            logger.warning("FAISS: NaN/Inf in vector for fact_id=%s", fact_id)
            return
        if np.all(arr == 0):
            logger.warning("FAISS: all-zero vector for fact_id=%s", fact_id)
            return

        self._ensure_index()
        arr = arr.reshape(1, -1)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        self._index.add(arr)
        self._id_map.append(fact_id)
        logger.debug("FAISS: added fact_id=%d (total=%d)", fact_id, len(self._id_map))

    def search(self, vector: list[float], k: int = 5) -> list[tuple[int, float]]:
        """Return up to k (fact_id, cosine_similarity) pairs.
        Filters: similarity must be finite and in [0, 1] range."""
        if self._index is None or self._index.ntotal == 0:
            return []

        if not vector or len(vector) != self._dim:
            logger.warning("FAISS search: dim mismatch (got %d, expected %d)", len(vector) if vector else 0, self._dim)
            return []

        arr = np.array(vector, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            logger.warning("FAISS search: NaN/Inf in query vector")
            return []

        arr = arr.reshape(1, -1)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm

        actual_k = min(k, self._index.ntotal)
        scores, indices = self._index.search(arr, actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue
            sim = float(score)
            if not np.isfinite(sim):
                continue
            # Clamp to [0, 1] — IP on normalized vectors should already be in this range
            sim = max(0.0, min(1.0, sim))
            results.append((self._id_map[idx], sim))
        return results

    def rebuild_from_facts(self, facts: list[dict]):
        """Bootstrap the index from stored fact dicts (called at startup or after restart)."""
        self._index = None
        self._id_map = []
        self._ensure_index()
        loaded = 0
        for f in facts:
            raw = f.get("embedding")
            vec = _safe_load_vector(raw)
            if vec is None:
                continue
            fid = f.get("id")
            if fid is None:
                continue
            self.add(fid, vec)
            loaded += 1
        logger.info("FAISS: rebuilt index from %d/%d facts", loaded, len(facts))

    @property
    def size(self) -> int:
        return len(self._id_map)


# Module-level singleton — shared across the process
_FAISS_INDEX = FaissIndex()


def get_faiss_index() -> FaissIndex:
    return _FAISS_INDEX


def faiss_search(vector: list[float], k: int = 5, threshold: float = 0.55) -> list[tuple[int, float]]:
    """Search the module-level FAISS index. Returns (fact_id, similarity) pairs above threshold."""
    raw_results = _FAISS_INDEX.search(vector, k)
    return [(fid, sim) for fid, sim in raw_results if sim >= threshold]


# ---------- Similarity helpers (kept for backward-compat and explicit pair comparisons) ----------

def _safe_load_vector(raw) -> list[float] | None:
    """Defensively parse a stored embedding: handles None, empty string, malformed JSON."""
    if raw is None:
        return None
    if isinstance(raw, list):
        vec = raw
    elif isinstance(raw, str):
        if not raw.strip():
            return None
        try:
            vec = json.loads(raw)
        except json.JSONDecodeError:
            return None
    else:
        return None

    if not isinstance(vec, list) or len(vec) == 0:
        return None
    if not all(isinstance(x, (int, float)) for x in vec):
        return None
    if all(x == 0 for x in vec):
        return None
    return vec


def cosine_sim(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a, dtype=float), np.array(b, dtype=float)
    if a_arr.size == 0 or b_arr.size == 0 or a_arr.shape != b_arr.shape:
        return 0.0
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def top_similar(new_fact: dict, candidate_facts: list[dict], k: int = 5, threshold: float = 0.50) -> list[dict]:
    """Return up to k candidate facts most similar to new_fact above threshold.
    Directly evaluates candidate_facts to avoid same-document index crowding."""
    new_emb = _safe_load_vector(new_fact.get("embedding"))
    if new_emb is None:
        logger.info("skipping retrieval for fact_id=%s: no valid embedding", new_fact.get("id"))
        return []

    new_id = new_fact.get("id")
    scored = []
    for cand in candidate_facts:
        cid = cand.get("id")
        if cid is None or cid == new_id:
            continue
        cand_emb = _safe_load_vector(cand.get("embedding"))
        if cand_emb is None:
            continue
        sim = cosine_sim(new_emb, cand_emb)
        # No artificial similarity boosting — metric preference belongs in retrieve.rerank
        if sim >= threshold:
            scored.append((sim, cand))

    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:k]]
