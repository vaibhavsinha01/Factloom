"""
Evaluation harness. Ingests starter documents and runs comparison.

Usage:
    python -m backend.eval
    python -m backend.eval --fresh-db

Set STORE_BACKEND=sqlite for local eval without Docker, or use postgres after
`docker compose up -d`.
"""
import os
import sys
import glob

from backend import store
from backend.extract import process_pdf
from backend.compare import compare_new_facts
from backend import embed as embed_mod

DATASETS = [
    "data/starter-datasets/delhivery/*.pdf",
    "data/starter-datasets/india-macroeconomy/*.pdf",
]


def main():
    if "--fresh-db" in sys.argv:
        if store.using_postgres():
            print("NOTE: --fresh-db only removes local SQLite file; for Postgres truncate manually.")
        db_path = store.DB_PATH
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"removed {db_path}")

    store.init_db()

    pdf_paths = []
    for pattern in DATASETS:
        pdf_paths.extend(sorted(glob.glob(pattern)))

    print(f"processing {len(pdf_paths)} documents...\n")
    for path in pdf_paths:
        print(f"--- {os.path.basename(path)} ---")
        facts = process_pdf(path)
        for fact in facts:
            if fact.get("embedding_status") == "success" and fact.get("embedding"):
                import json
                try:
                    vec = json.loads(fact["embedding"]) if isinstance(fact["embedding"], str) else fact["embedding"]
                    if vec:
                        embed_mod.get_faiss_index().add(fact["id"], vec)
                except Exception:
                    pass
        print(f"  extracted {len(facts)} facts")
        compare_new_facts(facts)

    relations = store.get_all_relations()
    print(f"\n{len(relations)} total relations found:\n")
    by_type = {}
    for r in relations:
        by_type.setdefault(r["relation_type"], []).append(r)

    for rel_type in ("corroborates", "contradicts", "reconcilable", "needs_review"):
        rels = by_type.get(rel_type, [])
        print(f"=== {rel_type} ({len(rels)}) ===")
        for r in rels[:5]:
            fa = store.get_fact(r["fact_a_id"])
            fb = store.get_fact(r["fact_b_id"])
            if not fa or not fb:
                continue
            print(
                f"  [{fa['document_id']} p{fa['page_no']}] {fa['entity']} {fa['metric']}={fa['value']}{fa['unit'] or ''}"
                f"  <->  [{fb['document_id']} p{fb['page_no']}] {fb['entity']} {fb['metric']}={fb['value']}{fb['unit'] or ''}"
            )
            print(f"    reason={r['reason']} confidence={r['confidence']}  {r['explanation']}")
        print()

    print("Note: 'unrelated' facts are not stored as relations by design.")


if __name__ == "__main__":
    main()
