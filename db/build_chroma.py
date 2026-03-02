"""
db/build_chroma.py

Builds the ChromaDB vector collection from company_profiles.json.

Usage:
    python -m db.build_chroma
"""

import sys
import re
import traceback
import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_PATH, CHROMA_COLLECTION, EMBED_MODEL
from utils.data_loader import load_profiles, describe_profiles
from utils.text_utils import safe_str, safe_float


BATCH_SIZE = 50  

VALID_ID = re.compile(
    r'^[a-zA-Z0-9][a-zA-Z0-9._-]{0,510}[a-zA-Z0-9]$|^[a-zA-Z0-9]$'
)


# ---------------------------------------------------------------------------
# ID Sanitization
# ---------------------------------------------------------------------------

def sanitize_id(name: str) -> str:
    """Convert startup_name into a valid ChromaDB ID."""
    clean = re.sub(r'[^a-zA-Z0-9._-]', '_', str(name))
    clean = clean.strip('._-')
    clean = re.sub(r'_+', '_', clean)
    clean = clean[:512]
    return clean if clean else None


# ---------------------------------------------------------------------------
# Document Preparation
# ---------------------------------------------------------------------------

def prepare_documents(profiles):
    """Build ids, documents, and metadata from company profiles."""
    ids, documents, metadatas = [], [], []
    skipped = []

    for i, company in enumerate(profiles):
        raw_name = company.get("startup_name", "")
        safe_id = sanitize_id(raw_name)

        if not safe_id:
            skipped.append((i, raw_name))
            continue

        ids.append(safe_id)
        documents.append(company["embedding_text"])
        metadatas.append({
            "industry": safe_str(company.get("industry")),
            "city": safe_str(company.get("city")),
            "num_rounds": int(company.get("num_rounds") or 0),
            "total_funding": safe_float(company.get("total_funding")),
            "has_funding": 1 if company.get("total_funding") else 0,
            "original_name": safe_str(raw_name),
        })

    if skipped:
        print(f"Skipped {len(skipped)} entries due to invalid IDs.")

    # Deduplicate after sanitization
    seen = set()
    deduped_ids, deduped_docs, deduped_metas = [], [], []
    duplicates = 0

    for id_, doc, meta in zip(ids, documents, metadatas):
        if id_ not in seen:
            seen.add(id_)
            deduped_ids.append(id_)
            deduped_docs.append(doc)
            deduped_metas.append(meta)
        else:
            duplicates += 1

    if duplicates:
        print(f"Removed {duplicates} duplicate IDs after sanitization.")

    return deduped_ids, deduped_docs, deduped_metas


# ---------------------------------------------------------------------------
# Embedding + Insert
# ---------------------------------------------------------------------------

def insert_all(ids, documents, metadatas):
    total = len(ids)
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"Computing embeddings for {total} documents...")
    embeddings = model.encode(
        documents,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print(f"Embeddings ready: shape={embeddings.shape}")

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # Rebuild collection cleanly
    try:
        client.delete_collection(CHROMA_COLLECTION)
        print("Deleted existing collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}, # Create a new collection using cosine similarity for vector search
    )

    print(f"Collection '{CHROMA_COLLECTION}' created.")

    print(f"\nInserting documents in batches of {BATCH_SIZE}...")

    for i in range(0, total, BATCH_SIZE):
        batch_ids = ids[i:i + BATCH_SIZE]
        batch_docs = documents[i:i + BATCH_SIZE]
        batch_metas = metadatas[i:i + BATCH_SIZE]
        batch_embs = embeddings[i:i + BATCH_SIZE].tolist()

        try:
            collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
                embeddings=batch_embs,
            )
        except Exception as e:
            print(f"\nBatch failed at index {i}")
            print(f"Error: {e}")
            traceback.print_exc()
            raise

        done = min(i + BATCH_SIZE, total)
        print(f"Inserted {done}/{total}")

    count = collection.count()

    if count != total:
        raise RuntimeError(f"Count mismatch: expected {total}, got {count}")

    print(f"\nAll {count} documents inserted successfully.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Building ChromaDB collection...\n")

    print("Loading profiles...")
    profiles = load_profiles(validate=True)
    describe_profiles(profiles)

    print("\nPreparing documents...")
    ids, documents, metadatas = prepare_documents(profiles)

    print(f"Ready to insert {len(ids)} documents.")
    print(f"Sample ID: '{ids[0]}' (original: '{metadatas[0]['original_name']}')")

    print("\nEmbedding and inserting into ChromaDB...")
    insert_all(ids, documents, metadatas)

    print("\nBuild complete.")
    print(f"Collection: {CHROMA_COLLECTION}")
    print(f"Path: {CHROMA_PATH}")
    print(f"Documents: {len(ids)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nBuild failed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)