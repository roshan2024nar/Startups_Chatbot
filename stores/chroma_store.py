"""
stores/chroma_store.py

Loads the saved ChromaDB collection and provides:

- semantic_search()
- filtered_search()
- get_companies_by_name()
- get_all_company_names()

"""

import logging
import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_PATH, CHROMA_COLLECTION, EMBED_MODEL, validate_files


# ---------------------------------------------------------------------------
# Initialization 
# ---------------------------------------------------------------------------

validate_files()

model = SentenceTransformer(EMBED_MODEL)
client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_collection(name=CHROMA_COLLECTION)

logger = logging.getLogger("chatbot")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_collection_count() -> int:
    return collection.count()


def encode(query: str) -> list:
    """Encode a query into a vector."""
    return model.encode([query], show_progress_bar=False).tolist()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def semantic_search(query: str, top_k: int = 5, threshold: float = 0.3) -> list[dict]:
    """
    semantic search across all companies.
    Returns results sorted by similarity.
    """
    q_embedding = encode(query) # user query to embedding vector

    results = collection.query(
        query_embeddings=q_embedding,
        n_results=top_k,
        include=["metadatas", "distances", "documents"],
    )

    output = []

    for i, id_ in enumerate(results["ids"][0]):
        sim = round(1 - results["distances"][0][i] / 2, 4)

        if sim >= threshold:
            meta = results["metadatas"][0][i]
            output.append({
                "startup_name": meta.get("original_name", id_),
                "score": sim,
                "metadata": meta,
                "embedding_text": results["documents"][0][i],
            })

    return output


def filtered_search(
    query: str,
    filters: dict,
    top_k: int = 5,
    threshold: float = 0.3,
) -> list[dict]:
    """
    Semantic search with help of  metadata filters.
    Falls back to semantic_search if filtering fails.
    """
    if not filters:
        return semantic_search(query, top_k, threshold)

    # Build Chroma metadata filter conditions from the filters dict
    conditions = []

    for key, value in filters.items():
        if not value:
            continue
        conditions.append({key: {"$eq": value}})

    if not conditions:
        where = None
    elif len(conditions) == 1:
        where = conditions[0]
    else:
        where = {"$and": conditions}

    # Encode query into embedding vector
    q_embedding = encode(query)

    try:
        # Run filtered vector search in Chroma
        results = collection.query(
            query_embeddings=q_embedding,
            n_results=top_k,
            where=where,
            include=["metadatas", "distances", "documents"],
        )
    except Exception:
        # Fallback to pure semantic search if filtered query fails
        return semantic_search(query, top_k, threshold)

    output = []

    # Process results and compute similarity score
    for i, id_ in enumerate(results["ids"][0]):
        sim = round(1 - results["distances"][0][i] / 2, 4)

        # Keep only results above similarity threshold
        if sim >= threshold:
            meta = results["metadatas"][0][i]
            output.append({
                "startup_name": meta.get("original_name", id_),
                "score": sim,
                "metadata": meta,
                "embedding_text": results["documents"][0][i],
            })

    return output


# ---------------------------------------------------------------------------
# Exact Retrieval
# ---------------------------------------------------------------------------

def get_companies_by_name(company_names: list[str]) -> list[dict]:
    """
    Retrieve companies by exact name using metadata filtering.
    """
    results = []

    for name in company_names:
        try:
            res = collection.get(
                where={"startup_name": name},
                limit=1,
                include=["metadatas", "documents"],
            )

            if res["documents"] and res["metadatas"]:
                results.append({
                    "startup_name": name,
                    "score": 1.0,  # placeholder
                    "metadata": res["metadatas"][0],
                    "embedding_text": res["documents"][0],
                })

        except Exception as e:
            logger.error(f"Error fetching company '{name}': {e}")

    return results


def get_all_company_names() -> list[str]:
    """
    Return all unique startup names in the collection.
    """
    try:
        results = collection.get(include=["metadatas"])
        names = {
            meta["startup_name"]
            for meta in results["metadatas"]
            if meta and "startup_name" in meta
        }
        return list(names)

    except Exception as e:
        logger.error(f"Error fetching company names: {e}")
        return []
