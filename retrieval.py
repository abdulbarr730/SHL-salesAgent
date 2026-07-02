import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Global cache to keep the model in memory across API requests
_catalog = None
_vectorizer = None
_tfidf_matrix = None

def _init_retriever():
    global _catalog, _vectorizer, _tfidf_matrix
    if _catalog is not None:
        return

    catalog_path = "catalog.json"
    if not os.path.exists(catalog_path):
        _catalog = []
        return

    with open(catalog_path, "r") as f:
        _catalog = json.load(f)

    if not _catalog:
        return

    # Combine text fields to give the vectorizer rich context to search against
    documents = [
        f"{item.get('name', '')} {item.get('description', '')} {item.get('test_type', '')}"
        for item in _catalog
    ]

    _vectorizer = TfidfVectorizer(stop_words='english')
    _tfidf_matrix = _vectorizer.fit_transform(documents)

def search_catalog(query: str, top_k: int = 3):
    """
    Lightweight TF-IDF matcher replacing heavy FAISS/SentenceTransformers.
    Matches the exact function interface required by agent.py while using <15MB RAM.
    """
    _init_retriever()

    if not _catalog or _vectorizer is None:
        return []

    # Vectorize incoming user query and calculate cosine similarity
    query_vec = _vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, _tfidf_matrix).flatten()

    # Sort indices by highest similarity score
    top_indices = similarities.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        if similarities[idx] > 0.0:  # Only include if there is a contextual overlap
            results.append(_catalog[idx])

    # Fallback safety: if no strict keywords match, return default catalog items 
    # so the agent always has catalog structural context to work with.
    if not results:
        results = _catalog[:top_k]

    return results