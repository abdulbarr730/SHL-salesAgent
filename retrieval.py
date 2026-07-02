import os
import json
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

# Constants
CATALOG_FILE = "catalog.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Initialize global state for the RAG engine
encoder = None
index = faiss.IndexFlatL2(EMBEDDING_DIM)
catalog_data: List[Dict[str, Any]] = []

def initialize_rag_engine():
    """Loads the embedding model and builds the FAISS index on startup."""
    global encoder, catalog_data
    
    try:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}...")
        encoder = SentenceTransformer(EMBEDDING_MODEL)
        
        if not os.path.exists(CATALOG_FILE):
            logger.error(f"CRITICAL: {CATALOG_FILE} not found. Run the data ingestion script first.")
            return

        with open(CATALOG_FILE, 'r', encoding='utf-8') as f:
            catalog_data = json.load(f)
        
        if not catalog_data:
            logger.warning(f"{CATALOG_FILE} is empty. RAG engine will return no results.")
            return

        logger.info(f"Building FAISS index for {len(catalog_data)} catalog items...")
        
        # Format text for semantic embedding
        catalog_texts = [f"{item.get('name', '')} - {item.get('description', '')}" for item in catalog_data]
        embeddings = encoder.encode(catalog_texts)
        index.add(np.array(embeddings))
        
        logger.info("FAISS index built successfully.")
        
    except Exception as e:
        logger.critical(f"Failed to initialize RAG engine: {e}")

# Run initialization when the module loads
initialize_rag_engine()

def search_catalog(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Embeds the user query and retrieves the top_k most relevant SHL assessments.
    """
    if not query or index.ntotal == 0 or encoder is None:
        logger.warning("Search aborted: Missing query, empty index, or uninitialized encoder.")
        return []
        
    try:
        # Generate embedding for the user's query
        query_vector = encoder.encode([query])
        
        # Search the FAISS index
        distances, indices = index.search(np.array(query_vector), top_k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(catalog_data):
                # Strip the description before sending to the LLM to save tokens and speed up inference
                item = catalog_data[idx].copy()
                item.pop("description", None)
                results.append(item)
                
        return results
        
    except Exception as e:
        logger.error(f"Error during semantic search: {e}")
        return []