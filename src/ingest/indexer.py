import os
import json
import logging
import pickle
import numpy as np
import faiss
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from src.corpus.models import Chunk

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.getcwd(), "data")
CHUNKS_FILE = os.path.join(DATA_DIR, "corpus", "chunks", "chunks.jsonl")
VECTOR_INDEX_DIR = os.path.join(DATA_DIR, "indexes", "vector")
FAISS_INDEX_FILE = os.path.join(VECTOR_INDEX_DIR, "faiss.index")
CHUNK_ID_MAP_FILE = os.path.join(VECTOR_INDEX_DIR, "chunk_ids.pkl")

# Embedding Model
MODEL_NAME = "all-MiniLM-L6-v2"

def build_index(force: bool = False):
    os.makedirs(VECTOR_INDEX_DIR, exist_ok=True)
    
    if os.path.exists(FAISS_INDEX_FILE) and not force:
        logger.info("Index already exists. Use force=True to rebuild.")
        return

    logger.info(f"Loading chunks from {CHUNKS_FILE}")
    chunks = []
    chunk_ids = []
    texts = []
    
    if not os.path.exists(CHUNKS_FILE):
        logger.error("Chunks file not found. Run ingestion first.")
        return

    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                chunk = Chunk(**data)
                chunks.append(chunk)
                chunk_ids.append(chunk.chunk_id)
                texts.append(chunk.text)
            except Exception as e:
                logger.warning(f"Skipping invalid line: {e}")

    if not texts:
        logger.warning("No chunks to index.")
        return

    logger.info(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    logger.info(f"Encoding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    # Normalize for cosine similarity (Inner Product in Faiss)
    faiss.normalize_L2(embeddings)
    
    d = embeddings.shape[1]
    logger.info(f"Building FAISS index (dim={d})...")
    
    index = faiss.IndexFlatIP(d) # Inner Product = Cosine Similarity (if normalized)
    index.add(embeddings)
    
    # Save
    logger.info(f"Saving index to {FAISS_INDEX_FILE}")
    faiss.write_index(index, FAISS_INDEX_FILE)
    
    with open(CHUNK_ID_MAP_FILE, 'wb') as f:
        pickle.dump(chunk_ids, f)
        
    logger.info("Indexing complete.")

if __name__ == "__main__":
    build_index(force=True)
