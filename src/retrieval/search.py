import os
import json
import logging
import pickle
import numpy as np
import faiss
import sqlite3
from typing import List, Dict, Optional, Any
from sentence_transformers import SentenceTransformer
from src.corpus.models import Chunk, DocumentMeta
from src.ingest.db import MetaDB

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.getcwd(), "data")
VECTOR_INDEX_DIR = os.path.join(DATA_DIR, "indexes", "vector")
FAISS_INDEX_FILE = os.path.join(VECTOR_INDEX_DIR, "faiss.index")
CHUNK_ID_MAP_FILE = os.path.join(VECTOR_INDEX_DIR, "chunk_ids.pkl")
METADATA_DB = os.path.join(DATA_DIR, "metadata", "meta.sqlite")
MODEL_NAME = "all-MiniLM-L6-v2"

class HybridSearcher:
    def __init__(self):
        self.db = MetaDB(METADATA_DB)
        self.vector_index = None
        self.chunk_ids = None
        self.model = None
        self._load_vector_index()

    def _load_vector_index(self):
        if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(CHUNK_ID_MAP_FILE):
            logger.info("Loading vector index...")
            self.vector_index = faiss.read_index(FAISS_INDEX_FILE)
            with open(CHUNK_ID_MAP_FILE, 'rb') as f:
                self.chunk_ids = pickle.load(f)
            self.model = SentenceTransformer(MODEL_NAME)
            logger.info(f"Vector index loaded with {len(self.chunk_ids)} chunks.")
        else:
            logger.warning("Vector index not found. Only BM25 available.")

    def search_bm25(self, query: str, k: int = 50) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(METADATA_DB)
        c = conn.cursor()
        # FTS5 query syntax: match query
        # We should sanitize query a bit or just pass it
        try:
            c.execute('''
                SELECT chunk_id, text, rank 
                FROM chunks_fts 
                WHERE chunks_fts MATCH ? 
                ORDER BY rank 
                LIMIT ?
            ''', (query, k))
            results = [{"chunk_id": row[0], "text": row[1], "score": -row[2]} for row in c.fetchall()] # rank is usually negative or smaller is better? FTS5 rank: smaller is better. So -rank is score.
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            results = []
        conn.close()
        return results

    def search_vector(self, query: str, k: int = 50) -> List[Dict[str, Any]]:
        if not self.vector_index or not self.model:
            return []
        
        embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(embedding)
        
        scores, indices = self.vector_index.search(embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.chunk_ids):
                results.append({
                    "chunk_id": self.chunk_ids[idx],
                    "score": float(score)
                })
        return results

    def _rrf_fusion(self, results_dict: Dict[str, Dict[str, float]], k: int = 60) -> List[Dict[str, Any]]:
        # RRF score = 1 / (k + rank)
        final_scores = {}
        
        for method, results in results_dict.items():
            for rank, item in enumerate(results):
                chunk_id = item["chunk_id"]
                score = 1.0 / (k + rank + 1)
                if chunk_id not in final_scores:
                    final_scores[chunk_id] = 0.0
                final_scores[chunk_id] += score
        
        sorted_chunks = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return [{"chunk_id": chunk_id, "score": score} for chunk_id, score in sorted_chunks]

    def _hydrate_chunks(self, chunk_scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not chunk_scores:
            return []
        
        chunk_ids = [item["chunk_id"] for item in chunk_scores]
        placeholders = ','.join(['?'] * len(chunk_ids))
        
        conn = sqlite3.connect(METADATA_DB)
        c = conn.cursor()
        
        # Get chunk details + doc title/uri
        query = f'''
            SELECT c.chunk_id, c.text, c.doc_id, c.section_path, c.page_start, c.page_end, 
                   d.title, d.source_uri, d.source_id
            FROM chunks c
            JOIN documents d ON c.doc_id = d.doc_id
            WHERE c.chunk_id IN ({placeholders})
        '''
        c.execute(query, chunk_ids)
        rows = c.fetchall()
        conn.close()
        
        chunk_map = {row[0]: {
            "chunk_id": row[0],
            "text": row[1],
            "doc_id": row[2],
            "section_path": row[3],
            "page_start": row[4],
            "page_end": row[5],
            "title": row[6],
            "source_uri": row[7],
            "source_id": row[8]
        } for row in rows}
        
        hydrated = []
        for item in chunk_scores:
            cid = item["chunk_id"]
            if cid in chunk_map:
                data = chunk_map[cid]
                data["score"] = item["score"]
                hydrated.append(data)
                
        return hydrated

    def search(self, query: str, top_k: int = 12, bm25_k: int = 60, vec_k: int = 60) -> List[Dict[str, Any]]:
        bm25_res = self.search_bm25(query, bm25_k)
        vec_res = self.search_vector(query, vec_k)
        
        fused = self._rrf_fusion({"bm25": bm25_res, "vector": vec_res}, k=60)
        top_fused = fused[:top_k]
        
        return self._hydrate_chunks(top_fused)

# Global instance
_searcher = None

def get_searcher():
    global _searcher
    if _searcher is None:
        _searcher = HybridSearcher()
    return _searcher
