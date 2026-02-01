import os
import json
import logging
import feedparser
import requests
import hashlib
from typing import List, Dict, Any
from urllib.parse import urlparse
from pydantic import ValidationError

from src.corpus.models import DocumentMeta, Chunk, Span
from src.ingest.db import MetaDB
from src.tools.arxiv_paper import _extract_arxiv_id
from src.tools.pdf import extract_text_from_pdf_bytes # You might need to enhance this

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.getcwd(), "data")
SEEDS_DIR = os.path.join(DATA_DIR, "seeds")
CORPUS_DIR = os.path.join(DATA_DIR, "corpus")
RAW_DIR = os.path.join(CORPUS_DIR, "raw")
PARSED_DIR = os.path.join(CORPUS_DIR, "parsed")
CHUNKS_DIR = os.path.join(CORPUS_DIR, "chunks")
METADATA_DB = os.path.join(DATA_DIR, "metadata", "meta.sqlite")

def _download_pdf(url: str, save_path: str) -> bool:
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        logger.error(f"Failed to download PDF {url}: {e}")
        return False

def ingest_from_seeds(seed_path: str = None, force: bool = False):
    if seed_path is None:
        seed_path = os.path.join(SEEDS_DIR, "seed.jsonl")
    
    logger.info(f"Starting ingestion from {seed_path}")
    
    db = MetaDB(METADATA_DB)
    
    # Ensure directories
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PARSED_DIR, exist_ok=True)
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    
    with open(seed_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                item = json.loads(line)
                url = item.get("url")
                if not url: continue
                
                # Check if Arxiv
                arxiv_id = _extract_arxiv_id(url)
                if arxiv_id:
                    process_arxiv(db, arxiv_id, url, force)
                else:
                    logger.warning(f"Unsupported URL type (not arxiv): {url}")
                    
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON line: {line}")

def process_arxiv(db: MetaDB, arxiv_id: str, original_url: str, force: bool):
    doc_id = DocumentMeta.generate_id(f"arxiv:{arxiv_id}")
    
    # Check if exists
    existing_doc = db.get_document(doc_id)
    if existing_doc and existing_doc.status == "processed" and not force:
        logger.info(f"Skipping {arxiv_id}, already processed.")
        return

    logger.info(f"Processing Arxiv ID: {arxiv_id}")
    
    # 1. Get Metadata
    try:
        # Re-use logic from src.tools.arxiv_paper (simplified here or import)
        # Using feedparser directly to get structured data
        api_url = f"http://export.arxiv.org/api/query?search_query=id:{arxiv_id}&start=0&max_results=1"
        feed = feedparser.parse(api_url)
        if not feed.entries:
            logger.error(f"Arxiv ID {arxiv_id} not found.")
            return
        
        entry = feed.entries[0]
        title = entry.title.replace('\n', ' ').strip()
        summary = entry.summary.replace('\n', ' ').strip()
        published = entry.published
        authors = ", ".join([a.name for a in entry.authors])
        
        pdf_url = None
        for link in entry.links:
            if link.type == 'application/pdf':
                pdf_url = link.href
        
        if not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        # Create DocMeta
        doc = DocumentMeta(
            doc_id=doc_id,
            source_id=f"arxiv:{arxiv_id}",
            source_uri=original_url,
            title=title,
            authors=authors,
            published_date=published,
            summary=summary,
            status="downloading"
        )
        db.upsert_document(doc)
        
        # 2. Download PDF
        pdf_filename = f"{arxiv_id}.pdf"
        pdf_path = os.path.join(RAW_DIR, pdf_filename)
        
        if not os.path.exists(pdf_path) or force:
            if not _download_pdf(pdf_url, pdf_path):
                doc.status = "failed"
                doc.error = "PDF Download failed"
                db.upsert_document(doc)
                return
        
        # 3. Parse & Chunk
        doc.status = "parsing"
        db.upsert_document(doc)
        
        # Using pypdf to extract text (simple version for MVP)
        # In a real system, use src.tools.pdf logic or better parser
        from pypdf import PdfReader
        
        chunks = []
        try:
            reader = PdfReader(pdf_path)
            full_text = ""
            
            chunk_size = 1000 # chars, rough approx
            overlap = 100
            
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text: continue
                
                # Simple sliding window chunking
                # Ideally split by paragraphs, but MVP: fixed char window
                start = 0
                while start < len(text):
                    end = min(start + chunk_size, len(text))
                    chunk_text = text[start:end]
                    
                    chunk = Chunk(
                        chunk_id=Chunk.generate_id(doc_id, chunk_text),
                        doc_id=doc_id,
                        source_id=doc.source_id,
                        source_uri=doc.source_uri,
                        title=doc.title,
                        section_path=[], # PDF simple extraction usually loses sections
                        span=Span(page_start=page_idx+1, page_end=page_idx+1),
                        text=chunk_text,
                        text_hash=hashlib.md5(chunk_text.encode()).hexdigest(),
                        len_chars=len(chunk_text)
                    )
                    chunks.append(chunk)
                    start += (chunk_size - overlap)
            
            # 4. Save
            # Save chunks to DB
            db.clear_chunks_for_doc(doc_id) # Clear old chunks if re-processing
            db.insert_chunks(chunks)
            
            # Save parsed doc meta to jsonl
            with open(os.path.join(PARSED_DIR, "docs.jsonl"), "a", encoding="utf-8") as f:
                f.write(doc.model_dump_json() + "\n")
                
            # Save chunks to jsonl
            with open(os.path.join(CHUNKS_DIR, "chunks.jsonl"), "a", encoding="utf-8") as f:
                for c in chunks:
                    f.write(c.model_dump_json() + "\n")
            
            doc.status = "processed"
            doc.error = None
            db.upsert_document(doc)
            logger.info(f"Successfully processed {arxiv_id}, generated {len(chunks)} chunks.")
            
        except Exception as e:
            logger.error(f"Error parsing PDF {arxiv_id}: {e}")
            doc.status = "failed"
            doc.error = str(e)
            db.upsert_document(doc)

    except Exception as e:
        logger.error(f"Unexpected error processing {arxiv_id}: {e}")
        doc.status = "failed"
        doc.error = str(e)
        db.upsert_document(doc)

if __name__ == "__main__":
    ingest_from_seeds()
