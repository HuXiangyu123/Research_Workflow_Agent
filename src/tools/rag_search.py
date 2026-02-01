from langchain_core.tools import tool
from src.retrieval.search import get_searcher

@tool
def rag_search(query: str) -> str:
    """
    Search the local knowledge base (RAG) for relevant document chunks.
    Use this to find specific details, facts, experiments, or related work from ingested papers.
    Returns a list of chunks with source metadata.
    """
    try:
        searcher = get_searcher()
        # Default top_k=8 for context window balance
        results = searcher.search(query, top_k=8)
        
        if not results:
            return "No relevant documents found in the knowledge base."
        
        formatted = []
        for i, res in enumerate(results):
            # Format: [i] Title (Page x): Text...
            source = f"{res['title']} (Page {res.get('page_start','?')})"
            # Add source_uri for reference if needed, but for context we keep it concise
            text = res['text'].replace('\n', ' ')
            formatted.append(f"[{i+1}] Source: {source}\nURL: {res['source_uri']}\nContent: {text}\n")
            
        return "\n---\n".join(formatted)
    except Exception as e:
        return f"Error during search: {str(e)}"
