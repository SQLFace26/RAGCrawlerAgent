import asyncio
import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# Qdrant and Ollama configuration
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "tonygray_palottery_ai_docs"
OLLAMA_URL = "http://localhost:11434"  # Default Ollama server URL (for local models)
EMBEDDING_MODEL = "mistral"  # Use Mistral 7B for embeddings via Ollama
LLM_MODEL = "mistral"       # Use Mistral 7B for LLM responses via Ollama

# Initialize asynchronous Qdrant client
qdrant_client = AsyncQdrantClient(url=QDRANT_URL)

async def embed_text(text: str) -> list[float]:
    """Generate an embedding vector for the given text using Ollama's embedding API."""
    payload = {"model": EMBEDDING_MODEL, "input": text}
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{OLLAMA_URL}/api/embed", json=payload)
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings")
        if not embeddings:
            raise ValueError("No embedding returned from Ollama for the given text.")
        # Return the first embedding vector (for single text input)
        return embeddings[0]

async def search_documents(query: str, top_k: int = 5) -> list[dict]:
    """Search for relevant document chunks in the Qdrant collection for the given query."""
    # Embed the query text and perform similarity search in Qdrant
    query_vector = await embed_text(query)
    results = await qdrant_client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True
    )
    # Compile search results into a list of documents with metadata
    documents = []
    for point in results:
        payload = point.payload or {}
        documents.append({
            "title": payload.get("title"),
            "url": payload.get("url"),
            "chunk_number": payload.get("chunk_number"),
            "content": payload.get("content") or payload.get("text") or ""
        })
    return documents

async def list_documents() -> list[dict]:
    """List all unique documents (titles and URLs) stored in the Qdrant collection."""
    titles = {}
    next_offset = None
    # Iterate over all points in the collection using pagination (scroll)
    while True:
        points, next_offset = await qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION,
            offset=next_offset,
            limit=100,
            # Only retrieve minimal fields for efficiency
            with_payload=["title", "url", "chunk_number"]
        )
        for point in points:
            payload = point.payload or {}
            title = payload.get("title")
            url = payload.get("url")
            if title and title not in titles:
                titles[title] = url
        if next_offset is None:
            break
    # Convert gathered titles to list of dicts
    return [{"title": t, "url": titles[t]} for t in titles]

async def get_document(title: str) -> dict:
    """Retrieve and reconstruct the full document (all chunks) by its title."""
    # Filter to get all chunks for the given title
    title_filter = Filter(must=[FieldCondition(key="title", match=MatchValue(value=title))])
    all_points = []
    next_offset = None
    # Scroll through all points that match the title filter
    while True:
        points, next_offset = await qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION,
            offset=next_offset,
            limit=100,
            filter=title_filter,
            with_payload=True
        )
        all_points.extend(points)
        if next_offset is None:
            break
    # Sort chunks by their chunk_number to maintain correct order
    all_points.sort(key=lambda p: (p.payload.get("chunk_number", 0) if p.payload else 0))
    # Combine all chunk contents into one document text
    content_parts = []
    for point in all_points:
        payload = point.payload or {}
        text = payload.get("content") or payload.get("text") or ""
        if text:
            content_parts.append(text.strip())
    full_content = "\n\n".join(content_parts)
    # Use the URL from the first chunk (if available) for reference
    doc_url = all_points[0].payload.get("url") if all_points else None
    return {"title": title, "url": doc_url, "content": full_content}

async def answer_question(question: str) -> str:
    """Answer a user question using relevant documents from Qdrant as context (RAG)."""
    # Retrieve top relevant document chunks for the question
    docs = await search_documents(question, top_k=5)
    # Build context from the retrieved documents
    context_snippets = []
    for doc in docs:
        if not doc["content"]:
            continue
        title = doc.get("title") or "Document"
        content = doc["content"]
        context_snippets.append(f"Title: {title}\nContent: {content}")
    context_text = "\n\n".join(context_snippets)
    # Formulate the prompt for the Mistral model with the context
    prompt = (
        "Use the following document excerpts to answer the question.\n"
        f"{context_text}\n\n"
        f"Question: {question}\nAnswer:"
    )
    # Generate the answer using Ollama's local Mistral model
    payload = {"model": LLM_MODEL, "prompt": prompt, "stream": False}
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
        resp.raise_for_status()
        result = resp.json()
    # Return the answer text from the model's response
    return result.get("response", "")
