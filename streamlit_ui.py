from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os
from datetime import datetime
import pytz

import streamlit as st
import json
from dotenv import load_dotenv

# HTTP and Qdrant imports
import httpx
from qdrant_client import AsyncQdrantClient
# NOTE: We must import VectorParams to specify the config when recreating the collection:
from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
    PointStruct,
    VectorParams
)

# Internal imports (your crawler functions, etc.)
from crawler import crawl_parallel, get_urls_from_sitemap

load_dotenv()

# -----------------------------------------------------------------------------
# QDRANT & OLLAMA CONFIGURATION
# -----------------------------------------------------------------------------
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "tonygray_palottery_ai_docs"  # Qdrant collection name
OLLAMA_URL = "http://localhost:11434"  # Default Ollama server URL
EMBEDDING_MODEL = "mistral"  # Mistral 7B for embeddings
LLM_MODEL = "mistral"  # Mistral 7B for LLM responses

# Asynchronous Qdrant client
qdrant_client = AsyncQdrantClient(url=QDRANT_URL)


async def embed_text(text: str) -> list[float]:
    """
    Generate an embedding vector for the given text using Ollama's embedding API.
    """
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
    """
    Search for relevant document chunks in Qdrant for the given query.
    Returns a list of dicts with 'title', 'url', 'chunk_number', and 'content'.
    """
    query_vector = await embed_text(query)
    results = await qdrant_client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True
    )
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


async def answer_question(question: str) -> str:
    """
    RAG-style: retrieve top relevant chunks, build a context prompt, and generate
    an answer using Ollama's local Mistral model.
    """
    docs = await search_documents(question, top_k=5)
    # Build context from retrieved documents
    context_snippets = []
    for doc in docs:
        if not doc["content"]:
            continue
        title = doc.get("title") or "Document"
        content = doc["content"]
        context_snippets.append(f"Title: {title}\nContent: {content}")
    context_text = "\n\n".join(context_snippets)

    prompt = (
        "Use the following document excerpts to answer the question.\n"
        f"{context_text}\n\n"
        f"Question: {question}\nAnswer:"
    )

    payload = {"model": LLM_MODEL, "prompt": prompt, "stream": False}
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
        resp.raise_for_status()
        result = resp.json()

    return result.get("response", "")


async def upsert_chunks_to_qdrant(title: str, url: str, chunks: list[str]):
    """
    Given a list of text chunks, embed each chunk and upsert them into Qdrant.
    """
    points = []
    for i, chunk_content in enumerate(chunks):
        if not chunk_content.strip():
            continue
        vector = await embed_text(chunk_content)
        point = PointStruct(
            id=None,  # Qdrant can auto-assign IDs if set to None
            vector=vector,
            payload={
                "title": title,
                "url": url,
                "chunk_number": i,
                "content": chunk_content.strip()
            }
        )
        points.append(point)

    if points:
        await qdrant_client.upsert(collection_name=QDRANT_COLLECTION, points=points)


async def count_qdrant_docs() -> int:
    """
    Return the total count of vectors stored in the Qdrant collection.
    """
    count_result = await qdrant_client.count(collection_name=QDRANT_COLLECTION)
    return count_result.count


async def list_unique_urls_and_domains() -> tuple[list[str], list[str]]:
    """
    Scroll through all Qdrant points, gather unique URLs and source domains.
    """
    seen_urls = set()
    domains = set()
    next_offset = None

    while True:
        points, next_offset = await qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION,
            offset=next_offset,
            limit=200,
            with_payload=True
        )
        for p in points:
            payload = p.payload or {}
            url = payload.get("url")
            if url:
                seen_urls.add(url)
                # Very naive approach to parse domain:
                domain = url.split("//")[-1].split("/")[0]
                domains.add(domain)
        if next_offset is None:
            break

    return list(seen_urls), list(domains)


# -----------------------------------------------------------------------------
# STREAMLIT APP
# -----------------------------------------------------------------------------
class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""
    role: Literal["user", "model"]
    timestamp: str
    content: str


def format_sitemap_url(url: str) -> str:
    """Format URL to ensure proper sitemap URL structure."""
    url = url.rstrip("/")
    if not url.endswith("sitemap.xml"):
        url = f"{url}/sitemap.xml"
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    return url


async def get_db_stats() -> dict | None:
    """
    Gather statistics and information about the current Qdrant-based database.
    """
    try:
        doc_count = await count_qdrant_docs()
        if doc_count == 0:
            return None

        urls, domains = await list_unique_urls_and_domains()
        # We don't store last_updated in Qdrant, so we'll show current local time
        now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
        local_tz = datetime.now().astimezone().tzinfo
        now_local = now_utc.astimezone(local_tz).strftime("%Y-%m-%d %H:%M:%S %Z")

        return {
            "urls": urls,
            "domains": domains,
            "doc_count": doc_count,
            "last_updated": now_local,
        }
    except Exception as e:
        print(f"Error getting DB stats: {e}")
        return None


def initialize_session_state():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    if "urls_processed" not in st.session_state:
        st.session_state.urls_processed = set()
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "current_progress" not in st.session_state:
        st.session_state.current_progress = 0
    if "total_urls" not in st.session_state:
        st.session_state.total_urls = 0
    if "suggested_questions" not in st.session_state:
        st.session_state.suggested_questions = None


async def initialize_with_existing_data() -> dict | None:
    """Check for existing data in Qdrant and set session state accordingly."""
    stats = await get_db_stats()
    if stats and stats["doc_count"] > 0:
        st.session_state.processing_complete = True
        st.session_state.urls_processed = set(stats["urls"])
        return stats
    return None


async def process_url(url: str):
    """Process a URL by crawling its sitemap or single page and upserting to Qdrant."""
    try:
        progress_container = st.empty()
        with progress_container.container():
            formatted_url = format_sitemap_url(url)
            st.write(f"🔄 Processing {formatted_url}...")

            st.write("📑 Attempting to fetch sitemap...")
            urls = get_urls_from_sitemap(formatted_url)

            if urls:
                st.write(f"📎 Found {len(urls)} URLs in sitemap")
                progress_bar = st.progress(0, text="Processing URLs...")
                st.session_state.total_urls = len(urls)

                # Crawl the sitemap URLs
                status_placeholder = st.empty()
                status_placeholder.text("⏳ Crawling web pages...")
                crawl_results = await crawl_parallel(urls)

                status_placeholder.text("🧮 Computing embeddings & upserting...")
                total_docs = len(crawl_results)
                for idx, doc in enumerate(crawl_results):
                    await upsert_chunks_to_qdrant(
                        doc["title"], doc["url"], doc["chunks"]
                    )
                    progress_bar.progress(
                        int((idx + 1) / total_docs * 100),
                        text="Processing URLs..."
                    )

                progress_bar.progress(100, text="Processing complete!")
                status_placeholder.empty()

            else:
                st.write("❌ No sitemap found or empty sitemap.")
                st.write("🔍 Attempting to process as a single URL...")
                original_url = url.rstrip("/sitemap.xml")
                st.session_state.total_urls = 1

                status_placeholder = st.empty()
                status_placeholder.text("⏳ Crawling webpage...")
                crawl_results = await crawl_parallel([original_url])
                status_placeholder.empty()

                if crawl_results:
                    doc = crawl_results[0]
                    await upsert_chunks_to_qdrant(doc["title"], doc["url"], doc["chunks"])

            # Summary
            try:
                doc_count = await count_qdrant_docs()
                st.success(
                    f"""
                    ✅ Processing complete! 

                    Documents in database: {doc_count}
                    Last processed URL: {url}

                    You can now ask questions about the content.
                    """
                )
            except Exception as e:
                st.error(f"Unable to get document count: {str(e)}")

    except Exception as e:
        st.error(f"Error processing URL: {str(e)}")


def display_history_messages():
    """
    Displays messages from st.session_state.messages in the Streamlit chat UI.
    Each message is a dict with "role" and "content".
    """
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])


async def generate_contextual_questions() -> list[str]:
    """
    Generate example questions based on existing content.
    Here it's a simple static approach; enhance as needed for real context.
    """
    return [
        "What is this website or documentation mainly about?",
        "Tell me the key points from the crawled pages.",
        "How does the site describe its primary services or features?",
        "Can you summarize the latest updates mentioned in the documents?",
    ]


async def main():
    st.set_page_config(
        page_title="Dynamic RAG Chat System (Qdrant + Ollama)",
        page_icon="🤖",
        layout="wide"
    )

    initialize_session_state()
    existing_data = await initialize_with_existing_data()

    st.title("Dynamic RAG Chat System (Qdrant + Ollama)")

    # Show system status info
    if existing_data:
        st.success("💡 System is ready with existing knowledge base!")
        with st.expander("Knowledge Base Information", expanded=True):
            st.markdown(
                f"""
                ### Current Knowledge Base Stats:
                - 📚 Number of documents: {existing_data['doc_count']}
                - 🌐 Number of sources: {len(existing_data['domains'])}
                - 🕒 Last updated: {existing_data['last_updated']}

                ### Sources include:
                {', '.join(existing_data['domains'])}

                ### You can ask questions about:
                - Any content from the processed websites
                - Specific information from any of the loaded pages
                - Technical details, documentation, or other content
                ### Loaded URLs:
                """
            )
            for url in existing_data["urls"]:
                st.write(f"- {url}")
    else:
        st.info("👋 Welcome! Start by adding a website to build your knowledge base.")

    # Create two main columns
    input_col, chat_col = st.columns([1, 2])

    # -----------------------------------
    # LEFT COLUMN: Add/Manage Content
    # -----------------------------------
    with input_col:
        st.subheader("Add Content to RAG System")
        st.write("Enter a website URL to process. The system will:")
        st.write("1. Try to find and process the sitemap (appending '/sitemap.xml').")
        st.write("2. If no sitemap found, process the URL as a single page.")

        url_input = st.text_input(
            "Website URL",
            key="url_input",
            placeholder="example.com or https://example.com",
        )

        if url_input:
            formatted_preview = format_sitemap_url(url_input)
            st.caption(f"Will try: {formatted_preview}")

        col1, col2 = st.columns(2)
        with col1:
            process_button = st.button(
                "Process URL",
                disabled=st.session_state.is_processing,
                type="primary"
            )
        with col2:
            if st.button(
                    "Clear Database",
                    disabled=st.session_state.is_processing,
                    type="secondary",
            ):
                try:
                    # Wipe entire Qdrant collection by dropping it
                    asyncio.run(qdrant_client.delete_collection(QDRANT_COLLECTION))
                    # Recreate it so it's ready for new upserts
                    vectors_config = VectorParams(size=384, distance="Cosine")
                    asyncio.run(
                        qdrant_client.recreate_collection(
                            collection_name=QDRANT_COLLECTION,
                            vectors_config=vectors_config
                        )
                    )
                    # Reset session state
                    st.session_state.processing_complete = False
                    st.session_state.urls_processed = set()
                    st.session_state.messages = []
                    st.session_state.suggested_questions = None
                    st.success("Database cleared successfully!")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error clearing database: {str(e)}")

        if process_button and url_input:
            if url_input not in st.session_state.urls_processed:
                st.session_state.is_processing = True
                await process_url(url_input)
                st.session_state.urls_processed.add(url_input)
                st.session_state.processing_complete = True
                st.session_state.is_processing = False
                st.session_state.suggested_questions = None
                st.experimental_rerun()
            else:
                st.warning("This URL has already been processed!")

        if st.session_state.urls_processed:
            st.subheader("Processed URLs:")
            all_urls = list(st.session_state.urls_processed)
            for url in all_urls[:3]:
                st.write(f"✓ {url}")
            remaining = len(all_urls) - 3
            if remaining > 0:
                st.write(f"_...and {remaining} more_")
                with st.expander("Show all URLs"):
                    for url in all_urls[3:]:
                        st.write(f"✓ {url}")

    # -----------------------------------
    # RIGHT COLUMN: Chat Interface
    # -----------------------------------
    with chat_col:
        if st.session_state.processing_complete:
            chat_container = st.container()
            with chat_container:
                st.subheader("Chat Interface")

                # Suggested questions
                with st.expander("📝 Suggested Questions", expanded=False):
                    if existing_data and existing_data["doc_count"] > 0:
                        if st.session_state.suggested_questions is None:
                            st.session_state.suggested_questions = await generate_contextual_questions()
                        st.markdown("Try asking:")
                        for q in st.session_state.suggested_questions:
                            st.markdown(f"- {q}")
                        if st.button("🔄 Refresh Suggestions"):
                            st.session_state.suggested_questions = await generate_contextual_questions()
                            st.experimental_rerun()
                    else:
                        st.markdown("Process some URLs to get contextual suggestions.")

                # Display message history
                messages_container = st.container()
                with messages_container:
                    display_history_messages()

                # Input at the bottom
                user_input = st.chat_input(
                    "Ask a question about the processed content...",
                    disabled=st.session_state.is_processing,
                )

                if user_input:
                    st.session_state.messages.append(
                        {"role": "user", "content": user_input}
                    )
                    with st.chat_message("user"):
                        st.markdown(user_input)

                    with st.chat_message("assistant"):
                        st.write("Thinking...")
                        ans = await answer_question(user_input)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": ans}
                        )
                        st.markdown(ans)

                # Clear chat button
                if st.button("Clear Chat History"):
                    st.session_state.messages = []
                    st.experimental_rerun()
        else:
            if existing_data:
                st.info("The knowledge base is ready! Start asking questions below.")
            else:
                st.info("Please process a URL first to start chatting!")

    # Footer or system status
    st.markdown("---")
    if existing_data:
        st.markdown(
            f"System Status: 🟢 Ready with {existing_data['doc_count']} documents "
            f"from {len(existing_data['domains'])} sources"
        )
    else:
        st.markdown("System Status: 🟡 Waiting for content")


if __name__ == "__main__":
    # Attempt to recreate the collection at startup (optional)
    try:
        vectors_config = VectorParams(size=384, distance="Cosine")
        asyncio.run(
            qdrant_client.recreate_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=vectors_config
            )
        )
    except Exception as e:
        print(f"Warning: Could not recreate collection - {e}")

    # Launch the Streamlit app
    asyncio.run(main())
