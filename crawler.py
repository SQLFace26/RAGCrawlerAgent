import os
import json
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
import httpx  # For local Mistral API calls

from db import init_collection

load_dotenv()

# Initialize Qdrant collection
qdrant_client, collection_name = init_collection()

# Set up local Mistral LLM API (assuming Ollama)
MISTRAL_URL = "http://localhost:11434/api/generate"  # Adjust if different


@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]


def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks while preserving meaningful breaks."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        chunk = text[start:end]
        last_break = max(chunk.rfind("\n\n"), chunk.rfind(". "), chunk.rfind(" "))
        if last_break > chunk_size * 0.3:
            end = start + last_break + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(start + 1, end)

    return chunks


async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using local Mistral LLM."""
    system_prompt = """You are an AI that extracts titles and summaries from web content chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""

    payload = {
        "model": "mistral",  # Update if using a different local model
        "prompt": f"{system_prompt}\n\nURL: {url}\n\nContent:\n{chunk[:1000]}...",
        "stream": False,
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(MISTRAL_URL, json=payload)
            response.raise_for_status()
            output = response.json().get("response", "{}")
            return json.loads(output) if output else {"title": "Unknown", "summary": "No summary available"}
        except Exception as e:
            print(f"Error getting title and summary: {e}")
            return {"title": "Error", "summary": "Error processing summary"}


async def get_embedding(text: str) -> List[float]:
    """Generate embeddings using a local model (adjust if using a specific embedding model)."""
    payload = {"model": "mistral", "prompt": f"Generate an embedding for: {text}", "stream": False}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(MISTRAL_URL, json=payload)
            response.raise_for_status()
            embedding = response.json().get("embedding", [])
            return embedding if embedding else [0.0] * 1536
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return [0.0] * 1536


async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    extracted = await get_title_and_summary(chunk, url)
    embedding = await get_embedding(chunk)

    metadata = {
        "source": urlparse(url).netloc,
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path,
    }

    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted["title"],
        summary=extracted["summary"],
        content=chunk,
        metadata=metadata,
        embedding=embedding,
    )


async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Qdrant."""
    try:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                rest.PointStruct(
                    id=f"{chunk.url}_{chunk.chunk_number}",
                    vector=chunk.embedding,
                    payload={
                        "url": chunk.url,
                        "chunk_number": chunk.chunk_number,
                        "title": chunk.title,
                        "summary": chunk.summary,
                        **chunk.metadata,
                    },
                )
            ],
        )
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
    except Exception as e:
        print(f"Error inserting chunk: {e}")


async def process_and_store_document(url: str, markdown: str):
    """Process a document and store its chunks in parallel."""
    chunks = chunk_text(markdown)
    tasks = [process_chunk(chunk, i, url) for i, chunk in enumerate(chunks)]
    processed_chunks = await asyncio.gather(*tasks)
    insert_tasks = [insert_chunk(chunk) for chunk in processed_chunks]
    await asyncio.gather(*insert_tasks)


async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    print(f"Found {len(urls)} URLs to crawl")

    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        semaphore = asyncio.Semaphore(max_concurrent)
        total_urls = len(urls)
        processed_urls = 0

        async def process_url(url: str):
            nonlocal processed_urls
            async with semaphore:
                result = await crawler.arun(url=url, config=crawl_config, session_id="session1")
                if result.success:
                    processed_urls += 1
                    print(f"Successfully crawled: {url} ({processed_urls}/{total_urls})")
                    await process_and_store_document(url, result.markdown_v2.raw_markdown)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")

        await asyncio.gather(*[process_url(url) for url in urls])
        print(f"Completed crawling {processed_urls} out of {total_urls} URLs")
    finally:
        await crawler.close()


def get_urls_from_sitemap(sitemap_url: str) -> List[str]:
    """Get URLs from a sitemap."""
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()

        root = ElementTree.fromstring(response.content)
        namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls = [loc.text for loc in root.findall(".//ns:loc", namespace)]

        print(f"Found {len(urls)} URLs in sitemap: {sitemap_url}")
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []


async def main():
    """Test function to crawl a single URL."""
    urls = ["https://example.com"]
    await crawl_parallel(urls)


if __name__ == "__main__":
    asyncio.run(main())
