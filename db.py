# db.py
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

def get_qdrant_client():
    """
    Instantiate and return a Qdrant client.
    Adjust host, port, or other parameters as needed
    for your setup.
    """
    return QdrantClient(
        url="http://localhost:6333",  # or wherever Qdrant is running
        # api_key="YOUR_API_KEY"      # If applicable
    )

def init_collection(
    collection_name: str = "tonygray_palottery_ai_docs",
    vector_size: int = 1536  # Adjust dimension to match your embeddings
):
    """
    Create the collection in Qdrant if it does not exist.
    Returns the Qdrant client and the collection name so you can
    perform upserts, searches, etc. using the client.
    """
    client = get_qdrant_client()

    # Check if the collection exists by trying to get it
    try:
        client.get_collection(collection_name=collection_name)
        # Collection exists; do nothing
    except Exception:
        # If the collection is not found, create it
        client.create_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(
                size=vector_size,
                distance=rest.Distance.COSINE  # or euclid, dot
            )
        )

    # Return the client and collection name
    return client, collection_name


if __name__ == "__main__":
    # Initialize the Qdrant collection
    client, collection = init_collection()

    # Now you can insert vectors, query, etc.
    # For example:
    # client.upsert(
    #     collection_name=collection,
    #     points=[
    #         rest.PointStruct(
    #             id=1,
    #             vector=[0.1, 0.2, 0.3, ...],  # your embedding
    #             payload={"some_metadata": "foo"}
    #         )
    #         # ... more points
    #     ]
    # )
