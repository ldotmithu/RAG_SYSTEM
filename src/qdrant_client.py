import logging
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.constants import EMBEDDING_DIMENSION, QDRANT_COLLECTION_NAME
from src.utils import setup_logging

# Initialize logger
setup_logging()
logger = logging.getLogger(__name__)

# Use in-memory Qdrant for simplicity (no external DB needed)
# To use persistent storage: QdrantClient(path="./qdrant_storage")
# To use remote: QdrantClient(url="http://localhost:6333")
client = None


def get_qdrant_client() -> QdrantClient:
    """
    Initializes and returns a Qdrant client (in-memory).

    Returns:
        QdrantClient: Configured Qdrant client instance.
    """
    global client
    if client is None:
        client = QdrantClient(":memory:")
        logger.info("Qdrant client initialized (in-memory).")
    return client


def ensure_collection_exists() -> bool:
    """
    Ensures that the Qdrant collection exists, creates it if not.

    Returns:
        bool: True if collection exists or was created successfully, False otherwise.
    """
    try:
        c = get_qdrant_client()
        collections = c.get_collections()
        collection_names = [col.name for col in collections.collections]

        if QDRANT_COLLECTION_NAME not in collection_names:
            c.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection '{QDRANT_COLLECTION_NAME}'")
        else:
            logger.info(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' already exists.")
        return True
    except Exception as e:
        logger.error(f"Error ensuring collection exists: {str(e)}")
        return False


def add_points(
    points_data: List[Dict[str, Any]], embedding_list: List[List[float]]
) -> bool:
    """
    Adds documents to Qdrant collection.

    Args:
        points_data (List[Dict[str, Any]]): List of document data dictionaries.
        embedding_list (List[List[float]]): List of embeddings corresponding to documents.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        c = get_qdrant_client()
        
        # Create points for Qdrant
        points = []
        for idx, (data, embedding) in enumerate(zip(points_data, embedding_list)):
            point = PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    "text": data.get("text", ""),
                    "document_name": data.get("document_name", ""),
                    "page_number": data.get("page_number", 0),
                }
            )
            points.append(point)
        
        c.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=points,
        )
        logger.info(f"Added {len(points)} points to Qdrant collection.")
        return True
    except Exception as e:
        logger.error(f"Error adding points to Qdrant: {str(e)}")
        return False


def search(
    query_embedding: List[float], top_k: int = 5, score_threshold: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Performs vector similarity search in Qdrant.

    Args:
        query_embedding (List[float]): Query embedding vector.
        top_k (int, optional): Number of top results to retrieve. Defaults to 5.
        score_threshold (float, optional): Minimum similarity score. Defaults to 0.0.

    Returns:
        List[Dict[str, Any]]: List of search results with text and metadata.
    """
    try:
        c = get_qdrant_client()
        
        # For qdrant-client 1.17.1, use query_points
        results = c.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=query_embedding,
            limit=top_k,
            score_threshold=score_threshold,
        ).points
        
        # Format results to match OpenSearch format
        formatted_results = []
        for result in results:
            formatted_results.append({
                "_id": str(result.id),
                "_source": {
                    "text": result.payload.get("text", ""),
                    "document_name": result.payload.get("document_name", ""),
                    "page_number": result.payload.get("page_number", 0),
                },
                "_score": result.score,
            })
        
        logger.info(f"Search completed. Found {len(formatted_results)} results.")
        return formatted_results
    except Exception as e:
        logger.error(f"Error searching Qdrant: {str(e)}")
        # Fallback: return empty list or all documents
        try:
            c = get_qdrant_client()
            points, _ = c.scroll(collection_name=QDRANT_COLLECTION_NAME, limit=top_k)
            
            formatted_results = []
            for point in points:
                formatted_results.append({
                    "_id": str(point.id),
                    "_source": {
                        "text": point.payload.get("text", ""),
                        "document_name": point.payload.get("document_name", ""),
                        "page_number": point.payload.get("page_number", 0),
                    },
                    "_score": 1.0,
                })
            
            logger.info(f"Fallback search returned {len(formatted_results)} results.")
            return formatted_results
        except Exception as e2:
            logger.error(f"Fallback search also failed: {str(e2)}")
            return []


def delete_by_document_name(document_name: str) -> bool:
    """
    Deletes all points with a specific document name.

    Args:
        document_name (str): Name of the document to delete.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        c = get_qdrant_client()
        
        c.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector={
                "filter": {
                    "must": [
                        {"key": "document_name", "match": {"value": document_name}}
                    ]
                }
            }
        )
        logger.info(f"Deleted document '{document_name}' from Qdrant.")
        return True
    except Exception as e:
        logger.error(f"Error deleting document from Qdrant: {str(e)}")
        return False


def get_all_documents() -> List[str]:
    """
    Retrieves all unique document names from the collection.

    Returns:
        List[str]: List of unique document names.
    """
    try:
        c = get_qdrant_client()
        # Scroll through all points to get unique document names
        points, _ = c.scroll(collection_name=QDRANT_COLLECTION_NAME, limit=10000)
        
        unique_docs = set()
        for point in points:
            doc_name = point.payload.get("document_name", "")
            if doc_name:
                unique_docs.add(doc_name)
        
        logger.info(f"Retrieved {len(unique_docs)} unique documents from Qdrant.")
        return sorted(list(unique_docs))
    except Exception as e:
        logger.error(f"Error retrieving documents from Qdrant: {str(e)}")
        return []
