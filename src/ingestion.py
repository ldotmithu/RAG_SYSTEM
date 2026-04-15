import logging
from typing import Any, Dict, List, Tuple

from src.constants import ASSYMETRIC_EMBEDDING
from src.qdrant_client import add_points
from src.utils import setup_logging

# Initialize logger
setup_logging()
logger = logging.getLogger(__name__)
def bulk_index_documents(documents: List[Dict[str, Any]]) -> Tuple[int, List[Any]]:
    """
    Indexes multiple documents into Qdrant.

    Args:
        documents (List[Dict[str, Any]]): List of document dictionaries with 'doc_id', 'text', 'embedding', and 'document_name'.

    Returns:
        Tuple[int, List[Any]]: Tuple with the number of successfully indexed documents and a list of any errors.
    """
    errors = []
    
    try:
        # Prepare documents and embeddings for Qdrant
        points_data = []
        embedding_list = []
        
        for idx, doc in enumerate(documents):
            document_name = doc["document_name"]
            
            # Prefix each document's text with "passage: " for the asymmetric embedding model
            if ASSYMETRIC_EMBEDDING:
                prefixed_text = f"passage: {doc['text']}"
            else:
                prefixed_text = f"{doc['text']}"
            
            points_data.append({
                "text": prefixed_text,
                "document_name": document_name,
                "page_number": doc.get("page_number", 0),
            })
            
            # Convert embedding to list
            embedding_list.append(doc["embedding"].tolist())
        
        # Add points to Qdrant
        success = add_points(points_data, embedding_list)
        
        if success:
            logger.info(f"Bulk indexed {len(documents)} documents into Qdrant.")
            return len(documents), errors
        else:
            logger.error("Failed to bulk index documents into Qdrant.")
            errors.append("Failed to add points to Qdrant")
            return 0, errors
            
    except Exception as e:
        logger.error(f"Error bulk indexing documents: {str(e)}")
        errors.append(str(e))
        return 0, errors
