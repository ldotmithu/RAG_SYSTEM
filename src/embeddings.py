import logging
from typing import Any, List
import os
import time
import socket

from sentence_transformers import SentenceTransformer
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.constants import EMBEDDING_MODEL_PATH
from src.utils import setup_logging
from dotenv import load_dotenv
load_dotenv()
from huggingface_hub import login
login(token=os.getenv("HF_API_KEY"))

# Initialize logger
setup_logging()
logger = logging.getLogger(__name__)

# Global model cache
_embedding_model = None


def _configure_requests_timeout():
    """
    Configure requests library with longer timeouts and automatic retries
    for HuggingFace downloads.
    """
    # Increase socket timeout for large downloads
    socket.setdefaulttimeout(600)  # 10 minutes
    
    # Ensure HF environment variables are set
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '1800'  # 30 minutes
    os.environ['HF_HUB_OFFLINE'] = 'False'
    os.environ['REQUESTS_TIMEOUT'] = '1800'


def get_embedding_model() -> SentenceTransformer:
    """
    Loads and caches the embedding model with improved error handling.
    Uses HuggingFace cache with extended timeouts and retry logic.

    Returns:
        SentenceTransformer: The loaded embedding model.
    """
    global _embedding_model
    
    if _embedding_model is not None:
        return _embedding_model
    
    # Configure timeout settings
    _configure_requests_timeout()
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL_PATH} (Attempt {attempt}/{max_retries})")
            
            # Load with extended timeout and trust remote code for custom models
            # This is required for Jina embeddings and other models with custom code
            _embedding_model = SentenceTransformer(
                EMBEDDING_MODEL_PATH,
                cache_folder=os.path.expanduser("~/.cache/sentence-transformers"),
                device="cpu",
                trust_remote_code=True
            )
            
            logger.info("✅ Embedding model loaded successfully")
            return _embedding_model
            
        except (ModuleNotFoundError, ImportError) as import_error:
            # Handle missing custom modules (common with Jina models)
            error_msg = str(import_error)
            logger.warning(f"⚠️  Import error on attempt {attempt}/{max_retries}: {error_msg}")
            
            if attempt < max_retries:
                # Try to recover by clearing cache and retrying
                logger.info("Attempting recovery: clearing model cache...")
                try:
                    import shutil
                    cache_dir = os.path.expanduser("~/.cache/sentence-transformers")
                    if os.path.exists(cache_dir):
                        shutil.rmtree(cache_dir)
                        logger.info("Cache cleared, retrying...")
                except:
                    pass
                
                wait_time = retry_delay * (2 ** (attempt - 1))
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"❌ Failed to load model after {max_retries} attempts due to import error")
                raise
                
        except (TimeoutError, ConnectionError, requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            logger.warning(f"⚠️  Network timeout on attempt {attempt}/{max_retries}: {str(e)}")
            if attempt < max_retries:
                wait_time = retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"❌ Failed to load model after {max_retries} attempts")
                raise
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {str(e)}")
            raise
