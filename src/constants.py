EMBEDDING_MODEL_PATH = "all-MiniLM-L6-v2"  
ASSYMETRIC_EMBEDDING = False  
EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2 produces 384-dimensional embeddings
TEXT_CHUNK_SIZE = 300  # Maximum number of characters in each text chunk for
QDRANT_COLLECTION_NAME = "documents"  # Qdrant collection name for storing embeddings

GROQ_MODEL_NAME = (
    "openai/gpt-oss-20b"  # Name of the model used in Groq for chat functionality
)
GROQ_API_KEY = None  # Will be loaded from GROQ_API_KEY environment variable



# Logging
LOG_FILE_PATH = "logs/app.log"  # File path for the application log file

