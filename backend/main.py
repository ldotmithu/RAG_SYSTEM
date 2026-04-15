
import logging
import os
from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.chat import generate_response_streaming
from src.ingestion import bulk_index_documents
from src.qdrant_client import get_all_documents, delete_by_document_name
from src.utils import setup_logging
from src.embeddings import get_embedding_model

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


# Request/Response Models
class ChatRequest(BaseModel):
    query: str
    use_rag: bool = True
    use_hybrid: bool = True
    num_results: int = 5
    temperature: float = 0.7
    chat_history: list = []


class ChatResponse(BaseModel):
    response: str
    sources: list


class DocumentInfo(BaseModel):
    name: str
    chunks: int


class DocumentList(BaseModel):
    documents: list


# Lifespan context manager for app startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle app startup and shutdown events"""
    # Startup
    logger.info(" RAG Backend starting up...")
    try:
        from src.qdrant_client import ensure_collection_exists
        ensure_collection_exists()
        logger.info(" Qdrant collection ready")
        
        # Pre-load embedding model in background (non-blocking)
        logger.info(" Pre-loading embedding model (this may take a few minutes on first run)...")
        import threading
        def load_model_bg():
            try:
                from src.embeddings import get_embedding_model
                get_embedding_model()
                logger.info(" Embedding model cached and ready")
            except Exception as e:
                logger.warning(f" Model loading failed (will retry on first request): {e}")
        
        thread = threading.Thread(target=load_model_bg, daemon=True)
        thread.start()
        
    except Exception as e:
        logger.error(f" Failed to initialize: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info(" RAG Backend shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="AI - RAG API",
    description="Production-grade RAG system with Groq LLM and Qdrant vector DB",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "AI RAG Backend",
        "status": "running",
        "version": "2.0.0",
        "docs": "http://localhost:8000/docs",
        "endpoints": {
            "health": "/health",
            "chat": "POST /api/chat",
            "documents": {
                "list": "GET /api/documents",
                "upload": "POST /api/documents/upload",
                "delete": "DELETE /api/documents/{name}"
            },
            "stats": "GET /api/system/stats"
        }
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "service": "AI RAG Backend",
        "version": "2.0.0"
    }


# Chat endpoints
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Stream chat response with RAG context
    
    Args:
        request: ChatRequest with query, RAG settings, and chat history
        
    Returns:
        ChatResponse with LLM response and source documents
    """
    try:
        logger.info(f" Chat request: {request.query[:50]}...")
        
        response_stream = generate_response_streaming(
            query=request.query,
            use_hybrid_search=request.use_rag,
            num_results=request.num_results,
            temperature=request.temperature,
            chat_history=request.chat_history,
        )
        
        if response_stream is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate response. Check API keys and configuration."
            )
        
        # Collect streamed response
        full_response = ""
        for chunk in response_stream:
            if isinstance(chunk, str):
                full_response += chunk
        
        logger.info(f"✅ Response generated ({len(full_response)} chars)")
        
        return ChatResponse(
            response=full_response,
            sources=[]  # Sources can be extracted from the Qdrant search
        )
    
    except Exception as e:
        logger.error(f" Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Document management endpoints
@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and index a PDF document
    
    Args:
        file: PDF file to upload
        
    Returns:
        Upload status with document info
    """
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )
        
        logger.info(f" Uploading document: {file.filename}")
        
        # Read file contents
        contents = await file.read()
        
        # Save to uploaded_files directory
        os.makedirs("uploaded_files", exist_ok=True)
        filepath = os.path.join("uploaded_files", file.filename)
        
        with open(filepath, "wb") as f:
            f.write(contents)
        
        logger.info(f" File saved: {filepath}")
        
        # Process document synchronously
        try:
            import PyPDF2
            from src.ingestion import bulk_index_documents
            from src.embeddings import get_embedding_model
            from src.utils import chunk_text
            
            # Extract text from PDF
            pdf_reader = PyPDF2.PdfReader(BytesIO(contents))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            if not text.strip():
                raise ValueError("No text extracted from PDF")
            
            # Clean and chunk text
            chunks = chunk_text(text, chunk_size=500, overlap=100)
            logger.info(f" Created {len(chunks)} chunks from {file.filename}")
            
            # Generate embeddings
            embedding_model = get_embedding_model()
            documents_to_index = []
            
            for idx, chunk in enumerate(chunks):
                embedding = embedding_model.encode(chunk)
                documents_to_index.append({
                    "document_name": file.filename,
                    "text": chunk,
                    "embedding": embedding,
                    "page_number": idx
                })
            
            
            success_count, errors = bulk_index_documents(documents_to_index)
            
            if errors:
                logger.warning(f" Indexing completed with errors: {errors}")
            else:
                logger.info(f" Successfully indexed {success_count} chunks from {file.filename}")
            
        except Exception as e:
            logger.error(f" Error indexing document: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Indexing error: {str(e)}")
        
        return {
            "status": "success",
            "filename": file.filename,
            "message": "Document uploaded and indexed"
        }
    
    except Exception as e:
        logger.error(f" Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents", response_model=DocumentList)
async def list_documents():
    """Get list of all indexed documents"""
    try:
        logger.info(" Fetching document list...")
        documents = get_all_documents()
        doc_list = [{"name": doc} for doc in documents]
        logger.info(f" Found {len(doc_list)} documents")
        return DocumentList(documents=doc_list)
    
    except Exception as e:
        logger.error(f" Error fetching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/documents/{document_name}")
async def delete_document(document_name: str):
    """
    Delete a document and all its indexed chunks
    
    Args:
        document_name: Name of document to delete
        
    Returns:
        Deletion status
    """
    try:
        logger.info(f" Deleting document: {document_name}")
        delete_by_document_name(document_name)
        logger.info(f" Document deleted: {document_name}")
        
        return {
            "status": "success",
            "message": f"Document '{document_name}' deleted successfully"
        }
    
    except Exception as e:
        logger.error(f" Deletion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/system/stats")
async def system_stats():
    """Get system statistics"""
    try:
        documents = get_all_documents()
        return {
            "indexed_documents": len(documents),
            "vector_db": "Qdrant (In-Memory)",
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_provider": "Groq",
            "search_mode": "Hybrid (Vector + Keyword)"
        }
    except Exception as e:
        logger.error(f" Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
