import logging
import os
from typing import Dict, Iterable, List, Optional

from groq import Groq
import streamlit as st

from src.constants import ASSYMETRIC_EMBEDDING, GROQ_MODEL_NAME
from src.embeddings import get_embedding_model
from src.qdrant_client import search as qdrant_search
from src.utils import setup_logging
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize logger
setup_logging()
logger = logging.getLogger(__name__)


def get_groq_client() -> Groq:
    """
    Initializes the Groq client with API key from environment variable.

    Returns:
        Groq: The initialized Groq client.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY environment variable not set. Please set it to use Groq.")
        raise ValueError("GROQ_API_KEY environment variable is required")
    
    # Create client with only api_key parameter (no caching to avoid Streamlit issues)
    return Groq(api_key=api_key)


def run_groq_streaming(prompt: str, temperature: float) -> Optional[Iterable[str]]:
    """
    Uses Groq's Python library to run the model with streaming enabled.

    Args:
        prompt (str): The prompt to send to the model.
        temperature (float): The response generation temperature.

    Returns:
        Optional[Iterable[str]]: A generator yielding response chunks as strings, or None if an error occurs.
    """

    try:
        # Initialize Groq client
        client = get_groq_client()
        
        # Stream the response from the model
        logger.info("Streaming response from Groq model.")
        stream = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=temperature,
        )
        
        # Generator to yield text chunks from the stream
        def chunk_generator():
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        return chunk_generator()
    except Exception as e:
        logger.error(f"Error during streaming with Groq: {str(e)}")
        logger.info("Falling back to non-streaming response.")
        def error_response():
            yield "I apologize, but I encountered an error while connecting to Groq. Please make sure your GROQ_API_KEY is set correctly and try again."
        return error_response()


def prompt_template(query: str, context: str, history: List[Dict[str, str]]) -> str:
    """
    Builds the prompt with context, conversation history, and user query.

    Args:
        query (str): The user's query.
        context (str): Context text gathered from hybrid search.
        history (List[Dict[str, str]]): Conversation history to include in the prompt.

    Returns:
        str: Constructed prompt for Groq model.
    """
    prompt = """You are an expert AI assistant that answers questions clearly and accurately.

FORMATTING RULES (IMPORTANT):
- Use clear, well-structured responses with proper formatting.
- Use **bold** for key terms and important concepts.
- Use bullet points (- ) or numbered lists (1. ) for multiple items.
- Use headings (## or ###) to organize longer answers into sections.
- Use code blocks (```) for any code, commands, or technical syntax.
- Use > blockquotes for important notes or callouts.
- Keep paragraphs short and readable (2-3 sentences max per paragraph).
- Always end with a clear conclusion or summary if the answer is long.

"""
    if context:
        prompt += (
            "Use the following context from the user's documents to answer the question. "
            "Reference specific details from the context when relevant.\n\n"
            "---\n"
            "CONTEXT:\n"
            + context
            + "\n---\n\n"
        )
    else:
        prompt += "No document context is available. Answer to the best of your knowledge.\n\n"

    if history:
        prompt += "CONVERSATION HISTORY:\n"
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"]
            prompt += f"{role}: {content}\n"
        prompt += "\n"

    prompt += f"User: {query}\nAssistant:"
    logger.info("Prompt constructed with context and conversation history.")
    return prompt


def generate_response_streaming(
    query: str,
    use_hybrid_search: bool,
    num_results: int,
    temperature: float,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Optional[Iterable[str]]:
    """
    Generates a chatbot response by performing semantic search and incorporating conversation history.

    Args:
        query (str): The user's query.
        use_hybrid_search (bool): Whether to use RAG for context retrieval.
        num_results (int): The number of search results to include in the context.
        temperature (float): The temperature for the response generation.
        chat_history (Optional[List[Dict[str, str]]]): List of chat history messages.

    Returns:
        Optional[Iterable[str]]: A generator yielding response chunks as strings, or None if an error occurs.
    """
    chat_history = chat_history or []
    max_history_messages = 10
    history = chat_history[-max_history_messages:]
    context = ""

    
    if use_hybrid_search:
        if ASSYMETRIC_EMBEDDING:
            prefixed_query = f"passage: {query}"
        else:
            prefixed_query = f"{query}"
        
        embedding_model = get_embedding_model()
        query_embedding = embedding_model.encode(
            prefixed_query
        ).tolist()  
        
        logger.info("Performing VECTOR search with Qdrant.")
        search_results = qdrant_search(query_embedding, top_k=num_results)
        logger.info(f"Vector search completed with {len(search_results)} results.")

        for i, result in enumerate(search_results):
            doc_name = result['_source'].get('document_name', 'Unknown')
            page = result['_source'].get('page_number', '?')
            score = result.get('_score', 0)
            context += f"[Source: {doc_name}, Chunk {page}, Score: {score:.3f}]\n{result['_source']['text']}\n\n"

    
    prompt = prompt_template(query, context, history)

    return run_groq_streaming(prompt, temperature)
