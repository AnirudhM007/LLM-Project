from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
import requests
import io
from PyPDF2 import PdfReader
from docx import Document

# --- CORS Middleware for Frontend Communication ---
from fastapi.middleware.cors import CORSMiddleware

# --- Gemini API Integration ---
import google.generativeai as genai

# --- Pinecone Integration ---
from pinecone import Pinecone, PodSpec, ServerlessSpec # Added ServerlessSpec
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid

# --- PostgreSQL Integration ---
import asyncpg

# Load environment variables from .env file (e.g., API keys, DB credentials)
load_dotenv()

app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System Backend",
    version="1.0.0",
    description="Processes documents from URLs, performs RAG with Pinecone, answers questions using Gemini API, and logs to PostgreSQL."
)

# Define allowed origins for CORS. Your React app runs on http://localhost:3000
origins = [
    "http://localhost",
    "http://localhost:3000", # Crucial for your React app to communicate
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configure Gemini API ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it in your .env file.")
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# --- Configure Pinecone ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") # e.g., "us-west-2", "gcp-starter"
# Corrected PINECONE_INDEX_NAME based on your image
PINECONE_INDEX_NAME = "myindex" # Corrected to "myindex"

pinecone_client = None
pinecone_index = None
if PINECONE_API_KEY and PINECONE_ENVIRONMENT:
    try:
        pinecone_client = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        # Check if index exists, create if not (dimension corrected to 1024 for llama-text-embed-v2)
        # COMMENTED OUT: This line caused "Resource already exists" error as index is already created.
        # if PINECONE_INDEX_NAME not in pinecone_client.list_indexes():
        #     pinecone_client.create_index(
        #         name=PINECONE_INDEX_NAME,
        #         dimension=1024, # Corrected dimension based on your image (llama-text-embed-v2)
        #         metric='cosine',
        #         spec=ServerlessSpec(cloud='aws', region='us-east-1') # Corrected to ServerlessSpec and explicit cloud/region
        #     )
        # Access the Index instance via pinecone_client.Index()
        pinecone_index = pinecone_client.Index(PINECONE_INDEX_NAME)
        print(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}")
    except Exception as e:
        print(f"Failed to connect to Pinecone: {e}. Pinecone functionality will be disabled.")
        pinecone_client = None
        pinecone_index = None
else:
    print("Pinecone API key or environment not set. Pinecone functionality will be disabled.")

# --- Configure Embedding Model (OpenAI for text-embedding-ada-002) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embedding_model_client = None
if OPENAI_API_KEY:
    embedding_model_client = OpenAI(api_key=OPENAI_API_KEY)
    print("OpenAI embedding model client initialized.")
else:
    print("OPENAI_API_KEY not set. Embedding generation will be disabled.")
    # If you are using a different embedding model (like llama-text-embed-v2),
    # you would initialize its client here instead of OpenAI's.
    # For example:
    # from some_llama_client import LlamaEmbeddingClient
    # LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
    # if LLAMA_API_KEY:
    #     embedding_model_client = LlamaEmbeddingClient(api_key=LLAMA_API_KEY)


# --- Configure PostgreSQL ---
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

db_pool = None

@app.on_event("startup")
async def startup_db_client():
    global db_pool
    if DB_HOST and DB_NAME and DB_USER and DB_PASSWORD:
        try:
            db_pool = await asyncpg.create_pool(
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                database=DB_NAME
            )
            print("Connected to PostgreSQL database.")
        except Exception as e:
            print(f"Failed to connect to PostgreSQL: {e}. Database logging will be disabled.")
            db_pool = None
    else:
        print("PostgreSQL credentials not fully set in .env. Database logging will be disabled.")

@app.on_event("shutdown")
async def shutdown_db_client():
    if db_pool:
        await db_pool.close()
        print("Disconnected from PostgreSQL database.")


# --- Pydantic Models for Request/Response Data Validation ---
class QueryRequest(BaseModel):
    documents: str # This field will receive the document URL from the frontend
    questions: List[str] # List of questions

class QueryResponse(BaseModel):
    answers: List[str] # List of answers to return

# --- Authentication Middleware ---
@app.middleware("http")
async def verify_authentication(request: Request, call_next):
    # Allow OPTIONS requests (CORS preflight) and favicon.ico requests to pass without authentication
    if request.method == "OPTIONS" or request.url.path == "/favicon.ico":
        return await call_next(request)

    # The expected token from your frontend's Authorization header
    expected_token = "Bearer 5ba298f10582e591e01dd5a437f580a8da1354f64898b2042fc74b9e0968f9d1"
    auth_header = request.headers.get("Authorization")

    # If the Authorization header doesn't match, return a 401 Unauthorized error
    if auth_header != expected_token:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # If authentication passes, proceed with the request
    response = await call_next(request)
    return response

# --- Helper Function for Document Fetching and Parsing ---
async def fetch_and_parse_document(url: str) -> str:
    """
    Fetches content from a given URL and attempts to parse it based on its
    content type. Currently supports PDF, DOCX, and plain text/HTML.
    """
    if not url.startswith(('http://', 'https://')):
        raise ValueError("Invalid URL format. URL must start with http:// or https://")

    try:
        # Fetch the document content from the URL
        response = requests.get(url, stream=True, timeout=10) # Added timeout
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        content_type = response.headers.get('Content-Type', '').lower()
        document_content = ""

        # Parse based on content type
        if 'application/pdf' in content_type:
            # For PDF, read bytes and use PyPDF2
            pdf_file = io.BytesIO(response.content)
            reader = PdfReader(pdf_file)
            # Check if PDF has pages before extracting text
            if len(reader.pages) > 0:
                for page in reader.pages:
                    document_content += page.extract_text() + "\n"
            else:
                print(f"Warning: PDF at {url} has no readable pages.")
                document_content = "" # No content to extract
        elif 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' in content_type:
            # For DOCX, read bytes and use python-docx
            docx_file = io.BytesIO(response.content)
            document = Document(docx_file)
            for para in document.paragraphs:
                document_content += para.text + "\n"
        elif 'text/plain' in content_type or 'text/html' in content_type:
            # For plain text or HTML, simply use response.text
            document_content = response.text
        else:
            # Fallback: if type is unknown, try to read as plain text and warn
            print(f"Warning: Unsupported content type for URL {url}: {content_type}. Attempting to read as plain text.")
            document_content = response.text

        return document_content.strip() # Return cleaned text content

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=408, detail=f"Request to document URL timed out: {url}")
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail=f"Could not connect to document URL: {url}. Check URL and network.")
    except requests.exceptions.RequestException as e:
        # Catch other request-related errors (e.g., 404, 500 from the document source)
        raise HTTPException(status_code=400, detail=f"Failed to fetch document from URL '{url}': {e}")
    except ValueError as e: # Catch invalid URL format from our added check
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle parsing or other unexpected errors
        raise HTTPException(status_code=500, detail=f"Error parsing document from '{url}': {e}")

# --- Main API Endpoint for Query Retrieval ---
@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_query_retrieval(request_data: QueryRequest):
    # Declare global variables to be modified within this function's except blocks
    # This ensures that even if initialization fails, these variables are accessible as None
    global pinecone_index, embedding_model_client

    document_url = request_data.documents
    questions = request_data.questions

    # Step 1: Fetch and Parse Document Content from the provided URL
    try:
        document_content = await fetch_and_parse_document(document_url)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during document processing: {e}")

    if not document_content.strip():
        raise HTTPException(status_code=400, detail="Document content could not be retrieved or was empty after parsing. Please check the URL and document content.")

    # --- Pinecone Integration: Chunking, Embedding, and Upserting (Simplified Ingestion) ---
    # NOTE: In a production system, document ingestion (chunking, embedding, upserting to Pinecone)
    # would typically be a separate, asynchronous background process or a dedicated endpoint,
    # triggered when a new document is added, not on every query.
    # For this demonstration, we'll perform it here for simplicity.
    if pinecone_index and embedding_model_client:
        try:
            # 1. Chunk the document
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(document_content)

            # 2. Generate embeddings for chunks and upsert to Pinecone
            vectors_to_upsert = []
            for i, chunk in enumerate(chunks):
                # Generate a unique ID for each chunk
                chunk_id = f"{uuid.uuid4()}" # Using UUID for unique ID
                embedding_response = embedding_model_client.embeddings.create(
                    input=[chunk],
                    model="text-embedding-ada-002" # Or a Google embedding model like 'text-embedding-004'
                )
                embedding = embedding_response.data[0].embedding
                vectors_to_upsert.append({
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": {"text": chunk, "source_url": document_url, "chunk_index": i}
                })
            
            # Upsert in batches if many vectors
            if vectors_to_upsert:
                # Clear existing vectors for this document if re-uploading, or manage IDs carefully
                # For simplicity, we just upsert new ones. In a real system, you'd manage updates.
                pinecone_index.upsert(vectors=vectors_to_upsert)
                print(f"Upserted {len(vectors_to_upsert)} chunks to Pinecone for {document_url}.")

        except Exception as e:
            print(f"Error during Pinecone ingestion for {document_url}: {e}. Proceeding without Pinecone search.")
            # Set pinecone_index to None to fallback to full document context
            pinecone_index = None # Temporarily disable Pinecone search for this request
            embedding_model_client = None # Also disable embedding client

    all_answers = []

    # Iterate through each question and generate an answer
    for question in questions:
        context = document_content # Default context is full document

        # --- Pinecone Integration: Semantic Search ---
        if pinecone_index and embedding_model_client:
            try:
                # Create embedding for the question
                query_embedding_response = embedding_model_client.embeddings.create(
                    input=[question],
                    model="text-embedding-ada-002" # Same model as for document chunks
                )
                query_embedding = query_embedding_response.data[0].embedding

                # Query Pinecone for relevant chunks
                results = pinecone_index.query(
                    vector=query_embedding,
                    top_k=5, # Retrieve top 5 most relevant chunks
                    include_metadata=True,
                    # Optional: filter by source_url if you're indexing multiple documents
                    # filter={"source_url": document_url}
                )
                # Concatenate relevant text chunks to form the context for the LLM
                relevant_texts = [match.metadata['text'] for match in results.matches]
                if relevant_texts:
                    context = "\n\n".join(relevant_texts)
                    print(f"Retrieved {len(relevant_texts)} relevant chunks from Pinecone.")
                else:
                    print("No relevant chunks found in Pinecone. Using full document content as context.")
                    context = document_content # Fallback if no relevant text found
            except Exception as e:
                print(f"Error during Pinecone semantic search for question '{question}': {e}. Using full document content as context.")
                context = document_content # Fallback on search error
        else:
            print("Pinecone or embedding model not active. Using full document content as context.")
            context = document_content # Fallback if Pinecone/embedding not configured/active

        # Step 5: Logic Evaluation (Decision processing using LLM)
        try:
            # Construct the prompt for the LLM.
            # This prompt instructs the LLM to answer based ONLY on the provided context.
            llm_prompt = f"Given the following document content:\n\n---\n{context}\n---\n\nAnswer the following question based ONLY on the provided document. If the information is not in the document, state that it's not found.\n\nQuestion: {question}\n\nAnswer:"

            # --- Gemini API Call ---
            response = await gemini_model.generate_content_async(llm_prompt)
            answer = response.text.strip()

            all_answers.append(answer)
        except Exception as e:
            print(f"Error processing question with Gemini API: {e}")
            all_answers.append(f"Error retrieving answer for: {question}. (LLM processing failed)")

    # Step 6: Return structured JSON Output
    response_data = QueryResponse(answers=all_answers)

    # --- PostgreSQL Integration: Log Query and Response ---
    if db_pool:
        try:
            async with db_pool.acquire() as connection:
                await connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS query_logs (
                        id SERIAL PRIMARY KEY,
                        document_url TEXT NOT NULL,
                        questions JSONB NOT NULL,
                        answers JSONB NOT NULL,
                        timestamp TIMESTAMPTZ DEFAULT NOW()
                    );
                    INSERT INTO query_logs (document_url, questions, answers, timestamp)
                    VALUES ($1, $2, $3, NOW())
                    """,
                    document_url, questions, all_answers # Store lists as JSONB
                )
                print("Query and answers logged to PostgreSQL.")
        except Exception as e:
            print(f"Error logging query to PostgreSQL: {e}")
    else:
        print("PostgreSQL connection not active. Skipping database logging.")

    return response_data
