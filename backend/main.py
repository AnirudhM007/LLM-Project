from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
import requests
import io
from PyPDF2 import PdfReader
from docx import Document
import json # <--- ADDED: Import the json module for serialization

# --- CORS Middleware for Frontend Communication ---
from fastapi.middleware.cors import CORSMiddleware

# --- Gemini API Integration ---
import google.generativeai as genai

# --- Pinecone Integration ---
from pinecone import Pinecone, PodSpec, ServerlessSpec
from openai import OpenAI # Re-import OpenAI for embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid

# --- PostgreSQL Integration ---
import asyncpg

# Load environment variables from .env file (e.g., API keys, DB credentials)
load_dotenv()

app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System Backend",
    version="1.0.0",
    description="Separated document ingestion and query retrieval processes for improved performance."
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

# --- Configure Gemini API (for Text Generation) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it in your .env file.")
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash') # For text generation

# --- Configure OpenAI Embedding Model ---
# NOTE: As discussed, your OpenAI free tier is exhausted.
# This code will still attempt to use OpenAI for embeddings.
# If you continue to get 429 errors, consider:
# 1. Adding payment details to your OpenAI account.
# 2. Switching to Google Gemini embeddings (requires modifying the embedding logic below)
# 3. Using a local open-source embedding model.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embedding_model_client = None
if OPENAI_API_KEY:
    embedding_model_client = OpenAI(api_key=OPENAI_API_KEY)
    print("OpenAI embedding model client initialized.")
else:
    print("OPENAI_API_KEY not set. Embedding generation will be disabled.")


# --- Configure Pinecone ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = "myindex" # Ensure this matches your Pinecone index name

pinecone_client = None
pinecone_index = None
if PINECONE_API_KEY and PINECONE_ENVIRONMENT:
    try:
        pinecone_client = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

        # Attempt to create index, but catch ALREADY_EXISTS specifically
        try:
            pinecone_client.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=1536, # IMPORTANT: Changed dimension back to 1536 for OpenAI's text-embedding-ada-002
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1') # Adjust cloud/region if different
            )
            print(f"Pinecone index '{PINECONE_INDEX_NAME}' created successfully.")
        except Exception as create_e:
            if "ALREADY_EXISTS" in str(create_e) or "(409)" in str(create_e):
                print(f"Pinecone index '{PINECONE_INDEX_NAME}' already exists. Connecting to existing index.")
            else:
                raise create_e

        pinecone_index = pinecone_client.Index(PINECONE_INDEX_NAME)
        print(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}")
    except Exception as e:
        print(f"Failed to connect to Pinecone: {e}. Pinecone functionality will be disabled.")
        pinecone_client = None
        pinecone_index = None
else:
    print("Pinecone API key or environment not set. Pinecone functionality will be disabled.")


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
            # --- Create query_logs table at startup if it doesn't exist ---
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
                    """
                )
                print("PostgreSQL 'query_logs' table ensured.")

            # --- Create ingestion_logs table at startup if it doesn't exist ---
            async with db_pool.acquire() as connection:
                await connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ingestion_logs (
                        id SERIAL PRIMARY KEY,
                        document_url TEXT NOT NULL,
                        indexed_chunks INT NOT NULL,
                        timestamp TIMESTAMPTZ DEFAULT NOW(),
                        status TEXT NOT NULL
                    );
                    """
                )
                print("PostgreSQL 'ingestion_logs' table ensured.")

        except Exception as e:
            print(f"Failed to connect to PostgreSQL or create tables: {e}. Database logging will be disabled.")
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
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

class IngestRequest(BaseModel):
    document_url: str

class IngestResponse(BaseModel):
    message: str
    status: str
    indexed_chunks: int = 0


# --- Authentication Middleware ---
@app.middleware("http")
async def verify_authentication(request: Request, call_next):
    # Allow OPTIONS requests (pre-flight CORS) and favicon without authentication
    if request.method == "OPTIONS" or request.url.path == "/favicon.ico":
        return await call_next(request)

    expected_token = "Bearer 5ba298f10582e591e01dd5a437f580a8da1354f64898b2042fc74b9e0968f9d1"
    auth_header = request.headers.get("Authorization")

    if auth_header != expected_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    response = await call_next(request)
    return response

# --- Helper Function for Document Fetching and Parsing ---
async def fetch_and_parse_document(url: str) -> str:
    if not url.startswith(('http://', 'https://')):
        raise ValueError("Invalid URL format. URL must start with http:// or https://")

    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '').lower()
        document_content = ""

        if 'application/pdf' in content_type:
            pdf_file = io.BytesIO(response.content)
            reader = PdfReader(pdf_file)
            if len(reader.pages) > 0:
                for page in reader.pages:
                    document_content += page.extract_text() + "\n"
            else:
                print(f"Warning: PDF at {url} has no readable pages.")
                document_content = ""
        elif 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' in content_type:
            docx_file = io.BytesIO(response.content)
            document = Document(docx_file)
            for para in document.paragraphs:
                document_content += para.text + "\n"
        elif 'text/plain' in content_type or 'text/html' in content_type:
            document_content = response.text
        else:
            print(f"Warning: Unsupported content type for URL {url}: {content_type}. Attempting to read as plain text.")
            document_content = response.text

        return document_content.strip()

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=status.HTTP_408_REQUEST_TIMEOUT, detail=f"Request to document URL timed out: {url}")
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Could not connect to document URL: {url}. Check URL and network.")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to fetch document from URL '{url}': {e}")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error parsing document from '{url}': {e}")


# --- NEW Endpoint for Document Ingestion ---
@app.post("/api/v1/documents/ingest", response_model=IngestResponse)
async def ingest_document_endpoint(request_data: IngestRequest):
    global pinecone_index, embedding_model_client, db_pool

    document_url = request_data.document_url

    if not pinecone_index or not embedding_model_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pinecone or Embedding model not configured/active. Cannot ingest documents."
        )

    try:
        document_content = await fetch_and_parse_document(document_url)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error fetching/parsing document for ingestion: {e}")

    if not document_content.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Document content could not be retrieved or was empty after parsing. Cannot ingest.")

    indexed_chunks_count = 0
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(document_content)

        vectors_to_upsert = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_url.replace('.', '_').replace('/', '_').replace(':', '_')}_{i}_{uuid.uuid4().hex[:8]}"
            # Use OpenAI's embedding model
            embedding_response = embedding_model_client.embeddings.create(
                input=[chunk],
                model="text-embedding-ada-002" # Using OpenAI's embedding model
            )
            embedding = embedding_response.data[0].embedding

            vectors_to_upsert.append({
                "id": chunk_id,
                "values": embedding,
                "metadata": {"text": chunk, "source_url": document_url, "chunk_index": i}
            })
        
        if vectors_to_upsert:
            pinecone_index.upsert(vectors=vectors_to_upsert)
            indexed_chunks_count = len(vectors_to_upsert)
            print(f"Successfully ingested {indexed_chunks_count} chunks to Pinecone for {document_url}.")

        if db_pool:
            try:
                async with db_pool.acquire() as connection:
                    await connection.execute(
                        """
                        INSERT INTO ingestion_logs (document_url, indexed_chunks, timestamp, status)
                        VALUES ($1, $2, NOW(), $3)
                        """,
                        document_url, indexed_chunks_count, "SUCCESS"
                    )
                    print(f"Ingestion for {document_url} logged to PostgreSQL.")
            except Exception as e:
                print(f"Error logging ingestion to PostgreSQL for {document_url}: {e}")
        
        return IngestResponse(
            message=f"Document '{document_url}' ingested successfully.",
            status="success",
            indexed_chunks=indexed_chunks_count
        )

    except Exception as e:
        if db_pool:
            try:
                async with db_pool.acquire() as connection:
                    await connection.execute(
                        """
                        INSERT INTO ingestion_logs (document_url, indexed_chunks, timestamp, status)
                        VALUES ($1, $2, NOW(), $3)
                        """,
                        document_url, 0, f"FAILED: {e}"
                    )
            except Exception as db_e:
                print(f"Critical: Failed to log ingestion error to DB: {db_e}")

        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to ingest document '{document_url}': {e}")


# --- MODIFIED Endpoint for Query Retrieval ---
@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_query_retrieval(request_data: QueryRequest):
    global pinecone_index, embedding_model_client

    document_url = request_data.documents
    questions = request_data.questions

    all_answers = []

    for question in questions:
        context = ""
        
        if pinecone_index and embedding_model_client:
            try:
                # Create embedding for the question
                embedding_response = embedding_model_client.embeddings.create(
                    input=[question],
                    model="text-embedding-ada-002" # Using OpenAI's embedding model
                )
                query_embedding = embedding_response.data[0].embedding

                # Query Pinecone for relevant chunks
                results = pinecone_index.query(
                    vector=query_embedding,
                    top_k=5,
                    include_metadata=True,
                )
                relevant_texts = [match.metadata['text'] for match in results.matches]
                
                if relevant_texts:
                    context = "\n\n".join(relevant_texts)
                    print(f"Retrieved {len(relevant_texts)} relevant chunks from Pinecone for '{question}'.")
                else:
                    print(f"No relevant chunks found in Pinecone for '{question}'.")
            except Exception as e:
                print(f"Error during Pinecone semantic search for question '{question}': {e}. Proceeding with empty context.")
                context = ""
        else:
            print("Pinecone or embedding model not active. Proceeding with empty context for LLM.")
            context = ""

        try:
            llm_prompt = f"Given the following document content (if any):\n\n---\n{context}\n---\n\nAnswer the following question based ONLY on the provided document content. If the information is not in the document, state that it's not found.\n\nQuestion: {question}\n\nAnswer:"

            response = await gemini_model.generate_content_async(llm_prompt)
            answer = response.text.strip()

            all_answers.append(answer)
        except Exception as e:
            print(f"Error processing question with Gemini API: {e}")
            all_answers.append(f"Error retrieving answer for: {question}. (LLM processing failed)")

    response_data = QueryResponse(answers=all_answers)

    if db_pool:
        try:
            async with db_pool.acquire() as connection:
                # Convert lists to JSON strings before inserting into JSONB columns
                questions_json = json.dumps(questions)
                answers_json = json.dumps(all_answers)

                await connection.execute(
                    """
                    INSERT INTO query_logs (document_url, questions, answers, timestamp)
                    VALUES ($1, $2, $3, NOW())
                    """,
                    document_url, questions_json, answers_json # Use the JSON strings here
                )
                print("Query and answers logged to PostgreSQL.")
        except Exception as e:
            print(f"Error logging query to PostgreSQL: {e}")
    else:
        print("PostgreSQL connection not active. Skipping database logging.")

    return response_data
