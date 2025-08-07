# main.py
# Entry point for the FastAPI-based RAG system, with Redis caching for performance.
# UPGRADED with Multi-way Recall and Re-ranking for improved accuracy and speed.

import os
import httpx
import asyncpg
import pypdf
import io
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Dict, Optional, Set
import hashlib
import re
import json
import asyncio
import time
import redis.asyncio as redis

# --- Configuration Management ---
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    google_api_key: str
    pinecone_api_key: str
    pinecone_environment: str
    pinecone_index_name: str
    db_host: str
    db_name: str
    db_user: str
    db_password: str
    redis_url: str = "redis://localhost"
    auth_token: str = "5ba298f10582e591e01dd5a437f580a8da1354f64898b2042fc74b9e0968f9d1"

settings = Settings()

# --- API Models for Submission ---
class HackRxRequest(BaseModel):
    documents: str = Field(..., description="URL to the PDF document to be processed.")
    questions: List[str] = Field(..., description="A list of questions to be answered based on the document.")

class HackRxResponse(BaseModel):
    answers: List[str]

# --- Service Implementations ---

class GeminiService:
    """Handles all interactions with the Google Gemini API."""
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.embedding_model = "models/embedding-001"
        self.generation_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        print("Gemini Service Initialized.")

    async def get_embeddings(self, texts: List[str], task_type: str) -> List[List[float]]:
        if not texts: return []
        print(f"Generating embeddings for {len(texts)} chunks for task: {task_type}...")
        try:
            all_embeddings = []
            for i in range(0, len(texts), 100):
                batch_texts = texts[i:i+100]
                result = genai.embed_content(model=self.embedding_model, content=batch_texts, task_type=task_type)
                all_embeddings.extend(result['embedding'])
            return all_embeddings
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {e}")

    async def generate_batch_hypothetical_answers(self, questions: List[str]) -> List[str]:
        print(f"Generating batch of hypothetical answers for {len(questions)} questions...")
        formatted_questions = "\n".join([f"Q{i+1}: {q}" for i, q in enumerate(questions)])
        prompt = f"You are an expert in insurance policies. Based on the user's list of questions, generate a concise, hypothetical answer for each one. Return the result as a JSON object where keys are 'A1', 'A2', etc. User Questions:\n{formatted_questions}\n\nJSON Output:"
        try:
            response = self.generation_model.generate_content(prompt)
            cleaned_response = response.text.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:-3].strip()
            elif cleaned_response.startswith("```"):
                 cleaned_response = cleaned_response[3:-3].strip()

            answers_dict = json.loads(cleaned_response)
            hypothetical_answers = [answers_dict.get(f"A{i+1}", q) for i, q in enumerate(questions)]
            print(f"Generated {len(hypothetical_answers)} HyDEs in a batch.")
            return hypothetical_answers
        except (json.JSONDecodeError, AttributeError, TypeError) as e:
            print(f"Could not generate or parse batch hypothetical answers. Error: {e}. Falling back to using questions as HyDEs.")
            return questions
        except Exception as e:
            print(f"An unexpected error occurred during HyDE generation. Error: {e}")
            return questions

    async def generate_batch_final_answers(self, contexts: List[str], questions: List[str]) -> List[str]:
        print(f"Generating batch of final answers for {len(questions)} questions...")
        combined_input = ""
        for i, (question, context) in enumerate(zip(questions, contexts)):
            # NEW: Context is now a single, top-ranked chunk.
            combined_input += f"**Question {i+1}:** {question}\n**Context for Q{i+1}:**\n---\n{context if context else 'No context found.'}\n---\n\n"
        prompt = f"You are a specialized assistant. Answer a list of user questions based *exclusively* on the text provided for each question in the 'CONTEXT' section. If the context for a question is insufficient, you MUST respond with 'Cannot answer based on the provided information.' for that specific question. Return a single JSON object with keys 'A1', 'A2', etc. \n\n{combined_input}\n\nJSON Output:"
        try:
            response = self.generation_model.generate_content(prompt)
            cleaned_response = response.text.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:-3].strip()
            elif cleaned_response.startswith("```"):
                 cleaned_response = cleaned_response[3:-3].strip()

            answers_dict = json.loads(cleaned_response)
            final_answers = [answers_dict.get(f"A{i+1}", "Failed to generate an answer.") for i in range(len(questions))]
            return final_answers
        except (json.JSONDecodeError, AttributeError, TypeError) as e:
            print(f"Could not generate or parse batch final answers. Error: {e}")
            return ["Failed to generate a valid answer due to a parsing error." for _ in questions]
        except Exception as e:
            print(f"An unexpected error occurred during final answer generation. Error: {e}")
            return ["Failed to generate an answer due to an unexpected error." for _ in questions]


class PineconeService:
    """Handles all interactions with the Pinecone vector database."""
    def __init__(self, api_key: str, environment: str, index_name: str):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        if self.index_name not in self.pc.list_indexes().names():
            print(f"Creating index '{self.index_name}'...")
            self.pc.create_index(name=self.index_name, dimension=768, metric='cosine', spec=ServerlessSpec(cloud='aws', region=environment))
        self.index = self.pc.Index(self.index_name)
        print("Pinecone Service Initialized.")

    async def upsert_documents(self, chunks: List[str], embeddings: List[List[float]], namespace: str):
        if not chunks or not embeddings: return
        print(f"Upserting {len(chunks)} documents into Pinecone namespace: {namespace}")
        vectors = [{"id": f"chunk_{i}", "values": emb, "metadata": {"text": chunk}} for i, (chunk, emb) in enumerate(zip(chunks, embeddings))]
        for i in range(0, len(vectors), 100):
            batch = vectors[i:i+100]
            self.index.upsert(vectors=batch, namespace=namespace)
        print("Upsert complete.")

    async def query_vector(self, query_embedding: List[float], top_k: int, namespace: str) -> List[str]:
        print(f"Querying Pinecone namespace '{namespace}' with vector search...")
        result = self.index.query(namespace=namespace, vector=query_embedding, top_k=top_k, include_metadata=True)
        return [match['metadata']['text'] for match in result['matches']]


class PostgresService:
    """Handles interactions with the PostgreSQL database for logging."""
    _pool: Optional[asyncpg.Pool] = None

    async def get_pool(self) -> asyncpg.Pool:
        if self._pool is None or self._pool._closed:
            print("Initializing PostgreSQL connection pool...")
            try:
                self._pool = await asyncpg.create_pool(host=settings.db_host, database=settings.db_name, user=settings.db_user, password=settings.db_password)
            except Exception as e:
                print(f"Failed to connect to or initialize PostgreSQL: {e}")
                raise
        return self._pool

    async def log_interaction(self, document_url: str, questions: List[str], answers: List[str]):
        print("Logging interaction to PostgreSQL...")
        try:
            pool = await self.get_pool()
            async with pool.acquire() as conn:
                await conn.execute("CREATE TABLE IF NOT EXISTS hackrx_run_logs (id SERIAL PRIMARY KEY, document_url TEXT, questions TEXT, answers TEXT, created_at TIMESTAMPTZ DEFAULT NOW());")
                await conn.execute("INSERT INTO hackrx_run_logs (document_url, questions, answers) VALUES ($1, $2, $3)", document_url, str(questions), str(answers))
        except Exception as e:
            print(f"WARNING: Failed to log interaction to PostgreSQL. Error: {e}")

# --- Document Processing Utilities ---

async def download_and_parse_pdf(url: str) -> str:
    print(f"Downloading and parsing document from: {url}")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, follow_redirects=True, timeout=12.0)
            response.raise_for_status()
            pdf_file = io.BytesIO(response.content)
            reader = pypdf.PdfReader(pdf_file)
            text = " ".join(page.extract_text() or "" for page in reader.pages)
            return re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download or parse document: {e}")

def sentence_aware_splitter(text: str, chunk_size: int = 3000) -> List[str]:
    if not text: return []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = ""
        current_chunk += sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return [c for c in chunks if c]

# --- NEW: Advanced RAG Core Logic ---

def keyword_search(query: str, chunks: List[str], top_k: int = 5) -> List[str]:
    """Performs a simple keyword search and returns top matching chunks."""
    query_words = set(query.lower().split())
    scores = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(query_words.intersection(chunk_words))
        if score > 0:
            scores.append((score, chunk))
    
    scores.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scores[:top_k]]

def rerank_chunks(query: str, chunks: List[str]) -> str:
    """
    A simple re-ranking implementation. It prioritizes chunks that contain more of the query's keywords.
    A more advanced version would use a dedicated cross-encoder model.
    """
    if not chunks:
        return ""
    
    query_words = set(query.lower().split())
    
    best_chunk = ""
    max_score = -1

    # Use a set to avoid re-ranking duplicate chunks from keyword and vector searches
    unique_chunks = list(set(chunks))

    for chunk in unique_chunks:
        chunk_words = set(chunk.lower().split())
        score = len(query_words.intersection(chunk_words))
        
        # Simple boost for chunks that contain the full query phrase
        if query.lower() in chunk.lower():
            score += 5 

        if score > max_score:
            max_score = score
            best_chunk = chunk
            
    return best_chunk

async def get_context_for_questions(hyde_embeddings: List[List[float]], questions: List[str], document_id: str, all_chunks: List[str]) -> List[str]:
    """
    Implements Multi-way Recall and Re-ranking to find the best context for each question.
    """
    final_contexts = []
    for i, (emb, question) in enumerate(zip(hyde_embeddings, questions)):
        print(f"  > [Q{i+1}] Performing multi-way recall...")
        # 1. Multi-way Recall
        vector_results_task = pinecone_service.query_vector(emb, top_k=5, namespace=document_id)
        keyword_results = keyword_search(question, all_chunks, top_k=5)
        
        vector_results = await vector_results_task
        
        # Combine results
        combined_chunks = vector_results + keyword_results
        print(f"  > [Q{i+1}] Found {len(vector_results)} vector results and {len(keyword_results)} keyword results.")

        # 2. Re-rank
        print(f"  > [Q{i+1}] Re-ranking {len(set(combined_chunks))} unique chunks...")
        best_chunk = rerank_chunks(question, combined_chunks)
        final_contexts.append(best_chunk)
        print(f"  > [Q{i+1}] Top ranked chunk selected.")

    return final_contexts

# --- FastAPI Application Setup ---

app = FastAPI(title="LLM Query Retrieval System", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Initialize services
gemini_service = GeminiService(api_key=settings.google_api_key)
pinecone_service = PineconeService(api_key=settings.pinecone_api_key, environment=settings.pinecone_environment, index_name=settings.pinecone_index_name)
postgres_service = PostgresService()
redis_client = redis.from_url(settings.redis_url, decode_responses=True)

async def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authentication scheme.")
    if authorization.split(" ")[1] != settings.auth_token:
        raise HTTPException(status_code=403, detail="Invalid token.")

# --- API Endpoint for Submission ---

@app.post("/api/v1/hackrx/run", response_model=HackRxResponse, dependencies=[Depends(verify_token)])
async def hackrx_run(request: HackRxRequest):
    full_request_start_time = time.time()
    document_id = hashlib.sha256(request.documents.encode()).hexdigest()

    # --- Step 1: Check Cache and Perform Ingestion if Necessary ---
    # NEW: We now cache the full text chunks in Redis as well.
    cached_chunks_json = await redis_client.get(f"{document_id}_chunks")
    
    if cached_chunks_json:
        print(f"CACHE HIT for document_id: {document_id}. Skipping ingestion.")
        text_chunks = json.loads(cached_chunks_json)
    else:
        print(f"CACHE MISS for document_id: {document_id}. Starting ingestion.")
        try:
            ingestion_start = time.time()
            document_text = await download_and_parse_pdf(request.documents)
            if not document_text.strip():
                raise HTTPException(status_code=500, detail="Extracted text from document is empty.")
            
            text_chunks = sentence_aware_splitter(document_text)
            if not text_chunks:
                raise HTTPException(status_code=500, detail="Could not extract text chunks.")
                
            chunk_embeddings = await gemini_service.get_embeddings(text_chunks, task_type="retrieval_document")
            await pinecone_service.upsert_documents(text_chunks, chunk_embeddings, namespace=document_id)
            
            # Cache the text chunks for keyword search and re-ranking
            await redis_client.set(f"{document_id}_chunks", json.dumps(text_chunks), ex=3600)
            
            ingestion_time = time.time() - ingestion_start
            print(f"Ingestion completed in {ingestion_time:.2f} seconds.")
        except Exception as e:
            print(f"Error during ingestion phase: {e}")
            raise HTTPException(status_code=500, detail=f"Failed during document ingestion: {e}")

    # --- Step 2: Batch-Optimized Question Answering with Advanced RAG ---
    answering_start = time.time()
    try:
        hyde_start = time.time()
        hypothetical_answers = await gemini_service.generate_batch_hypothetical_answers(request.questions)
        print(f" > HyDE generation took: {time.time() - hyde_start:.2f}s")

        embedding_start = time.time()
        hyde_embeddings = await gemini_service.get_embeddings(hypothetical_answers, task_type="retrieval_query")
        print(f" > HyDE embedding took: {time.time() - embedding_start:.2f}s")

        retrieval_start = time.time()
        # NEW: Call the advanced context retrieval function
        contexts = await get_context_for_questions(hyde_embeddings, request.questions, document_id, text_chunks)
        print(f" > Context retrieval & re-ranking took: {time.time() - retrieval_start:.2f}s")

        generation_start = time.time()
        answers = await gemini_service.generate_batch_final_answers(contexts, request.questions)
        print(f" > Final answer generation took: {time.time() - generation_start:.2f}s")

    except Exception as e:
        print(f"Error during answering phase: {e}")
        raise HTTPException(status_code=500, detail=f"Failed during question answering: {e}")
    
    answering_time = time.time() - answering_start
    print(f"Answering stage completed in {answering_time:.2f} seconds.")
    
    # --- Step 3: Logging and Response ---
    await postgres_service.log_interaction(request.documents, request.questions, answers)
    
    total_time = time.time() - full_request_start_time
    print(f"Full request completed in {total_time:.2f} seconds.")

    if total_time > 30:
        print(f"WARNING: Request took {total_time:.2f} seconds, which exceeds the 30s timeout.")

    return HackRxResponse(answers=answers)

@app.get("/health", status_code=200)
def health_check():
    return {"status": "ok"}
