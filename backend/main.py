# main.py
# Entry point for the FastAPI-based RAG system with an interactive workflow.

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

    async def rerank_chunks(self, question: str, chunks: List[str]) -> List[str]:
        """Uses the LLM to re-rank retrieved chunks for relevance."""
        if not chunks:
            return []
        print(f"Re-ranking {len(chunks)} chunks for question: '{question}'")
        
        chunk_text = ""
        for i, chunk in enumerate(chunks):
            chunk_text += f"[{i+1}] {chunk}\n\n"

        prompt = f"""
        You are a relevance ranking expert. Your task is to evaluate a list of document chunks based on their relevance to a user's question.

        User Question: "{question}"

        Document Chunks:
        {chunk_text}

        Instructions:
        Identify the top 5 most relevant document chunks that directly help answer the user question. Return your answer as a JSON list of the integer IDs of the most relevant chunks.
        For example:
        [1, 5, 3, 8, 2]
        """
        try:
            response = self.generation_model.generate_content(prompt)
            cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
            relevant_ids = json.loads(cleaned_response)
            
            reranked_chunks = [chunks[i-1] for i in relevant_ids if 0 < i <= len(chunks)]
            print(f"Re-ranked and selected {len(reranked_chunks)} chunks.")
            return reranked_chunks
        except Exception as e:
            print(f"Could not re-rank chunks, falling back to original chunks. Error: {e}")
            return chunks[:5]

    async def generate_final_answer(self, context: str, question: str) -> str:
        print(f"Generating final answer for question: '{question}'")
        prompt = f"CONTEXT:\n---\n{context}\n---\n\nBased exclusively on the CONTEXT provided, answer the following USER QUESTION: {question}. If the answer is not in the context, state 'Cannot answer based on the provided information.'"
        try:
            response = self.generation_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate an answer: {e}")


class PineconeService:
    """Handles all interactions with the Pinecone vector database."""
    def __init__(self, api_key: str, environment: str, index_name: str):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        if self.index_name not in self.pc.list_indexes().names():
            print(f"Creating index '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name, dimension=768, metric='cosine',
                spec=ServerlessSpec(cloud='aws', region=environment)
            )
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

    async def query(self, query_embedding: List[float], top_k: int, namespace: str) -> List[str]:
        print(f"Querying Pinecone namespace '{namespace}'...")
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
            response = await client.get(url, follow_redirects=True, timeout=15.0)
            response.raise_for_status()
            pdf_file = io.BytesIO(response.content)
            reader = pypdf.PdfReader(pdf_file)
            text = " ".join(page.extract_text() or "" for page in reader.pages)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {e}")

def sentence_aware_splitter(text: str, chunk_size: int = 1500) -> List[str]:
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

# --- Core RAG Logic for a Single Question ---
async def answer_one_question(question: str, document_id: str) -> str:
    """Encapsulates the logic for answering a single question with re-ranking."""
    # Step 1: Broad Retrieval
    question_embedding_list = await gemini_service.get_embeddings([question], task_type="retrieval_query")
    question_embedding = question_embedding_list[0]
    candidate_chunks = await pinecone_service.query(question_embedding, top_k=20, namespace=document_id)
    
    if not candidate_chunks:
        return "Could not find any potentially relevant information in the document."

    # Step 2: Re-ranking for Precision
    reranked_chunks = await gemini_service.rerank_chunks(question, candidate_chunks)
    
    if not reranked_chunks:
        return "Information was found, but none of it was relevant enough to form an answer."

    # Step 3: Generation from the Best Context
    context = "\n\n---\n\n".join(reranked_chunks)
    return await gemini_service.generate_final_answer(context, question)

# --- FastAPI Application Setup ---

app = FastAPI(title="LLM Query Retrieval System", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

gemini_service = GeminiService(api_key=settings.google_api_key)
pinecone_service = PineconeService(api_key=settings.pinecone_api_key, environment=settings.pinecone_environment, index_name=settings.pinecone_index_name)
postgres_service = PostgresService()

async def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authentication scheme.")
    if authorization.split(" ")[1] != settings.auth_token:
        raise HTTPException(status_code=403, detail="Invalid token.")

# --- API Endpoint for Submission ---

@app.post("/api/v1/hackrx/run", response_model=HackRxResponse, dependencies=[Depends(verify_token)])
async def hackrx_run(request: HackRxRequest):
    """
    Handles the entire RAG pipeline in a single, performance-optimized request.
    """
    start_time = time.time()

    # --- Step 1: Document Ingestion ---
    try:
        ingestion_start = time.time()
        document_text = await download_and_parse_pdf(request.documents)
        if not document_text.strip():
            raise HTTPException(status_code=500, detail="Extracted text from document is empty.")
        
        document_id = hashlib.sha256(request.documents.encode()).hexdigest()
        text_chunks = sentence_aware_splitter(document_text)
        print(f"Created {len(text_chunks)} chunks.")
        
        if not text_chunks:
            raise HTTPException(status_code=500, detail="Could not extract text chunks from the document.")
            
        chunk_embeddings = await gemini_service.get_embeddings(text_chunks, task_type="retrieval_document")
        await pinecone_service.upsert_documents(text_chunks, chunk_embeddings, namespace=document_id)
        ingestion_time = time.time() - ingestion_start
        print(f"Ingestion completed in {ingestion_time:.2f} seconds.")
    except Exception as e:
        print(f"Error during ingestion phase: {e}")
        raise HTTPException(status_code=500, detail=f"Failed during document ingestion: {e}")

    # --- Step 2: Concurrent Question Answering ---
    answering_start = time.time()
    try:
        tasks = [answer_one_question(q, document_id) for q in request.questions]
        answers = await asyncio.gather(*tasks)
    except Exception as e:
        print(f"Error during answering phase: {e}")
        raise HTTPException(status_code=500, detail=f"Failed during question answering: {e}")
    
    answering_time = time.time() - answering_start
    print(f"Answering completed in {answering_time:.2f} seconds.")
    
    # --- Step 3: Logging and Response ---
    await postgres_service.log_interaction(request.documents, request.questions, answers)
    
    total_time = time.time() - start_time
    print(f"Full request completed in {total_time:.2f} seconds.")

    if total_time > 30:
        print(f"WARNING: Request took {total_time:.2f} seconds, which exceeds the 30s timeout.")

    return HackRxResponse(answers=answers)

@app.get("/health", status_code=200)
def health_check():
    return {"status": "ok"}
