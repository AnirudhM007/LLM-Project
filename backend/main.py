# main.py
# Entry point for the FastAPI-based RAG system, optimized for submission requirements.

import os
import httpx
import asyncpg
import pymupdf4llm
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
from langchain_text_splitters import MarkdownTextSplitter
from fastembed import TextEmbedding

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
    """Handles all interactions with the Google Gemini API for generation."""
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.generation_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        print("Gemini Service Initialized.")

    async def generate_hypothetical_answer(self, question: str) -> str:
        print(f"Generating hypothetical answer for: '{question}'")
        prompt = f"You are an expert in insurance policies. Based on the user's question, generate a concise, hypothetical answer. This answer will be used to find similar text in a document. Phrase it as a statement. User Question: \"{question}\""
        try:
            response = self.generation_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Could not generate hypothetical answer. Error: {e}")
            return question

    async def generate_final_answer(self, context: str, question: str) -> str:
        print(f"Generating final answer for question: '{question}'")
        prompt = f"CONTEXT:\n---\n{context}\n---\n\nBased exclusively on the CONTEXT provided, answer the following USER QUESTION: {question}. If the answer is not in the context, state 'Cannot answer based on the provided information.'"
        try:
            response = self.generation_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate an answer: {e}")

class EmbeddingService:
    """Handles high-speed embedding generation using FastEmbed."""
    def __init__(self):
        # BAAI/bge-small-en-v1.5 is a fast and efficient model.
        self.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        print("FastEmbed Service Initialized.")

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts: return
        print(f"Generating embeddings for {len(texts)} chunks...")
        # FastEmbed is synchronous but highly optimized for CPU batch processing.
        # We run it in a thread to avoid blocking the asyncio event loop.
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None, # Use the default ThreadPoolExecutor
            lambda: list(self.embedding_model.embed(texts))
        )
        return embeddings

class PineconeService:
    """Handles all interactions with the Pinecone vector database."""
    def __init__(self, api_key: str, environment: str, index_name: str):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        if self.index_name not in self.pc.list_indexes().names():
            print(f"Creating index '{self.index_name}'...")
            # Dimension for BAAI/bge-small-en-v1.5 is 384
            self.pc.create_index(
                name=self.index_name, dimension=384, metric='cosine',
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
        if self._pool is None:
            print("Initializing PostgreSQL connection pool...")
            try:
                self._pool = await asyncpg.create_pool(host=settings.db_host, database=settings.db_name, user=settings.db_user, password=settings.db_password)
            except Exception as e:
                print(f"Failed to connect to or initialize PostgreSQL: {e}")
                raise
        return self._pool

    async def log_interaction(self, document_url: str, questions: List[str], answers: List[str]):
        print("Logging interaction to PostgreSQL...")
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute("CREATE TABLE IF NOT EXISTS hackrx_run_logs (id SERIAL PRIMARY KEY, document_url TEXT, questions TEXT, answers TEXT, created_at TIMESTAMPTZ DEFAULT NOW());")
            await conn.execute("INSERT INTO hackrx_run_logs (document_url, questions, answers) VALUES ($1, $2, $3)", document_url, str(questions), str(answers))

# --- Document Processing Utilities ---

async def download_and_parse_pdf_to_markdown(url: str) -> str:
    print(f"Downloading and parsing document from: {url}")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, follow_redirects=True, timeout=12.0)
            response.raise_for_status()
            pdf_bytes = response.content
            # pymupdf4llm is synchronous, so run it in a thread
            loop = asyncio.get_running_loop()
            markdown_text = await loop.run_in_executor(
                None,
                pymupdf4llm.to_markdown,
                pdf_bytes
            )
            return markdown_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF to markdown: {e}")

# --- Core RAG Logic for a Single Question ---
async def answer_one_question(question: str, document_id: str) -> str:
    """Encapsulates the HyDE logic for answering a single question."""
    hypothetical_answer = await gemini_service.generate_hypothetical_answer(question)
    hyde_embedding_list = await embedding_service.get_embeddings([hypothetical_answer])
    hyde_embedding = hyde_embedding_list
    retrieved_chunks = await pinecone_service.query(hyde_embedding, top_k=7, namespace=document_id)
    if not retrieved_chunks:
        return "Could not find relevant information in the document to answer this question."
    context = "\n\n---\n\n".join(retrieved_chunks)
    return await gemini_service.generate_final_answer(context, question)

# --- FastAPI Application Setup ---

app = FastAPI(title="LLM Query Retrieval System", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

gemini_service = GeminiService(api_key=settings.google_api_key)
embedding_service = EmbeddingService()
pinecone_service = PineconeService(api_key=settings.pinecone_api_key, environment=settings.pinecone_environment, index_name=settings.pinecone_index_name)
postgres_service = PostgresService()

async def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authentication scheme.")
    if authorization.split(" ")![1]= settings.auth_token:
        raise HTTPException(status_code=403, detail="Invalid token.")

# --- API Endpoint for Submission ---

@app.post("/api/v1/hackrx/run", response_model=HackRxResponse, dependencies=)
async def hackrx_run(request: HackRxRequest):
    """
    Handles the entire RAG pipeline in a single, performance-optimized request.
    """
    start_time = time.time()

    # --- Step 1: Document Ingestion ---
    try:
        ingestion_start = time.time()
        markdown_text = await download_and_parse_pdf_to_markdown(request.documents)
        if not markdown_text.strip():
            raise HTTPException(status_code=500, detail="Extracted text from document is empty.")
        
        document_id = hashlib.sha256(request.documents.encode()).hexdigest()
        
        splitter = MarkdownTextSplitter(chunk_size=2000, chunk_overlap=100)
        text_chunks = splitter.split_text(markdown_text)
        print(f"OPTIMIZATION: Created {len(text_chunks)} chunks using MarkdownSplitter.")
        
        if not text_chunks:
            raise HTTPException(status_code=500, detail="Could not extract text chunks from the document.")
            
        chunk_embeddings = await embedding_service.get_embeddings(text_chunks)
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
