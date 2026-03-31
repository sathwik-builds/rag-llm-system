# RAG LLM System

A production-style Retrieval-Augmented Generation (RAG) pipeline 
built with LangChain, FAISS, and OpenAI — designed to answer 
questions grounded strictly in provided documents.

## Architecture

User Query → Document Chunking → OpenAI Embeddings → 
FAISS Vector Store → Top-K Retrieval → Prompt Template → 
GPT-4o-mini → Grounded Answer

## Stack

- **LLM:** OpenAI GPT-4o-mini
- **Embeddings:** OpenAI Embeddings
- **Vector Store:** FAISS (local), Pinecone (cloud)
- **Framework:** LangChain
- **Backend:** FastAPI
- **Containerization:** Docker

## Features

- PDF document ingestion and intelligent chunking
- Semantic similarity search with configurable Top-K retrieval
- Strict grounding — model answers ONLY from retrieved context
- Chunk overlap for context continuity across boundaries
- Easily swappable vector store (FAISS → Pinecone)

## Setup
```bash
git clone https://github.com/sathwik-builds/rag-llm-system
cd rag-llm-system
pip install -r requirements.txt
cp .env.example .env  #sk-p....on8956
python rag_pipeline.py
```

## How It Works

1. **Load** — PDF is loaded and parsed page by page
2. **Chunk** — Split into 500-token chunks with 50-token overlap
3. **Embed** — Each chunk converted to a vector via OpenAI Embeddings
4. **Store** — Vectors indexed in FAISS for fast similarity search
5. **Retrieve** — Top-2 most relevant chunks fetched for the query
6. **Generate** — GPT-4o-mini answers using ONLY retrieved context

## Evaluation Approach

Grounding is enforced at the prompt level — the system prompt 
explicitly instructs the model to answer only from retrieved 
context or declared knowledge, and to say "not available" 
otherwise. This eliminates hallucination by design rather than 
measuring it after the fact.

Interaction logs capture every query, retrieved chunks, and 
response with timestamps — providing a manual audit trail for 
quality review.

Formal RAGAS evaluation (faithfulness, context recall, answer 
relevance) is planned as a next iteration.