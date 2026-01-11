```md
# RAG + Hybrid Search API (Node.js + OpenAI + LanceDB)

A production-style **Retrieval Augmented Generation (RAG)** backend built with **Node.js**, **OpenAI SDK**, and **LanceDB** (VectorDB).  
It exposes a clean **REST API** (`POST /ask`) that your frontend can call to get answers grounded in your documents.

---

## Table of Contents

- [What this project does](#what-this-project-does)
- [Architecture](#architecture)
  - [Indexing pipeline](#indexing-pipeline)
  - [Query pipeline](#query-pipeline)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Environment variables](#environment-variables)
- [How to run](#how-to-run)
  - [1) Index documents](#1-index-documents)
  - [2) Start the API server](#2-start-the-api-server)
- [API Reference](#api-reference)
  - [GET /health](#get-health)
  - [POST /ask](#post-ask)
  - [POST /reindex](#post-reindex)
- [Hybrid Search explained (Vector + BM25)](#hybrid-search-explained-vector--bm25)
- [Augmented RAG explained (Multi-query + HyDE)](#augmented-rag-explained-multi-query--hyde)
- [Filters and Must-Include Keywords](#filters-and-must-include-keywords)
- [Caching](#caching)
- [Security notes](#security-notes)
- [Troubleshooting](#troubleshooting)
- [Interview-ready summary](#interview-ready-summary)

---

## What this project does

- Ingests documents from `data/`
- Splits them into chunks (recursive splitting + overlap)
- Embeds chunks using OpenAI embeddings
- Stores embeddings in **LanceDB** (VectorDB) with:
  - vector index for semantic search
  - BM25/FTS index for keyword search
- Answers questions via API by retrieving relevant chunks and calling an LLM with **context-only** instructions

---

## Architecture

### Indexing pipeline

Runs when documents change:

1. **Load** raw documents from `data/`
2. **Chunk** using recursive splitting + overlap
3. **Embed** chunks using OpenAI embeddings
4. **Store** into LanceDB table + create indexes

**Flow**
```

data/\* → chunking → embeddings → LanceDB table + indexes

```

### Query pipeline

Runs per API request (`POST /ask`):

1. (Optional) **Augment** the query (Multi-query + HyDE)
2. **Embed** query variants (cached)
3. **Retrieve** from LanceDB using hybrid search:
   - vector semantic search
   - BM25 keyword search
   - fuse with RRF
4. Apply (optional) **filters** and **mustInclude** constraints
5. Select diverse context chunks (MMR-like)
6. **Generate answer** using LLM with context-only prompt
7. Return `{ answer, sources, debug? }`

**Flow**
```

question → augmentation → embeddings → hybrid retrieval → constraints → answer

```

---

## Tech stack

- **Node.js** (ESM)
- **OpenAI SDK**
  - Embeddings: `text-embedding-3-small`
  - Generation: configurable (default: `gpt-4.1-mini`)
- **LanceDB** (VectorDB)
  - Vector ANN index
  - FTS/BM25 index
- **Fastify**
  - CORS
  - rate limiting

---

## Project structure

```

data/ # raw docs (.md/.txt)
src/
lib.js # OpenAI client, recursive chunking, vector math, helpers
loadDocs.js # load + chunk docs into chunk objects
prompts.js # prompts: answer / multi-query / hyde
embed.js # embeddings wrapper (returns unit vectors)
vectorStore.js # LanceDB wrapper (vector + FTS + hybrid + RRF)
indexer.js # indexing pipeline (docs → embeddings → LanceDB)
index.js # CLI: rebuild index
rag/
engine.js # query pipeline (augment → embed → retrieve → answer)
server.js # REST API server (Fastify)
.cache/ # runtime caches (created automatically)
.lancedb/ # LanceDB storage (created automatically)

````

---

## Setup

### Install dependencies
```bash
npm install
````

### Add documents

Put `.md` / `.txt` files into `data/`, e.g.

```
data/policies.md
data/faq.txt
```

### Create `.env`

Copy `.env.example` to `.env` and set required values:

```bash
cp .env.example .env
```

---

## Environment variables

### Required

- `OPENAI_API_KEY`
  Your OpenAI API key (server-side only).

### Server

- `PORT` (default: `3001`)
- `CORS_ORIGIN` (example: `http://localhost:5173`)

### Optional API protection

- `RAG_API_KEY`
  If set, server requires header `x-api-key` for `/ask` and `/reindex`.

### Models

- `RAG_GEN_MODEL` (default: `gpt-4.1-mini`)
- `RAG_EMBED_MODEL` (default: `text-embedding-3-small`)

### VectorDB (LanceDB)

- `LANCEDB_URI` (default: `./.lancedb`)
- `LANCEDB_TABLE` (default: `rag_chunks`)

### Retrieval tuning

- `RAG_PER_QUERY_TOPK` (default: `8`)
- `RAG_FINAL_TOPK` (default: `25`)
- `RAG_CONTEXT_K` (default: `6`)

### Augmentation

- `RAG_MULTI_QUERY=true|false`
- `RAG_HYDE=true|false`

### Hybrid Search

- `RAG_HYBRID=true|false`
- `RAG_RRF_K` (default: `60`)
- `RAG_FTS_COLUMN` (default: `content`)
- `RAG_VECTOR_COLUMN` (default: `vector`)

### Debug

- `RAG_DEBUG=true|false`

---

## How to run

### 1) Index documents

This builds the LanceDB table and indexes:

```bash
npm run index
```

### 2) Start the API server

```bash
npm run serve
```

Server runs on:

```
http://localhost:3001
```

---

## API Reference

### GET /health

**Response**

```json
{ "ok": true }
```

---

### POST /ask

**Request body**

```json
{
  "question": "What is our refund policy?",
  "filters": {
    "sources": ["data/policies.md", "data/faq.txt"],
    "sourcePrefix": "data/"
  },
  "mustInclude": ["refund", "partial"],
  "mustIncludeMode": "all"
}
```

**Fields**

- `question` (required): string
- `filters` (optional):

  - `sources`: allow-list of exact source filenames
  - `sourcePrefix`: allow only sources that start with prefix

- `mustInclude` (optional): array of keywords (or a single space-separated string)
- `mustIncludeMode` (optional): `"all"` (default) or `"any"`

**Response**

```json
{
  "answer": "....",
  "sources": ["data/policies.md#2", "data/faq.txt#0"],
  "debug": {
    "hybrid": true,
    "rrfK": 60,
    "contextChunks": 6
  }
}
```

**cURL example**

```bash
curl -X POST http://localhost:3001/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What is our refund policy?\"}"
```

If `RAG_API_KEY` is set:

```bash
curl -X POST http://localhost:3001/ask \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_SECRET" \
  -d "{\"question\":\"What is our refund policy?\"}"
```

---

### POST /reindex

Rebuilds the index from `data/` and reloads the VectorDB table.

**cURL**

```bash
curl -X POST http://localhost:3001/reindex \
  -H "x-api-key: YOUR_SECRET"
```

---

## Hybrid Search explained (Vector + BM25)

Hybrid search improves retrieval quality by combining:

- **Vector semantic search** (meaning-based)
- **BM25 keyword search** (exact term matching)

We fuse results using **RRF (Reciprocal Rank Fusion)**:

- each list gives a rank to a chunk
- fusion score adds:

  ```
  1 / (K + rank)
  ```

- `K` defaults to `60`

Why it helps:

- vector search handles synonyms and paraphrasing
- BM25 handles exact keywords, IDs, and legal/policy language

---

## Augmented RAG explained (Multi-query + HyDE)

Augmentation improves retrieval recall:

1. **Multi-query rewrites**

- LLM generates 3 alternate search queries

2. **HyDE**

- LLM generates a short hypothetical answer
- embedding this answer often retrieves more relevant passages for abstract questions

These are used only for retrieval; the final answer is still generated from retrieved context.

---

## Filters and Must-Include Keywords

### Filters

Use `filters.sources` to limit retrieval to specific docs:

```json
{ "filters": { "sources": ["data/policies.md"] } }
```

Use `filters.sourcePrefix` to limit by folder/prefix:

```json
{ "filters": { "sourcePrefix": "data/legal/" } }
```

### Must-include keywords

Enforce required terms in retrieved chunks:

```json
{ "mustInclude": ["refund", "partial"], "mustIncludeMode": "all" }
```

- `"all"`: chunk must contain all keywords
- `"any"`: chunk must contain at least one keyword

---

## Caching

Caches are stored in `.cache/`:

- `embeddings.json`
  Caches embeddings for query texts (saves cost).

- `augment.json`
  Caches multi-query rewrites and HyDE output.

- `answers.json`
  Caches final answers keyed by `(question + context hash)`.

Tip: keep `.cache/` persistent in production for best speed/cost.

---

## Security notes

- Never expose `OPENAI_API_KEY` to the frontend.
- `RAG_API_KEY` is OK for internal apps, but for public apps:

  - use proper user authentication (JWT/session)
  - apply authorization and abuse protection

---

## Troubleshooting

### 1) `429 insufficient_quota`

Your API project has no credits / billing not enabled.

- Add credits in OpenAI billing
- Confirm the correct Project/API key is used

### 2) ESM warning / import issues

Ensure `package.json` includes:

```json
"type": "module"
```

### 3) Hybrid search returns weak results

- Re-run `npm run index` to rebuild FTS index
- Check your docs/chunk settings (chunkSize/overlap)

### 4) CORS errors in frontend

Set `CORS_ORIGIN` to your frontend URL, e.g.:

```
CORS_ORIGIN=http://localhost:5173
```

---

## Interview-ready summary

“I built a Node.js RAG API that ingests documents, recursively chunks them with overlap, embeds each chunk using OpenAI embeddings, and stores vectors in LanceDB with both vector and BM25 indexes. At query time, I augment the question with multi-query rewrites and HyDE, embed variants, retrieve using hybrid search fused via RRF, apply optional constraints (source filters and must-include keywords), diversify context chunks with an MMR-like method, and generate a context-grounded answer with citations. The whole pipeline is exposed through a Fastify REST API so the frontend never touches the OpenAI key.”

```
::contentReference[oaicite:0]{index=0}
```
