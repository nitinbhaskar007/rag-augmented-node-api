```md
# RAG + Hybrid Search API (Node.js + OpenAI + LanceDB)

This project is a **backend RAG (Retrieval Augmented Generation) API** that answers user questions using your own documents.

It has **two pipelines**:

1. **Indexing pipeline** (offline / when docs change)
2. **Query pipeline** (online / per API request)

---

## What problems this solves

- Keeps your **OpenAI API key safe** (server-side only)
- Lets your **frontend call your own API** (`POST /ask`)
- Uses:
  - **OpenAI Embeddings** to represent text as vectors
  - **LanceDB** as a real **VectorDB**
  - **Hybrid retrieval**: _semantic vector search + keyword BM25_
  - **RRF fusion** (rank fusion) to merge vector + keyword results
  - **Augmented RAG**: multi-query rewrites + HyDE to improve recall
  - **Caching** to reduce cost and speed up repeat queries

---

## Folder structure
```

data/ # raw documents (.md/.txt)
src/
lib.js # utilities, recursive chunking, vector math, OpenAI client
loadDocs.js # load + chunk documents from data/
prompts.js # prompt templates (answer, multi-query, hyde)
embed.js # embeddings API wrapper
vectorStore.js # LanceDB wrapper (vector + FTS + hybrid + RRF)
indexer.js # indexing pipeline (docs -> vectors -> LanceDB)
index.js # CLI entry to buildIndex()
rag/engine.js # query pipeline (augment -> embed -> retrieve -> answer)
server.js # Fastify REST API
.cache/ # runtime caches (embeddings/augment/answers)
.lancedb/ # LanceDB database files (VectorDB storage)

````

---

## Environment variables

Create `.env` from `.env.example`:

- `OPENAI_API_KEY` (required)
- `PORT` (default 3001)
- `CORS_ORIGIN` (frontend origin)
- `RAG_API_KEY` (optional: protects /ask and /reindex)
- `RAG_GEN_MODEL` (LLM for generation/augmentation)
- `RAG_EMBED_MODEL` (embedding model)
- `LANCEDB_URI` (default `./.lancedb`)
- `LANCEDB_TABLE` (default `rag_chunks`)
- Hybrid settings:
  - `RAG_HYBRID=true`
  - `RAG_RRF_K=60`

---

## Indexing pipeline (Load → Chunk → Embed → Store)

You run this when documents change:

```bash
npm run index
````

### Indexing steps

1. **Load docs**

   - `src/loadDocs.js` reads `data/**/*.md|txt`

2. **Chunk docs (Recursive Text Splitting)**

   - `src/lib.js` `chunkTextRecursive()`
   - Separators priority:

     - `\n\n` (paragraphs)
     - `\n` (lines)
     - `" "` (spaces)
     - `""` (characters fallback)

   - Final chunks are merged with:

     - `chunkSize`
     - `chunkOverlap`

3. **Embed chunks**

   - `src/embed.js` calls OpenAI embeddings
   - Embeddings are normalized to **unit vectors**

4. **Store in VectorDB**

   - `src/vectorStore.js` writes rows to LanceDB:

     ```json
     { "id": "...", "source": "...", "chunkIndex": 0, "content": "...", "vector": [..] }
     ```

5. **Build indexes**

   - Vector ANN index (fast nearest-neighbor)
   - FTS (BM25) index for keyword search

### Indexing diagram

```
data/*.md/.txt
   ↓ loadDocs.js
chunks[{id, source, chunkIndex, content}]
   ↓ embed.js (OpenAI embeddings)
chunks + vectors
   ↓ vectorStore.js (LanceDB)
LanceDB table: rag_chunks
   + Vector index
   + FTS/BM25 index
```

---

## Query pipeline (API request: POST /ask)

Frontend sends:

```json
{
  "question": "What is the refund policy?",
  "filters": { "sources": ["data/policies.md"] },
  "mustInclude": ["refund", "partial"],
  "mustIncludeMode": "all"
}
```

Backend returns:

```json
{
  "answer": "...",
  "sources": ["data/policies.md#2", "data/faq.txt#0"],
  "debug": { ... } // only when RAG_DEBUG=true
}
```

### Query steps (per request)

1. **Augment (optional, improves recall)**

   - Multi-query rewrite: creates 3 short search queries
   - HyDE: creates a short hypothetical answer used for retrieval embedding

2. **Embed question variants**

   - Variant texts: `[question, rewrite1, rewrite2, rewrite3, hyde]`
   - Cached in `.cache/embeddings.json` by `(model + text hash)`

3. **Retrieve from LanceDB**

   - **Vector search**: semantic similarity
   - **FTS/BM25 search**: keyword matching
   - **Hybrid**: merges vector + BM25 results using **RRF (Reciprocal Rank Fusion)**

4. **Merge across variants**

   - Runs hybrid retrieval for each variant and merges best hits

5. **Apply enterprise constraints**

   - `filters`:

     - allow only certain files (`sources`)
     - or allow prefix (`sourcePrefix`)

   - `mustInclude`:

     - enforce that retrieved lookups contain required keywords
     - mode:

       - `all` (default): must contain all keywords
       - `any`: must contain at least one keyword

6. **Diversity selection (MMR-ish)**

   - Avoids sending 6 near-duplicate chunks to the LLM
   - Improves context coverage and reduces token waste

7. **Answer using context only**

   - Builds context block:

     ```
     [source: data/policies.md#2]
     chunk text...

     ---
     [source: data/faq.txt#0]
     chunk text...
     ```

   - Sends to LLM with strict instructions:

     - Use ONLY provided context
     - If not found: say "I don't know from the provided documents."

8. **Cache writes**

   - Embeddings cache
   - Augment cache (rewrites/HyDE)
   - Answer cache (question + context hash)

### Query diagram

```
POST /ask (question)
   ↓ engine.js
[optional] Multi-query + HyDE
   ↓ embed variants (cached)
Vectors + query texts
   ↓ vectorStore.js
Vector Search + BM25 Search
   ↓ RRF fusion (hybrid)
Merged candidates
   ↓ filters + mustInclude
Filtered candidates
   ↓ diversity selection
Top context chunks
   ↓ LLM answer (context-only)
Response { answer, sources }
```

---

## Caching strategy

Caches live in `.cache/`:

- `embeddings.json`

  - key: `emb:<model>:<hash(text)>`
  - value: normalized embedding vector (float array)

- `augment.json`

  - key: `mq:<model>:<hash(question)>` → rewrites
  - key: `hyde:<model>:<hash(question)>` → hyde text

- `answers.json`

  - key: `ans:<model>:<hash(question)>:<hash(context)>` → final answer

Why cache helps:

- Repeat questions become fast + cheaper
- Multi-query and HyDE are reused instead of re-generated

---

## Hybrid Retrieval (Vector + BM25) + RRF

- **Vector search** finds semantically similar chunks (meaning)
- **BM25/FTS** finds exact keyword matches (terms)

RRF merges both ranked lists:

For each list:

- Rank starts at 1
- Add: `1 / (K + rank)` to the document’s fusion score

Where:

- `K` (commonly 60) controls how strongly top ranks dominate

Benefits:

- Great for real enterprise docs where some queries require exact words (BM25) and some require meaning (vectors)

---

## REST API

### `GET /health`

Returns `{ ok: true }`

### `POST /ask`

Body:

```json
{
  "question": "string",
  "filters": { "sources": ["..."], "sourcePrefix": "..." },
  "mustInclude": ["keyword1", "keyword2"],
  "mustIncludeMode": "all" | "any"
}
```

### `POST /reindex` (optional)

Rebuilds LanceDB table from `data/` and reloads engine.

**Protect these endpoints** using `RAG_API_KEY`:

- Send header: `x-api-key: <your key>`

---

## Security notes (important)

- Do NOT call OpenAI from frontend (never expose `OPENAI_API_KEY`)
- `RAG_API_KEY` is your backend-only shared secret (ok for internal apps)
- For public apps:

  - don’t rely on static API keys in client bundles
  - use user auth (JWT/session) and server-side authorization

---

## How to run

1. Install

```bash
npm install
```

2. Put docs in `data/`

3. Build index (writes to LanceDB)

```bash
npm run index
```

4. Start API

```bash
npm run serve
```

5. Test

```bash
curl -X POST http://localhost:3001/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What is our refund policy?\"}"
```

---

## Deployment tips (simple)

- Keep these folders persistent:

  - `./.lancedb` (vector DB storage)
  - `./.cache` (optional but helpful)

- Typical hosting:

  - VM (EC2 / DigitalOcean) → best for persistent disk
  - Container → mount volumes for `.lancedb` and `.cache`

- Add:

  - reverse proxy (Nginx) in front
  - HTTPS
  - stronger auth if public

---

## Interview-ready explanation (30 seconds)

“I built a Node.js RAG API. I ingest local docs, recursively chunk with overlap, embed chunks using OpenAI, and store them in LanceDB as a vector database with both vector and BM25 indexes. At query time, I augment the user question with rewrites and HyDE, embed variants, run hybrid retrieval (vector + BM25) fused by RRF, enforce filters and keyword constraints, diversify chunks using an MMR-like strategy, and generate a context-grounded answer with citations. Everything is exposed via Fastify so the frontend never sees the OpenAI key.”
