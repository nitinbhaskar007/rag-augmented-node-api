// What it stores

// Each item includes:

// chunk content

// metadata

// embedding vector (unit normalized)

// Key functions
// search(queryEmbeddingUnit, topK)

// computes cosine similarity between query vector and each stored chunk vector

// returns topK matches

// Cosine similarity:

// Since both vectors are normalized, cosine = dot product

// Very fast and simple

// searchMulti(queryEmbeddingUnits, perQueryTopK, finalTopK)

// This is crucial for Augmented RAG.

// You have multiple query embeddings:

// original question embedding

// multi-query embeddings

// HyDE embedding

// For each query, get top results

// Merge results using Map() so duplicates are removed

// Keep best score per chunk

// Return final top list

// Interview line:
// // “We do union retrieval across multiple query embeddings to improve recall and cover synonyms and alternate phrasing.”

// This is the “Vector Store” in your diagram, but implemented as a clean wrapper.

// What it does
// A) Connect to LanceDB

// connect(uri) opens your local vector DB at ./.lancedb

// Opens/creates table like rag_chunks

// B) Write data

// overwrite(records) drops + recreates the table (clean rebuild)

// add(records) appends new rows

// Each row stored has:

// {
//   id, source, chunkIndex, content, vector
// }

// C) Indexes

// ensureIndexes() creates:

// a vector ANN index (IVF_PQ)

// a BM25 full-text index (FTS)

// D) Retrieval methods

// vectorSearch(queryVector)

// semantic similarity search

// ftsSearch(queryText)

// keyword/BM25 search (exact words)

// hybridSearch(queryVector, queryText)

// runs both and merges with RRF

// hybridSearchMulti(variantEmbeds, variantTexts)

// performs hybrid search for each query variant

// merges results across variants

// E) Ranking fusion (RRF)

// rrfFuse() merges the ranking lists:

// vector ranking list

// bm25 ranking list

// This is very commonly used in hybrid retrieval.

// Why it matters

// This file is what makes your project “real vector DB” instead of JSON.
import * as lancedb from "@lancedb/lancedb";

/**
 * LanceVectorStore
 * - Uses LanceDB as real VectorDB
 * - Supports:
 *   1) Vector search
 *   2) Full-text search (BM25) via FTS index
 *   3) Hybrid search: Vector + FTS merged using RRF (in our code)
 *
 * Table schema we store:
 * { id, source, chunkIndex, content, vector }
 */
export class LanceVectorStore {
  constructor({ uri, tableName, conn, table, vectorColumn, ftsColumn }) {
    this.uri = uri;
    this.tableName = tableName;
    this.conn = conn;
    this.table = table;
    this.vectorColumn = vectorColumn;
    this.ftsColumn = ftsColumn;
  }

  static async init({
    uri,
    tableName,
    vectorColumn = "vector",
    ftsColumn = "content",
  }) {
    const conn = await lancedb.connect(uri);
    let table = null;

    try {
      table = await conn.openTable(tableName);
    } catch {
      table = null;
    }

    return new LanceVectorStore({
      uri,
      tableName,
      conn,
      table,
      vectorColumn,
      ftsColumn,
    });
  }

  async overwrite(records) {
    try {
      await this.conn.dropTable(this.tableName);
    } catch {}

    // Create new table
    this.table = await this.conn.createTable(this.tableName, records, {
      mode: "overwrite",
    });
  }

  async add(records) {
    if (!this.table) {
      this.table = await this.conn.createTable(this.tableName, records, {
        mode: "create",
      });
      return;
    }
    await this.table.add(records);
  }

  async reload() {
    this.table = await this.conn.openTable(this.tableName);
  }

  /**
   * Ensure indexes exist:
   * - Vector index on vector column (ANN)
   * - FTS index (BM25) on content column
   *
   * We use createIndex + Index.fts() from LanceDB JS SDK. :contentReference[oaicite:4]{index=4}
   */
  async ensureIndexes({
    vectorIndex = {
      numPartitions: 128,
      numSubVectors: 16,
      distanceType: "cosine",
    },
    ftsIndex = {}, // add tokenizer options if needed
  } = {}) {
    if (!this.table)
      throw new Error(`LanceDB table not found: ${this.tableName}`);

    // Vector index
    try {
      await this.table.createIndex(this.vectorColumn, {
        config: lancedb.Index.ivfPq({
          numPartitions: vectorIndex.numPartitions,
          numSubVectors: vectorIndex.numSubVectors,
          distanceType: vectorIndex.distanceType,
        }),
      });
      // index name is `${column}_idx` in JS createIndex docs :contentReference[oaicite:5]{index=5}
      await this.table.waitForIndex(`${this.vectorColumn}_idx`);
    } catch {
      // ignore if already exists / cannot build right now
    }

    // FTS index (BM25)
    try {
      await this.table.createIndex(this.ftsColumn, {
        config: lancedb.Index.fts(ftsIndex),
      });
      await this.table.waitForIndex(`${this.ftsColumn}_idx`);
    } catch {
      // ignore if already exists / cannot build right now
    }
  }

  /* -------------------- Low-level searches -------------------- */

  async vectorSearch(queryVector, { topK = 8 } = {}) {
    if (!this.table)
      throw new Error(`LanceDB table not found: ${this.tableName}`);

    const rows = await this.table
      .vectorSearch(queryVector) // vectorSearch exists :contentReference[oaicite:6]{index=6}
      .column(this.vectorColumn)
      .distanceType("cosine")
      .limit(topK)
      .select(["id", "source", "chunkIndex", "content", this.vectorColumn])
      .toArray();

    // Cosine distance: smaller is better; we convert to similarity-ish score
    return rows.map((r) => ({
      item: {
        id: r.id,
        source: r.source,
        chunkIndex: r.chunkIndex,
        content: r.content,
        embeddingUnit: r[this.vectorColumn],
      },
      // if cosine distance = 1 - cosineSim, then sim = 1 - distance
      score: 1 - (r._distance ?? 1),
      _rankSource: "vector",
    }));
  }

  async ftsSearch(queryText, { topK = 8 } = {}) {
    if (!this.table)
      throw new Error(`LanceDB table not found: ${this.tableName}`);

    // Table.search(query, queryType?, ftsColumns?) supports queryType "fts" and ftsColumns :contentReference[oaicite:7]{index=7}
    const rows = await this.table
      .search(queryText, "fts", [this.ftsColumn])
      .limit(topK)
      .select(["id", "source", "chunkIndex", "content", this.vectorColumn])
      .toArray();

    // FTS returns _score (BM25 relevance). Bigger is better.
    return rows.map((r) => ({
      item: {
        id: r.id,
        source: r.source,
        chunkIndex: r.chunkIndex,
        content: r.content,
        embeddingUnit: r[this.vectorColumn],
      },
      score: r._score ?? 0,
      _rankSource: "fts",
    }));
  }

  /* -------------------- Hybrid: RRF fusion -------------------- */

  /**
   * RRF fusion:
   * score = Σ 1 / (K + rank)
   * where rank starts at 1.
   *
   * K default=60 is the common near-optimal constant in RRF literature and also
   * matches LanceDB’s RRF defaults conceptually. :contentReference[oaicite:8]{index=8}
   */
  static rrfFuse(vectorHits, ftsHits, { K = 60 } = {}) {
    const out = new Map();

    // 1) rank lists by "best first"
    const v = [...vectorHits].sort((a, b) => b.score - a.score);
    const f = [...ftsHits].sort((a, b) => b.score - a.score);

    const addRankScore = (hits, listName) => {
      hits.forEach((h, idx) => {
        const rank = idx + 1;
        const add = 1 / (K + rank);

        const prev = out.get(h.item.id);
        if (!prev) {
          out.set(h.item.id, {
            item: h.item,
            score: add,
            _rrf: {
              vector: listName === "vector" ? add : 0,
              fts: listName === "fts" ? add : 0,
            },
          });
        } else {
          prev.score += add;
          prev._rrf[listName] += add;
        }
      });
    };

    addRankScore(v, "vector");
    addRankScore(f, "fts");

    return [...out.values()].sort((a, b) => b.score - a.score);
  }

  /**
   * Hybrid search for ONE query variant:
   * - vectorSearch(embedding)
   * - ftsSearch(text)
   * - RRF fuse
   */
  async hybridSearch(queryVector, queryText, { topK = 8, rrfK = 60 } = {}) {
    const [vHits, fHits] = await Promise.all([
      this.vectorSearch(queryVector, { topK }),
      this.ftsSearch(queryText, { topK }),
    ]);

    return LanceVectorStore.rrfFuse(vHits, fHits, { K: rrfK }).slice(0, topK);
  }

  /**
   * Hybrid search for MULTIPLE query variants (multi-query + HyDE):
   * - For each variant i:
   *   - hybridSearch(variantEmbeds[i], variantTexts[i])
   * - Then merge by chunk-id, keeping best score per chunk.
   */
  async hybridSearchMulti(
    variantEmbeds,
    variantTexts,
    { perQueryTopK = 8, finalTopK = 25, rrfK = 60 } = {}
  ) {
    const best = new Map();

    for (let i = 0; i < variantEmbeds.length; i++) {
      const qVec = variantEmbeds[i];
      const qText = variantTexts[i];

      const hits = await this.hybridSearch(qVec, qText, {
        topK: perQueryTopK,
        rrfK,
      });

      for (const h of hits) {
        const prev = best.get(h.item.id);
        if (!prev || h.score > prev.score) best.set(h.item.id, h);
      }
    }

    return [...best.values()]
      .sort((a, b) => b.score - a.score)
      .slice(0, finalTopK);
  }
}
