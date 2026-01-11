// What it does

// This is your “ETL pipeline”:

// load+chunk docs

// batch embed chunks (e.g., 64 at a time)

// save store to disk

// Why batching matters:

// API efficiency

// less overhead

// Why this file exists:

// reused by /reindex endpoint too

// This is your offline ingestion pipeline.

// What it does

// Calls loadAndChunkDocs() → gets chunk objects

// Batches chunks (e.g., 64 at a time)

// Calls embedTexts() → gets vectors

// Creates LanceDB records:

// { id, source, chunkIndex, content, vector }

// Writes them into LanceDB table:

// first batch → overwrite (fresh table)

// next batches → add

// Builds indexes (ensureIndexes())

// Why it matters

// You only run this when docs change.
// It’s the “Load → Chunk → Embed → Store” flow in your slide.

import { loadAndChunkDocs } from "./loadDocs.js";
import { embedTexts } from "./embed.js";
import { LanceVectorStore } from "./vectorStore.js";

export async function buildIndex({
  dataDir = "data",
  exts = ["txt", "md"],
  chunk = { chunkSize: 1200, chunkOverlap: 200 },
  batchSize = 64,
  logger = console,
  lanceUri = process.env.LANCEDB_URI || "./.lancedb",
  tableName = process.env.LANCEDB_TABLE || "rag_chunks",
  embedModel = process.env.RAG_EMBED_MODEL || "text-embedding-3-small",
  vectorColumn = process.env.RAG_VECTOR_COLUMN || "vector",
  ftsColumn = process.env.RAG_FTS_COLUMN || "content",
} = {}) {
  logger.info?.("Indexing: loading + chunking docs...");
  const chunks = await loadAndChunkDocs({ dataDir, exts, chunk });
  logger.info?.(`Indexing: chunks = ${chunks.length}`);

  const store = await LanceVectorStore.init({
    uri: lanceUri,
    tableName,
    vectorColumn,
    ftsColumn,
  });

  let created = false;

  for (let i = 0; i < chunks.length; i += batchSize) {
    const batch = chunks.slice(i, i + batchSize);
    const vectors = await embedTexts(
      batch.map((c) => c.content),
      { model: embedModel }
    );

    const records = batch.map((c, idx) => ({
      id: c.id,
      source: c.source,
      chunkIndex: c.chunkIndex,
      content: c.content,
      [vectorColumn]: vectors[idx],
    }));

    if (!created) {
      logger.info?.("Indexing: creating table (overwrite)...");
      await store.overwrite(records);
      created = true;
    } else {
      await store.add(records);
    }

    logger.info?.(
      `Indexing: embedded ${Math.min(i + batchSize, chunks.length)}/${
        chunks.length
      }`
    );
  }

  // ✅ Build indexes: vector index + FTS(BM25) index
  logger.info?.("Indexing: building vector + FTS indexes...");
  await store.ensureIndexes();

  logger.info?.(
    `Indexing: ✅ done. LanceDB uri=${lanceUri} table=${tableName}`
  );
  return { chunksCount: chunks.length };
}
