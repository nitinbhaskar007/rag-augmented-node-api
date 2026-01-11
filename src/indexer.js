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

import { loadAndChunkDocs } from "./loadDocs.js";
import { LocalVectorStore } from "./vectorStore.js";
import { embedTexts } from "./embed.js";
import fs from "node:fs/promises";

/**
 * Build index from documents and save to index/store.json.
 * Exported so API can call it on /reindex.
 */
export async function buildIndex({
  dataDir = "data",
  exts = ["txt", "md"],
  chunk = { maxChars: 1200, overlapChars: 200 },
  indexPath = "index/store.json",
  embedModel = process.env.RAG_EMBED_MODEL || "text-embedding-3-small",
  batchSize = 64,
  logger = console,
} = {}) {
  logger.info?.("Indexing: loading + chunking docs...");
  const chunks = await loadAndChunkDocs({ dataDir, exts, chunk });
  logger.info?.(`Indexing: chunks = ${chunks.length}`);

  const store = new LocalVectorStore([]);

  for (let i = 0; i < chunks.length; i += batchSize) {
    const batch = chunks.slice(i, i + batchSize);
    const vectors = await embedTexts(
      batch.map((c) => c.content),
      { model: embedModel }
    );

    batch.forEach((c, idx) => {
      store.add({
        id: c.id,
        source: c.source,
        chunkIndex: c.chunkIndex,
        content: c.content,
        embeddingUnit: vectors[idx],
      });
    });

    logger.info?.(
      `Indexing: embedded ${Math.min(i + batchSize, chunks.length)}/${
        chunks.length
      }`
    );
  }

  await fs.mkdir("index", { recursive: true });
  await store.save(indexPath);
  logger.info?.(`Indexing: ✅ saved ${indexPath}`);

  return { store, chunksCount: chunks.length };
}
