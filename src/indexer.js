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
import {
  DEFAULT_MANIFEST_PATH,
  loadManifest,
  saveManifest,
  computeChunkMeta,
  diffManifests,
} from "./recordManager.js";

/**
 * buildIndex()
 * -----------
 * mode:
 * - "incremental" (default): only embed/add new chunks, delete removed chunks
 * - "full": rebuild entire DB table from scratch
 */
export async function buildIndex({
  mode = "incremental",
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
  manifestPath = DEFAULT_MANIFEST_PATH,
} = {}) {
  logger.info?.(`Indexing started (mode=${mode})...`);

  // 1) Load + chunk docs
  const chunks = await loadAndChunkDocs({ dataDir, exts, chunk });
  logger.info?.(`Chunks produced: ${chunks.length}`);

  // 2) Compute chunk meta + stable IDs
  const { items, idSet: currentIdsSet } = computeChunkMeta(chunks);

  // 3) Init vector store
  const store = await LanceVectorStore.init({
    uri: lanceUri,
    tableName,
    vectorColumn,
    ftsColumn,
  });

  // FULL rebuild: embed all and overwrite table
  if (mode === "full") {
    logger.info?.("Full rebuild: embedding all chunks...");
    const allRecords = [];

    for (let i = 0; i < items.length; i += batchSize) {
      const batch = items.slice(i, i + batchSize);
      const vectors = await embedTexts(
        batch.map((x) => x.content),
        { model: embedModel }
      );

      const records = batch.map((x, idx) => ({
        id: x.id,
        citationId: x.citationId,
        source: x.source,
        chunkIndex: x.chunkIndex,
        content: x.content,
        contentHash: x.contentHash,
        [vectorColumn]: vectors[idx],
      }));

      allRecords.push(...records);
      logger.info?.(
        `Embedded ${Math.min(i + batchSize, items.length)}/${items.length}`
      );
    }

    await store.overwrite(allRecords);
    await store.ensureIndexes();
    await saveManifest(currentIdsSet, manifestPath);

    logger.info?.("✅ Full rebuild complete.");
    return {
      mode,
      chunksCount: chunks.length,
      added: items.length,
      deleted: 0,
    };
  }

  // INCREMENTAL indexing: use manifest diff
  const previousIdsSet = await loadManifest(manifestPath);
  const { toAdd, toDelete } = diffManifests(previousIdsSet, currentIdsSet);

  logger.info?.(
    `Incremental plan: add=${toAdd.length}, delete=${toDelete.length}`
  );

  // 4) Delete removed chunks from DB
  if (toDelete.length > 0) {
    try {
      await store.deleteByIds(toDelete);
      logger.info?.(`Deleted ${toDelete.length} stale chunks from VectorDB`);
    } catch (e) {
      logger.warn?.(
        "Delete failed (VectorDB version/predicate issue). Consider running mode=full reindex.",
        { message: e?.message }
      );
    }
  }

  // 5) Add new/changed chunks
  const addItems = items.filter((x) => toAdd.includes(x.id));

  let added = 0;
  for (let i = 0; i < addItems.length; i += batchSize) {
    const batch = addItems.slice(i, i + batchSize);
    const vectors = await embedTexts(
      batch.map((x) => x.content),
      { model: embedModel }
    );

    const records = batch.map((x, idx) => ({
      id: x.id,
      citationId: x.citationId,
      source: x.source,
      chunkIndex: x.chunkIndex,
      content: x.content,
      contentHash: x.contentHash,
      [vectorColumn]: vectors[idx],
    }));

    await store.add(records);
    added += records.length;

    logger.info?.(
      `Added ${Math.min(i + batchSize, addItems.length)}/${addItems.length}`
    );
  }

  // 6) Ensure indexes exist (best-effort)
  await store.ensureIndexes();

  // 7) Save manifest
  await saveManifest(currentIdsSet, manifestPath);

  logger.info?.("✅ Incremental indexing complete.");
  return { mode, chunksCount: chunks.length, added, deleted: toDelete.length };
}
