import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";

/**
 * RecordManager (LangChain-like concept)
 * -------------------------------------
 * Responsibilities:
 * 1) Generate stable IDs for chunks
 * 2) Hash chunk content to detect changes
 * 3) Keep a manifest of indexed chunk IDs (what's already in DB)
 * 4) Produce a diff:
 *    - toAdd: chunks not in manifest (new/changed)
 *    - toDelete: chunks in manifest but not in current docs (removed)
 *
 * Note:
 * - We use content-hash based IDs so a content change becomes:
 *   delete old chunk + add new chunk (simple and robust).
 */

export const DEFAULT_MANIFEST_PATH = path.join(".cache", "record_manager.json");

export function sha256(text) {
  return crypto.createHash("sha256").update(text, "utf-8").digest("hex");
}

/**
 * Stable chunk id:
 * - Includes source + contentHash prefix (keeps it unique across docs)
 */
export function makeChunkId({ source, contentHash }) {
  return `${source}:${contentHash.slice(0, 20)}`;
}

/**
 * Citation id used for display/citations (human-readable):
 * - kept as source#chunkIndex
 */
export function makeCitationId({ source, chunkIndex }) {
  return `${source}#${chunkIndex}`;
}

export async function loadManifest(filePath = DEFAULT_MANIFEST_PATH) {
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    const data = JSON.parse(raw);
    // data = { ids: string[] }
    return new Set(Array.isArray(data.ids) ? data.ids : []);
  } catch {
    return new Set();
  }
}

export async function saveManifest(idsSet, filePath = DEFAULT_MANIFEST_PATH) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(
    filePath,
    JSON.stringify({ ids: [...idsSet] }, null, 2),
    "utf-8"
  );
}

/**
 * Compute current chunk IDs for all chunks.
 * Returns:
 * - items: [{ id, contentHash, citationId, source, chunkIndex, content }]
 * - idSet: Set<string>
 */
export function computeChunkMeta(chunks) {
  const items = chunks.map((c) => {
    const contentHash = sha256(c.content);
    const id = makeChunkId({ source: c.source, contentHash });
    const citationId = makeCitationId({
      source: c.source,
      chunkIndex: c.chunkIndex,
    });

    return {
      id,
      contentHash,
      citationId,
      source: c.source,
      chunkIndex: c.chunkIndex,
      content: c.content,
    };
  });

  const idSet = new Set(items.map((x) => x.id));
  return { items, idSet };
}

/**
 * Diff algorithm:
 * - toAdd: ids present in current but not in previous
 * - toDelete: ids present in previous but not in current
 */
export function diffManifests(previousIdsSet, currentIdsSet) {
  const toAdd = [];
  const toDelete = [];

  for (const id of currentIdsSet) {
    if (!previousIdsSet.has(id)) toAdd.push(id);
  }
  for (const id of previousIdsSet) {
    if (!currentIdsSet.has(id)) toDelete.push(id);
  }

  return { toAdd, toDelete };
}
