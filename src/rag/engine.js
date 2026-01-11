// This is the “brain” of your API.

// Why engine exists

// In server apps, we should:

// load index once

// reuse it for all requests

// avoid re-reading and re-parsing on each call

// What engine contains

// Caches

// Embeddings cache

// Augmentation cache (rewrites & hyde)

// Answer cache

// Stored in .cache.json

// Purpose:

// avoid repeated embedding cost

// faster responses

// Error handling

// classifyOpenAIError() decides:

// insufficient_quota

// rate_limit_exceeded

// transient 5xx/timeouts

// withRetry() does exponential backoff for transient + rate-limit errors

// This gives a real production pattern.

// Augmentation functions

// getMultiQueriesCached()

// getHydeCached()

// If quota is missing, engine falls back to normal RAG (no augmentation).

// Embedding function

// embedTextsCached() embeds only cache misses

// Diversity selection

// pickDiverse() reduces duplicate context chunks

// initRagEngine() factory
// Loads:

// vector store

// caches

// Returns:

// ask(question) function

// reloadStore() (for reindex)

// What ask(question) returns
// {
//   answer: "...",
//   sources: ["data/file#idx", ...],
//   debug: {...optional}
// }

// This is the most important file at runtime.

// What it does (step-by-step)
// A) Startup initialization

// connects to LanceDB (vector store)

// loads caches:

// .cache/embeddings.json

// .cache/augment.json

// .cache/answers.json

// ensures indexes exist (best effort)

// B) Implements ask(question, options)

// This function runs the full RAG pipeline:

// 1) Augmentation (optional)

// Multi-query rewrite (3 variants)

// HyDE hypothetical answer

// If quota is missing → skips augmentation safely

// 2) Embedding

// Embeds all query variants

// Uses embedding cache to avoid re-embedding same text

// 3) Retrieval

// Hybrid retrieval: vector + BM25

// Multi-variant retrieval: do it for each variant

// Merge results across variants

// 4) Filtering + must-include (if enabled)

// filters by allowed sources/prefix

// enforces keywords present in chunk text

// 5) Diversity selection

// picks top chunks but avoids near duplicates (MMR-ish)

// 6) Context packing

// builds prompt context with [source: ...]

// 7) Answer generation

// uses ANSWER_INSTRUCTIONS to keep output grounded

// 8) Cache persistence

// writes caches back to disk safely using a write queue

// Why it matters

// This is your “production-level RAG logic”, extracted cleanly so:

// server can call it per request

// you can test it separately

// easy to explain in interviews
import fs from "node:fs/promises";
import path from "node:path";
import { client, dot } from "../lib.js";
import { embedTexts } from "../embed.js";
import { LanceVectorStore } from "../vectorStore.js";
import {
  ANSWER_INSTRUCTIONS,
  MULTI_QUERY_INSTRUCTIONS,
  HYDE_INSTRUCTIONS,
} from "../prompts.js";

/* ---------------- Config ---------------- */
const CACHE_DIR = ".cache";
const EMBED_CACHE_PATH = path.join(CACHE_DIR, "embeddings.json");
const AUGMENT_CACHE_PATH = path.join(CACHE_DIR, "augment.json");
const ANSWER_CACHE_PATH = path.join(CACHE_DIR, "answers.json");

const GEN_MODEL = process.env.RAG_GEN_MODEL || "gpt-4.1-mini";
const EMBED_MODEL = process.env.RAG_EMBED_MODEL || "text-embedding-3-small";

const ENABLE_MULTI_QUERY = (process.env.RAG_MULTI_QUERY ?? "true") === "true";
const ENABLE_HYDE = (process.env.RAG_HYDE ?? "true") === "true";

const PER_QUERY_TOPK = Number(process.env.RAG_PER_QUERY_TOPK || 8);
const FINAL_TOPK = Number(process.env.RAG_FINAL_TOPK || 25);
const CONTEXT_K = Number(process.env.RAG_CONTEXT_K || 6);

const DEBUG = (process.env.RAG_DEBUG ?? "false") === "true";

// Hybrid toggle
const ENABLE_HYBRID = (process.env.RAG_HYBRID ?? "true") === "true";
const RRF_K = Number(process.env.RAG_RRF_K || 60);

const LANCEDB_URI = process.env.LANCEDB_URI || "./.lancedb";
const LANCEDB_TABLE = process.env.LANCEDB_TABLE || "rag_chunks";
const VECTOR_COLUMN = process.env.RAG_VECTOR_COLUMN || "vector";
const FTS_COLUMN = process.env.RAG_FTS_COLUMN || "content";

/* ---------------- Cache helpers ---------------- */
function hashKey(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0).toString(16);
}

async function ensureCacheDir() {
  await fs.mkdir(CACHE_DIR, { recursive: true });
}

async function readJsonSafe(filePath, fallbackObj) {
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    return JSON.parse(raw);
  } catch {
    return fallbackObj;
  }
}

let cacheWriteChain = Promise.resolve();
function queueCacheWrite(fn) {
  cacheWriteChain = cacheWriteChain.then(fn).catch(() => {});
  return cacheWriteChain;
}

async function writeJson(filePath, obj) {
  await ensureCacheDir();
  await fs.writeFile(filePath, JSON.stringify(obj, null, 2), "utf-8");
}

/* ---------------- Error handling ---------------- */
function classifyOpenAIError(err) {
  const status = err?.status;
  const code = err?.code || err?.error?.code;
  const msg = (err?.message || err?.error?.message || "").toLowerCase();

  const isQuota =
    status === 429 && (code === "insufficient_quota" || msg.includes("quota"));
  const isRateLimit =
    status === 429 &&
    (code === "rate_limit_exceeded" ||
      msg.includes("rate limit") ||
      msg.includes("too many requests"));

  const isTransient =
    (status >= 500 && status <= 599) ||
    msg.includes("timeout") ||
    msg.includes("temporarily");

  return { status, code, isQuota, isRateLimit, isTransient };
}

async function withRetry(
  fn,
  { label = "operation", maxRetries = 4, log } = {}
) {
  let attempt = 0;
  while (true) {
    try {
      return await fn();
    } catch (err) {
      const info = classifyOpenAIError(err);
      if (info.isQuota) throw err;

      const canRetry = info.isRateLimit || info.isTransient;
      if (!canRetry || attempt >= maxRetries) throw err;

      const backoffMs =
        Math.min(8000, 500 * 2 ** attempt) + Math.floor(Math.random() * 250);
      log?.warn?.(`${label} failed; retrying`, {
        attempt: attempt + 1,
        backoffMs,
        status: info.status,
        code: info.code,
      });
      await new Promise((r) => setTimeout(r, backoffMs));
      attempt++;
    }
  }
}

/* ---------------- Diversity selection (MMR-ish) ---------------- */
function pickDiverse(
  hits,
  { k = CONTEXT_K, lambda = 0.8, minKeep = 0.1 } = {}
) {
  const picked = [];
  const pickedEmbeds = [];

  for (const h of hits) {
    if (picked.length >= k) break;

    let maxSimToPicked = 0;
    for (const pe of pickedEmbeds) {
      const sim = dot(h.item.embeddingUnit, pe);
      if (sim > maxSimToPicked) maxSimToPicked = sim;
    }

    const mmrScore = lambda * h.score - (1 - lambda) * maxSimToPicked;

    if (picked.length === 0 || mmrScore > minKeep) {
      picked.push(h);
      pickedEmbeds.push(h.item.embeddingUnit);
    }
  }
  return picked;
}

function buildContextBlock(selectedHits) {
  return selectedHits
    .map((h) => {
      const { source, chunkIndex, content } = h.item;
      return `[source: ${source}#${chunkIndex}]\n${content}`;
    })
    .join("\n\n---\n\n");
}

/* ---------------- Filtering helpers ---------------- */

/**
 * filters:
 * {
 *   sources?: string[]        // allow-list of sources
 *   sourcePrefix?: string     // allow sources starting with prefix
 * }
 */
function applySourceFilters(hits, filters) {
  if (!filters) return hits;

  const allowSources = Array.isArray(filters.sources)
    ? new Set(filters.sources)
    : null;
  const prefix =
    typeof filters.sourcePrefix === "string" ? filters.sourcePrefix : null;

  return hits.filter((h) => {
    const src = h.item.source;

    if (allowSources && !allowSources.has(src)) return false;
    if (prefix && !src.startsWith(prefix)) return false;

    return true;
  });
}

/**
 * mustInclude: string[] keywords
 * mode: "all" | "any"
 *
 * Applies AFTER retrieval.
 * We retrieve extra candidates first so filtering doesn't kill recall.
 */
function applyMustInclude(hits, mustInclude, mode = "all") {
  if (!Array.isArray(mustInclude) || mustInclude.length === 0) return hits;

  const kws = mustInclude.map((k) => String(k).toLowerCase()).filter(Boolean);
  if (kws.length === 0) return hits;

  return hits.filter((h) => {
    const text = (h.item.content || "").toLowerCase();

    if (mode === "any") {
      return kws.some((k) => text.includes(k));
    }
    // default: "all"
    return kws.every((k) => text.includes(k));
  });
}

/* ---------------- Cached augmentation ---------------- */
async function getMultiQueriesCached(question, augmentCache, log) {
  const key = `mq:${GEN_MODEL}:${hashKey(question)}`;
  if (augmentCache[key]) return augmentCache[key];

  const resp = await withRetry(
    () =>
      client.responses.create({
        model: GEN_MODEL,
        instructions: MULTI_QUERY_INSTRUCTIONS,
        input: question,
        temperature: 0.2,
      }),
    { label: "multi-query", log }
  );

  const raw = resp.output_text?.trim() || "";
  let queries = [];
  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed.queries)) queries = parsed.queries.slice(0, 3);
  } catch {
    queries = [];
  }

  augmentCache[key] = queries;
  return queries;
}

async function getHydeCached(question, augmentCache, log) {
  const key = `hyde:${GEN_MODEL}:${hashKey(question)}`;
  if (augmentCache[key]) return augmentCache[key];

  const resp = await withRetry(
    () =>
      client.responses.create({
        model: GEN_MODEL,
        instructions: HYDE_INSTRUCTIONS,
        input: question,
        temperature: 0.3,
      }),
    { label: "hyde", log }
  );

  const hyde = (resp.output_text || "").trim();
  augmentCache[key] = hyde;
  return hyde;
}

/* ---------------- Cached embeddings ---------------- */
async function embedTextsCached(texts, embedCache, log) {
  const keys = texts.map((t) => `emb:${EMBED_MODEL}:${hashKey(t)}`);
  const misses = [];
  const missIndexes = [];

  keys.forEach((k, i) => {
    if (!embedCache[k]) {
      misses.push(texts[i]);
      missIndexes.push(i);
    }
  });

  if (misses.length > 0) {
    const vectors = await withRetry(
      () => embedTexts(misses, { model: EMBED_MODEL }),
      { label: "embeddings", log }
    );

    vectors.forEach((v, j) => {
      embedCache[keys[missIndexes[j]]] = v;
    });
  }

  return keys.map((k) => embedCache[k]);
}

/* ---------------- Cached answering ---------------- */
async function answerWithContextCached(question, context, answerCache, log) {
  const key = `ans:${GEN_MODEL}:${hashKey(question)}:${hashKey(context)}`;
  if (answerCache[key]) return answerCache[key];

  const resp = await withRetry(
    () =>
      client.responses.create({
        model: GEN_MODEL,
        instructions: ANSWER_INSTRUCTIONS,
        input: `CONTEXT:\n\n${context}\n\nUSER QUESTION:\n${question}`,
        temperature: 0.2,
      }),
    { label: "answer", log }
  );

  const out = resp.output_text;
  answerCache[key] = out;
  return out;
}

/* ---------------- Engine factory ---------------- */
export async function initRagEngine({ log = console } = {}) {
  const store = await LanceVectorStore.init({
    uri: LANCEDB_URI,
    tableName: LANCEDB_TABLE,
    vectorColumn: VECTOR_COLUMN,
    ftsColumn: FTS_COLUMN,
  });

  // best-effort index creation (safe if already exists)
  try {
    await store.ensureIndexes();
  } catch {}

  const embedCache = await readJsonSafe(EMBED_CACHE_PATH, {});
  const augmentCache = await readJsonSafe(AUGMENT_CACHE_PATH, {});
  const answerCache = await readJsonSafe(ANSWER_CACHE_PATH, {});

  async function persistCaches() {
    await queueCacheWrite(async () => {
      await writeJson(EMBED_CACHE_PATH, embedCache);
      await writeJson(AUGMENT_CACHE_PATH, augmentCache);
      await writeJson(ANSWER_CACHE_PATH, answerCache);
    });
  }

  return {
    store,

    async reloadStore() {
      await store.reload();
    },

    /**
     * ask(question, options?)
     * options:
     * {
     *   filters?: { sources?: string[], sourcePrefix?: string }
     *   mustInclude?: string[]
     *   mustIncludeMode?: "all" | "any"
     * }
     */
    async ask(question, options = {}) {
      const started = Date.now();
      const { filters, mustInclude, mustIncludeMode = "all" } = options;

      // 1) Augment query (best-effort)
      let rewrites = [];
      let hyde = "";

      try {
        if (ENABLE_MULTI_QUERY)
          rewrites = await getMultiQueriesCached(question, augmentCache, log);
        if (ENABLE_HYDE)
          hyde = await getHydeCached(question, augmentCache, log);
      } catch (err) {
        const info = classifyOpenAIError(err);
        if (info.isQuota) {
          log.warn?.(
            "No quota for augmentation; continuing without rewrites/HyDE.",
            { code: info.code }
          );
          rewrites = [];
          hyde = "";
        } else {
          throw err;
        }
      }

      // Keep texts aligned with embeddings
      const variantTexts = [question, ...rewrites, hyde].filter(Boolean);

      // 2) Embed query variants
      let variantEmbeds;
      try {
        variantEmbeds = await embedTextsCached(variantTexts, embedCache, log);
      } catch (err) {
        const info = classifyOpenAIError(err);
        if (info.isQuota) {
          const e = new Error(
            "No API quota for embeddings. Add credits or use local embeddings."
          );
          e.statusCode = 503;
          throw e;
        }
        throw err;
      }

      // 3) Retrieve MORE candidates first (important for filtering)
      const expandedFinalTopK = Math.max(FINAL_TOPK * 4, FINAL_TOPK);
      const expandedPerQueryTopK = Math.max(PER_QUERY_TOPK * 3, PER_QUERY_TOPK);

      let mergedHits;
      if (ENABLE_HYBRID) {
        mergedHits = await store.hybridSearchMulti(
          variantEmbeds,
          variantTexts,
          {
            perQueryTopK: expandedPerQueryTopK,
            finalTopK: expandedFinalTopK,
            rrfK: RRF_K,
          }
        );
      } else {
        // If you ever disable hybrid, you can add vector-only search here
        mergedHits = await store.hybridSearchMulti(
          variantEmbeds,
          variantTexts,
          {
            perQueryTopK: expandedPerQueryTopK,
            finalTopK: expandedFinalTopK,
            rrfK: RRF_K,
          }
        );
      }

      // 4) Apply Filters + Must-Include keywords
      let filtered = applySourceFilters(mergedHits, filters);
      filtered = applyMustInclude(filtered, mustInclude, mustIncludeMode);

      // If filters are too strict, you may end up with 0 chunks
      if (filtered.length === 0) {
        return {
          answer:
            "I couldn't find relevant passages that match your filters/keywords in the provided documents.",
          sources: [],
          debug: DEBUG
            ? {
                hybrid: ENABLE_HYBRID,
                filters,
                mustInclude,
                mustIncludeMode,
                retrievedCandidates: mergedHits.length,
                afterFiltering: 0,
                durationMs: Date.now() - started,
              }
            : undefined,
        };
      }

      // 5) Diversity select from filtered hits
      const selected = pickDiverse(filtered, { k: CONTEXT_K });

      // 6) Context
      const context = buildContextBlock(selected);

      // 7) Answer
      let answer;
      try {
        answer = await answerWithContextCached(
          question,
          context,
          answerCache,
          log
        );
      } catch (err) {
        const info = classifyOpenAIError(err);
        if (info.isQuota) {
          const e = new Error(
            "No API quota for answering. Add credits or use a local LLM."
          );
          e.statusCode = 503;
          throw e;
        }
        throw err;
      }

      await persistCaches();

      const durationMs = Date.now() - started;

      return {
        answer,
        sources: selected.map((h) => h.item.id),
        debug: DEBUG
          ? {
              hybrid: ENABLE_HYBRID,
              rrfK: RRF_K,
              filters,
              mustInclude,
              mustIncludeMode,
              rewrites,
              hydeUsed: Boolean(hyde),
              retrievedCandidates: mergedHits.length,
              afterFiltering: filtered.length,
              contextChunks: selected.length,
              durationMs,
            }
          : undefined,
      };
    },
  };
}
