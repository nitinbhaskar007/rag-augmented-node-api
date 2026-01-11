import fs from "node:fs/promises";
import path from "node:path";

import { client, dot } from "../lib.js";
import { LocalVectorStore } from "../vectorStore.js";
import { embedTexts } from "../embed.js";
import {
  ANSWER_INSTRUCTIONS,
  MULTI_QUERY_INSTRUCTIONS,
  HYDE_INSTRUCTIONS,
} from "../prompts.js";

/* ----------------------- Config ----------------------- */
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

/* ----------------------- Cache helpers ----------------------- */
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

/* ----------------------- Error handling ----------------------- */
function classifyOpenAIError(err) {
  const status = err?.status;
  const code = err?.code || err?.error?.code;
  const message = (err?.message || err?.error?.message || "").toLowerCase();

  const isQuota =
    status === 429 &&
    (code === "insufficient_quota" || message.includes("quota"));
  const isRateLimit =
    status === 429 &&
    (code === "rate_limit_exceeded" ||
      message.includes("rate limit") ||
      message.includes("too many requests"));

  const isTransient =
    (status >= 500 && status <= 599) ||
    message.includes("timeout") ||
    message.includes("temporarily");

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

      // Quota never succeeds by retry
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

/* ----------------------- Diversity selection (MMR-ish) ----------------------- */
/**
 * KT-friendly: picks top chunks but avoids near-duplicates in context.
 */
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
      const sim = dot(h.item.embeddingUnit, pe); // cosine (unit vectors)
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

/* ----------------------- Cached augmentation ----------------------- */
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

/* ----------------------- Cached embeddings ----------------------- */
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
      embedCache[keys[missIndexes[j]]] = v; // unit vectors
    });
  }

  return keys.map((k) => embedCache[k]);
}

/* ----------------------- Cached answering ----------------------- */
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

/* ----------------------- Engine factory ----------------------- */
/**
 * initRagEngine()
 * - loads store + caches once
 * - returns { ask(), reloadStore() }
 */
export async function initRagEngine({ indexPath, log = console } = {}) {
  // Load index
  let store = await LocalVectorStore.load(indexPath);

  // Load caches
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
    get store() {
      return store;
    },

    async reloadStore() {
      store = await LocalVectorStore.load(indexPath);
      return store.items.length;
    },

    /**
     * ask(question)
     * Returns: { answer, sources, debug? }
     */
    async ask(question) {
      const started = Date.now();

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
          // If quota missing, continue without augmentation
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

      const variantTexts = [question, ...rewrites, hyde].filter(Boolean);

      // 2) Embed query variants (required)
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

      // 3) Retrieve candidates
      const mergedHits = store.searchMulti(variantEmbeds, {
        perQueryTopK: PER_QUERY_TOPK,
        finalTopK: FINAL_TOPK,
      });

      // 4) Pick diverse top chunks
      const selected = pickDiverse(mergedHits, { k: CONTEXT_K, lambda: 0.8 });

      // 5) Build context
      const context = buildContextBlock(selected);

      // 6) Answer
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

      // 7) Persist caches (safe, serialized)
      await persistCaches();

      const durationMs = Date.now() - started;

      return {
        answer,
        sources: selected.map((h) => h.item.id),
        debug: DEBUG
          ? {
              rewrites,
              hydeUsed: Boolean(hyde),
              retrievedCandidates: mergedHits.length,
              contextChunks: selected.length,
              durationMs,
            }
          : undefined,
      };
    },
  };
}
