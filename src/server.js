// This wraps the engine in HTTP routes.

// Key points
// GET /health

// Simple health check.

// On startup:
// const engine = await initRagEngine(...)

// So index is loaded once.

// POST /ask

// validates input

// calls engine.ask(question)

// returns answer JSON

// POST /reindex

// protected endpoint

// rebuilds index from docs using buildIndex

// reloads store in engine

// Security: RAG_API_KEY

// If you set it in .env, server requires:

// header x-api-key

// If you don’t set it, no auth.

import "dotenv/config";
import Fastify from "fastify";
import cors from "@fastify/cors";
import rateLimit from "@fastify/rate-limit";

import { initRagEngine } from "./rag/engine.js";
import { buildIndex } from "./indexer.js";

const PORT = Number(process.env.PORT || 3001);
const INDEX_PATH = process.env.RAG_INDEX_PATH || "index/store.json";
const API_KEY = process.env.RAG_API_KEY || "";
const CORS_ORIGIN = process.env.CORS_ORIGIN || true;

function requireApiKey(req, reply) {
  if (!API_KEY) return;
  const key = req.headers["x-api-key"];
  if (key !== API_KEY) {
    reply.code(401).send({ error: "Unauthorized" });
  }
}

async function main() {
  const app = Fastify({ logger: true });

  await app.register(cors, {
    origin: CORS_ORIGIN === true ? true : CORS_ORIGIN,
  });

  await app.register(rateLimit, {
    max: 60,
    timeWindow: "1 minute",
  });

  // Health check
  app.get("/health", async () => ({ ok: true }));

  // Load engine once
  app.log.info("Loading RAG engine...");
  const engine = await initRagEngine({
    indexPath: INDEX_PATH,
    log: app.log,
  });
  app.log.info(`RAG engine ready. Chunks loaded: ${engine.store.items.length}`);

  // Main ask endpoint
  app.post("/ask", { preHandler: requireApiKey }, async (req, reply) => {
    try {
      const { question } = req.body || {};
      if (
        !question ||
        typeof question !== "string" ||
        question.trim().length < 2
      ) {
        return reply.code(400).send({ error: "question is required" });
      }

      const result = await engine.ask(question.trim());
      return reply.send(result);
    } catch (err) {
      req.log.error(err);
      return reply
        .code(err.statusCode || 500)
        .send({ error: err.message || "Internal error" });
    }
  });

  /**
   * Optional: /reindex to rebuild index from data/
   * Protected by x-api-key (if RAG_API_KEY is set).
   */
  app.post("/reindex", { preHandler: requireApiKey }, async (req, reply) => {
    try {
      app.log.info("Reindex requested...");
      await buildIndex({
        dataDir: "data",
        exts: ["txt", "md"],
        chunk: { maxChars: 1200, overlapChars: 200 },
        indexPath: INDEX_PATH,
        logger: app.log,
      });

      const newCount = await engine.reloadStore();
      app.log.info(`Reindex complete. Reloaded chunks: ${newCount}`);

      return reply.send({ ok: true, chunks: newCount });
    } catch (err) {
      req.log.error(err);
      return reply
        .code(err.statusCode || 500)
        .send({ error: err.message || "Reindex failed" });
    }
  });

  await app.listen({ port: PORT, host: "0.0.0.0" });
  app.log.info(`Server running: http://localhost:${PORT}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
