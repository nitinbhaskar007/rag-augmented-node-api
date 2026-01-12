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

// // If you don’t set it, no auth.
// This is your web server entry.

// What it does
// A) Fastify setup

// CORS enabled (so frontend can call)

// rate limit enabled (to prevent abuse)

// optional API key auth via x-api-key

// B) Loads engine once at startup
// const engine = await initRagEngine(...)

// C) Routes

// GET /health

// simple “server is alive” check

// POST /ask

// validates input

// passes question + filters to engine.ask()

// returns answer + sources to frontend

// POST /reindex (optional)

// runs indexing pipeline again

// reloads LanceDB table in engine

// Why it matters

// This file is what makes your RAG usable by frontend safely.

// Quick “KT Summary” (best for interviews)

// indexer.js + loadDocs.js + embed.js + vectorStore.js = offline ingestion pipeline

// engine.js = online query pipeline (Augmented + Hybrid RAG)

// server.js = REST wrapper for frontend
import "dotenv/config";
import Fastify from "fastify";
import cors from "@fastify/cors";
import rateLimit from "@fastify/rate-limit";

import { initRagEngine } from "./rag/engine.js";
import { buildIndex } from "./indexer.js";

const PORT = Number(process.env.PORT || 3001);
const API_KEY = process.env.RAG_API_KEY || "";
const CORS_ORIGIN = process.env.CORS_ORIGIN || true;

function requireApiKey(req, reply) {
  if (!API_KEY) return;
  const key = req.headers["x-api-key"];
  if (key !== API_KEY) reply.code(401).send({ error: "Unauthorized" });
}

function normalizeAskPayload(body) {
  const question = body?.question;

  // Optional filters
  const filters =
    body?.filters && typeof body.filters === "object"
      ? body.filters
      : undefined;

  // Optional mustInclude keywords
  let mustInclude = body?.mustInclude;
  if (typeof mustInclude === "string") {
    // allow "refund partial" convenience
    mustInclude = mustInclude.split(/\s+/).filter(Boolean);
  }
  if (!Array.isArray(mustInclude)) mustInclude = undefined;

  const mustIncludeMode = body?.mustIncludeMode === "any" ? "any" : "all";

  return { question, filters, mustInclude, mustIncludeMode };
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

  app.get("/health", async () => ({ ok: true }));

  app.log.info("Loading RAG engine...");
  const engine = await initRagEngine({ log: app.log });
  app.log.info("RAG engine ready.");

  /**
   * POST /ask
   * Body:
   * {
   *   question: string,
   *   filters?: { sources?: string[], sourcePrefix?: string },
   *   mustInclude?: string[] | "keyword keyword",
   *   mustIncludeMode?: "all" | "any"
   * }
   */
  app.post("/ask", { preHandler: requireApiKey }, async (req, reply) => {
    try {
      const { question, filters, mustInclude, mustIncludeMode } =
        normalizeAskPayload(req.body);

      if (
        !question ||
        typeof question !== "string" ||
        question.trim().length < 2
      ) {
        return reply.code(400).send({ error: "question is required" });
      }

      const result = await engine.ask(question.trim(), {
        filters,
        mustInclude,
        mustIncludeMode,
      });

      return reply.send(result);
    } catch (err) {
      req.log.error(err);
      return reply
        .code(err.statusCode || 500)
        .send({ error: err.message || "Internal error" });
    }
  });

  /**
   * POST /reindex
   * Protected by x-api-key (if RAG_API_KEY is set)
   */
  app.post("/reindex", { preHandler: requireApiKey }, async (req, reply) => {
    try {
      const mode = req.body?.mode === "full" ? "full" : "incremental";

      app.log.info(`Reindex requested (mode=${mode})...`);
      const result = await buildIndex({ mode, logger: app.log });

      await engine.reloadStore();
      return reply.send({ ok: true, ...result });
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
