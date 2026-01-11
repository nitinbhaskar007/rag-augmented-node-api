// What it does

// Calls embeddings endpoint once with an array of text inputs

// Returns normalized vectors (embeddingUnit)

// This is used in:

// indexing (embed chunks)

// asking (embed queries)

// Important: this is server-side only.

import { client, normalizeText } from "./lib.js";
import { toUnitEmbedding } from "./vectorStore.js";

/**
 * Embeds an array of texts. Returns unit vectors (normalized).
 */
export async function embedTexts(
  texts,
  { model = "text-embedding-3-small" } = {}
) {
  const input = texts.map((t) => normalizeText(t));

  const res = await client.embeddings.create({
    model,
    input,
    encoding_format: "float",
  });

  return res.data.map((d) => toUnitEmbedding(d.embedding));
}
