// What it does

// Calls embeddings endpoint once with an array of text inputs

// Returns normalized vectors (embeddingUnit)

// This is used in:

// indexing (embed chunks)

// asking (embed queries)

// Important: this is server-side only.
// This file calls OpenAI embeddings API and returns unit vectors.

// What it does

// Takes array of strings

// Calls:

// client.embeddings.create({ model, input, encoding_format: "float" })

// Normalizes each embedding using normalizeVec() so cosine similarity becomes dot-product-friendly.

// Why it matters

// Used in indexing (embed chunks)

// Used in querying (embed user question + augmented variants)
import { client, normalizeText } from "./lib.js";
import { normalizeVec } from "./lib.js";

/**
 * Embeds an array of texts. Returns UNIT vectors (normalized).
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

  return res.data.map((d) => normalizeVec(d.embedding));
}
