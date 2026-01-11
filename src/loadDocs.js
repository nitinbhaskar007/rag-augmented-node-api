// What it does

// Finds files in data/**/*.txt|md

// Reads them

// Chunks each file

// Converts into chunk objects:

// {
//   id: "data/policies.md#2",
//   source: "data/policies.md",
//   chunkIndex: 2,
//   content: "text of chunk..."
// }

// Why this matters

// Vectors don’t store “documents”, they store chunks.
// RAG retrieval works best on chunk-level.

// Interview line:
// “Instead of embedding whole documents, we embed chunked passages to increase retrieval precision.”

import fs from "node:fs/promises";
import path from "node:path";
import { glob } from "glob";
import { chunkTextRecursive } from "./lib.js";

export async function loadAndChunkDocs({
  dataDir = "data",
  exts = ["txt", "md"],
  chunk = { chunkSize: 1200, chunkOverlap: 200 },
} = {}) {
  const patterns = exts.map((e) => path.join(dataDir, `**/*.${e}`));
  const files = (await glob(patterns, { nodir: true })).sort();

  const chunks = [];
  for (const file of files) {
    const text = await fs.readFile(file, "utf-8");
    const parts = chunkTextRecursive(text, chunk);

    parts.forEach((content, idx) => {
      const normalizedFile = file.split(path.sep).join("/");
      chunks.push({
        id: `${normalizedFile}#${idx}`,
        source: normalizedFile,
        chunkIndex: idx,
        content,
      });
    });
  }

  return chunks;
}
