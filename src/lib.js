// What it does

// Creates OpenAI client:

// export const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Normalizes text:

// replace Windows line breaks

// remove extra spaces
// This improves chunk quality + embedding consistency.

// Chunking:

// breaks doc into smaller chunks with overlap

// overlap avoids losing meaning at chunk boundaries

// Vector math:

// dot, l2norm, normalizeVec

// used for cosine similarity

// JSON helpers:

// saveJSON / loadJSON
// These persist your local store and caches.

// Interview line:
// “lib.js is the foundational utilities module: input cleanup, chunking, vector math, file IO, and OpenAI client setup.”

// This is the foundation module used everywhere.

// What it contains
// A) OpenAI client
// export const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// All OpenAI calls (embeddings + responses) use this.

// B) Text normalization

// normalizeText() makes inputs consistent:

// Windows newline → \n

// collapses extra spaces

// trims

// This reduces weird chunk boundaries + embedding variance.

// C) Chunking (Recursive Text Splitting)

// chunkTextRecursive():

// Splits text using separators in this order:

// \n\n (paragraph)

// \n (line)

// space (word)

// characters (fallback)

// Then merges smaller splits into chunks with:

// chunkSize

// chunkOverlap

// Why it exists:
// This is the “Chunk” step in your diagram and matches the slide (LangChain-style splitting, but your own implementation).

// D) Vector math

// dot(), l2norm(), normalizeVec()

// Used to normalize embeddings and for similarity math during diversity selection.

// E) JSON helpers

// saveJSON(), loadJSON()

// Used for caches in .cache/ and occasional storage.
import "dotenv/config";
import fs from "node:fs/promises";
import path from "node:path";
import OpenAI from "openai";

export const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

export function normalizeText(s) {
  return String(s)
    .replace(/\r\n/g, "\n")
    .replace(/[ \t]+/g, " ")
    .trim();
}

/**
 * Recursive text splitter (LangChain-style idea, but implemented locally)
 * Separators order:
 * 1) "\n\n" (paragraphs)
 * 2) "\n"   (lines)
 * 3) " "    (words)
 * 4) ""     (characters)
 *
 * We first split using the best separator, then recursively split long pieces,
 * and finally MERGE into chunks of chunkSize with chunkOverlap.
 */
export function chunkTextRecursive(
  text,
  {
    chunkSize = 1200,
    chunkOverlap = 200,
    separators = ["\n\n", "\n", " ", ""],
  } = {}
) {
  const t = normalizeText(text);
  if (!t) return [];
  if (t.length <= chunkSize) return [t];

  // 1) Split recursively into smaller “units”
  const splits = splitRecursively(t, separators, chunkSize);

  // 2) Merge those units into final chunks with overlap
  return mergeSplitsWithOverlap(splits, chunkSize, chunkOverlap);
}

function splitRecursively(text, separators, chunkSize) {
  if (text.length <= chunkSize) return [text];

  // If no separators left, fall back to hard slicing
  if (!separators.length)
    return [
      text.slice(0, chunkSize),
      ...splitRecursively(text.slice(chunkSize), [], chunkSize),
    ];

  const [sep, ...rest] = separators;

  // If separator is "" → split into characters
  const parts = sep === "" ? Array.from(text) : text.split(sep);

  // If splitting did nothing (no separator found), try smaller separator
  if (parts.length === 1) return splitRecursively(text, rest, chunkSize);

  // Now recursively split any part that is still too large
  const out = [];
  for (const p of parts) {
    const piece = p.trim();
    if (!piece) continue;

    if (piece.length <= chunkSize) out.push(piece);
    else out.push(...splitRecursively(piece, rest, chunkSize));
  }
  return out;
}

function mergeSplitsWithOverlap(splits, chunkSize, chunkOverlap) {
  const chunks = [];
  let current = "";

  const pushCurrent = () => {
    const c = current.trim();
    if (c) chunks.push(c);
  };

  for (const s of splits) {
    if (!current) {
      current = s;
      continue;
    }

    // If adding this split would exceed chunkSize, finalize current
    if ((current + " " + s).length > chunkSize) {
      pushCurrent();

      // overlap: carry last chunkOverlap chars into new chunk
      const overlapText = current.slice(
        Math.max(0, current.length - chunkOverlap)
      );
      current = (overlapText + " " + s).trim();

      // If overlap itself makes it too big (rare), hard slice
      while (current.length > chunkSize) {
        chunks.push(current.slice(0, chunkSize));
        current = current.slice(Math.max(0, chunkSize - chunkOverlap)).trim();
      }
    } else {
      current = (current + " " + s).trim();
    }
  }

  pushCurrent();
  return chunks;
}

/** Vector math */
export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function l2norm(v) {
  return Math.sqrt(dot(v, v));
}

export function normalizeVec(v) {
  const n = l2norm(v) || 1;
  return v.map((x) => x / n);
}

/** JSON IO */
export async function saveJSON(filePath, obj) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, JSON.stringify(obj, null, 2), "utf-8");
}

export async function loadJSON(filePath) {
  const raw = await fs.readFile(filePath, "utf-8");
  return JSON.parse(raw);
}
