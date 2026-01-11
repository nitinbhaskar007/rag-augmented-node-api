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

export function chunkText(text, { maxChars = 1200, overlapChars = 200 } = {}) {
  const t = normalizeText(text);
  const chunks = [];
  let i = 0;

  while (i < t.length) {
    const end = Math.min(i + maxChars, t.length);
    chunks.push(t.slice(i, end));
    if (end === t.length) break;
    i = Math.max(0, end - overlapChars);
  }

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
