import { loadJSON, saveJSON, normalizeVec, dot } from "./lib.js";

/**
 * LocalVectorStore
 * Stores: [{ id, source, chunkIndex, content, embeddingUnit }]
 * embeddingUnit is expected to be a unit (normalized) vector.
 */
export class LocalVectorStore {
  constructor(items = []) {
    this.items = items;
  }

  static async load(filePath) {
    const data = await loadJSON(filePath);
    return new LocalVectorStore(data.items || []);
  }

  async save(filePath) {
    await saveJSON(filePath, { items: this.items });
  }

  add(item) {
    this.items.push(item);
  }

  search(queryEmbeddingUnit, { topK = 8 } = {}) {
    const scored = this.items.map((it) => ({
      item: it,
      // cosine similarity = dot since both are unit vectors
      score: dot(queryEmbeddingUnit, it.embeddingUnit),
    }));

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK);
  }

  /**
   * For augmented retrieval: union of results across multiple query embeddings,
   * keeping the best score per chunk id.
   */
  searchMulti(queryEmbeddingUnits, { perQueryTopK = 8, finalTopK = 20 } = {}) {
    const best = new Map(); // id -> { item, score }

    for (const q of queryEmbeddingUnits) {
      const hits = this.search(q, { topK: perQueryTopK });
      for (const h of hits) {
        const prev = best.get(h.item.id);
        if (!prev || h.score > prev.score) best.set(h.item.id, h);
      }
    }

    return [...best.values()]
      .sort((a, b) => b.score - a.score)
      .slice(0, finalTopK);
  }
}

export function toUnitEmbedding(embeddingFloatArray) {
  return normalizeVec(embeddingFloatArray);
}
