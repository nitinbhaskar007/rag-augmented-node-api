// hree prompts:
// ANSWER_INSTRUCTIONS

// Forces the model:

// only use provided context

// if missing, say “I don’t know…”

// cite sources in a consistent format

// This prevents hallucinations.

// MULTI_QUERY_INSTRUCTIONS

// Forces output as valid JSON:

// {"queries":["...","...","..."]}

// HYDE_INSTRUCTIONS

// Generates a hypothetical answer
// Used only for retrieval embedding (not shown to user).

// Interview line:
// “We use prompt constraints to control model behavior and reduce hallucination risk.”

export const ANSWER_INSTRUCTIONS = `
You are a careful assistant. Answer ONLY using the provided CONTEXT.
If the answer is not in the context, say: "I don't know from the provided documents."
Cite sources inline like: [source: filename#chunkIndex].
Keep the answer clear and structured.
`.trim();

export const MULTI_QUERY_INSTRUCTIONS = `
You generate search queries to retrieve relevant passages from a document store.
Return ONLY valid JSON with this exact shape:
{"queries":["...","...","..."]}

Rules:
- Create 3 short, different queries.
- Include key nouns, acronyms, and synonyms.
- No extra keys. No markdown. No commentary.
`.trim();

export const HYDE_INSTRUCTIONS = `
Write a short answer (3-6 sentences) to the user's question.
This is only used for retrieval. Do not mention it's hypothetical.
`.trim();
