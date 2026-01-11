import { buildIndex } from "./indexer.js";

async function main() {
  await buildIndex({
    dataDir: "data",
    exts: ["txt", "md"],
    chunk: { maxChars: 1200, overlapChars: 200 },
    indexPath: "index/store.json",
  });
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
