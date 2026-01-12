// Just calls buildIndex().

// This keeps your index build process separate from server runtime.
import { buildIndex } from "./indexer.js";

async function main() {
  const modeArg = process.argv.find((x) => x.startsWith("--mode="));
  const mode = modeArg ? modeArg.split("=")[1] : "incremental";

  await buildIndex({
    mode,
    dataDir: "data",
    exts: ["txt", "md"],
    chunk: { chunkSize: 1200, chunkOverlap: 200 },
  });
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
