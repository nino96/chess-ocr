import { spawnSync } from "node:child_process";

const args = process.argv.slice(2);
if (args[0] === "--") args.shift();
const result = spawnSync(
  process.env.CHESS_OCR_PYTHON ?? "python3",
  ["python/training_job.py", ...args],
  { cwd: process.cwd(), stdio: "inherit" },
);
if (result.error) throw result.error;
process.exitCode = result.status ?? 1;
