import { spawnSync } from "node:child_process";
const result = spawnSync(
  process.env.DATASET_PYTHON || "work/dataset-venv/bin/python",
  ["python/synthetic_job.py", ...process.argv.slice(2)],
  { stdio: "inherit" },
);
if (result.error) throw new Error("Dataset Python environment is unavailable");
process.exitCode = result.status ?? 1;
