import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
const python = process.env.DATASET_PYTHON || "work/dataset-venv/bin/python";
if (!existsSync(python) && !process.env.DATASET_PYTHON) {
  throw new Error(
    "Dataset environment missing: follow docs/dataset-pipeline.md setup.",
  );
}
const result = spawnSync(
  python,
  ["python/dataset_pipeline.py", ...process.argv.slice(2)],
  {
    stdio: "inherit",
  },
);
if (result.error)
  throw new Error("Unable to launch dataset Python environment.");
process.exitCode = result.status ?? 1;
