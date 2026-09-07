import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
const python = process.env.DATASET_PYTHON || "work/dataset-venv/bin/python";
if (!existsSync(python) && !process.env.DATASET_PYTHON) {
  throw new Error(
    "Dataset environment missing: follow docs/dataset-pipeline.md setup.",
  );
}
const serving = process.argv[2] === "serve";
if (serving) {
  const build = spawnSync(
    process.execPath,
    [
      "node_modules/typescript/bin/tsc",
      "--project",
      "tsconfig.dataset-app.json",
    ],
    { stdio: "inherit" },
  );
  if (build.status !== 0) process.exit(build.status ?? 1);
}
const result = spawnSync(
  python,
  [
    serving ? "python/dataset_server.py" : "python/dataset_pipeline.py",
    ...process.argv.slice(serving ? 3 : 2),
  ],
  {
    stdio: "inherit",
  },
);
if (result.error)
  throw new Error("Unable to launch dataset Python environment.");
process.exitCode = result.status ?? 1;
