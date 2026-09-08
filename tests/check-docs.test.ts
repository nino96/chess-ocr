import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const checker = fileURLToPath(
  new URL("../scripts/check-docs.mjs", import.meta.url),
);

function fixture(): string {
  const root = mkdtempSync(join(tmpdir(), "chess-ocr-docs-"));
  mkdirSync(join(root, "docs"));
  mkdirSync(join(root, ".agents"));
  mkdirSync(join(root, ".codex"));
  writeFileSync(join(root, "PLAN.md"), "# Plan\n");
  writeFileSync(join(root, "AGENTS.md"), "# Agents\n");
  writeFileSync(join(root, ".agents/delegation.md"), "# Deleg-sites\n");
  writeFileSync(join(root, ".codex/instructions.md"), "# Codex\n");
  writeFileSync(join(root, "docs/index.md"), "# Documentation\n");
  return root;
}

function run(root: string) {
  return spawnSync(process.execPath, [checker], {
    cwd: root,
    encoding: "utf8",
  });
}

test("documentation anchors follow GitHub Unicode and collision slugs", () => {
  const root = fixture();
  try {
    writeFileSync(
      join(root, "README.md"),
      [
        "# Café",
        "",
        "## Repeat",
        "",
        "## Repeat-1",
        "",
        "## Repeat",
        "",
        "[Unicode](#café) and [collision](#repeat-2).",
        "",
      ].join("\n"),
    );
    const result = run(root);
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.stdout, /2 anchors/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("documentation checks include tool-specific instruction files", () => {
  const root = fixture();
  try {
    writeFileSync(join(root, "README.md"), "# Readme\n");
    writeFileSync(join(root, ".agents/delegation.md"), "# Delegation  \n");
    const result = run(root);
    assert.notEqual(result.status, 0);
    assert.match(
      result.stderr,
      /\.agents\/delegation\.md: trailing whitespace/,
    );
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("documentation checks reject case-mismatched anchors", () => {
  const root = fixture();
  try {
    writeFileSync(join(root, "README.md"), "# Readme\n\n[Wrong](#Readme)\n");
    const result = run(root);
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /no heading '#Readme'/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
