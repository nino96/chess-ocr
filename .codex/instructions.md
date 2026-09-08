# Codex instructions

The repository-wide rules are in [AGENTS.md](../AGENTS.md). This file adds only
what is specific to running Codex here; it does not restate or override policy.

## Sandbox execution

In the restricted Codex sandbox, request escalation before `pnpm run check`,
Playwright and browser suites, Git writes, and networked Git/GitHub commands.
Do not first retry these known-incompatible commands inside the sandbox.
