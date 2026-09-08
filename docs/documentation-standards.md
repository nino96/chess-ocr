# Documentation standards

How this repository's prose is organized, and the rules that keep it from drifting.
Read this before adding a Markdown file or appending to one. `pnpm run check`
enforces the mechanical parts; the rest is review.

These rules exist because the corpus previously accreted the opposite of each one:
a flat `docs/` with no taxonomy, one rule restated in five files, and dead text left
standing under `Owner supersession, <date>:` headers.

## The four kinds of document

Every Markdown file in this repository is exactly one of these. If a file is two of
them, split it.

| Kind | Question it answers | Lifecycle | Lives in |
|---|---|---|---|
| **Entry point** | Where do I start? | Edited in place; always current | `README.md`, `docs/index.md` |
| **Reference** | How does this work, today? | Edited in place; always current | `docs/*.md` |
| **Decision** | What did we choose, and when? | Immutable once written; superseded, never edited | [`docs/decisions/`](decisions/index.md) |
| **Record** | What happened on this date? | Immutable once written | [`docs/archive/`](archive/index.md) |

`PLAN.md` is reference: the current forward plan. `AGENTS.md` is reference: the
current rules. Neither is a log.

The test for reference vs. record: **if re-running the work would change the text,
it is reference; if re-running the work would produce a second text, it is a record.**
A benchmark table is a record. The command that produces it is reference.

## Rules for reference documents

**No dates in the body.** A reference doc describes the present tense. If you find
yourself writing "as of 2026-09-07" or "currently" or "now implemented", you are
writing a record or a decision — put it in the right place and link to it.

**No status.** "Implemented", "pending", "remains undelivered" belong to issues, not
to docs. The exception is a single explicit `## Open items` or `## Outstanding gates`
section where an incomplete state is itself the durable fact a reader needs.

**Never leave superseded text in place.** When a rule changes, *edit the rule*. Do
not append a supersession note below the text it kills — a reader who stops early
gets the wrong answer, and both copies then drift. The history goes in
`docs/decisions/` and in git; the doc states only what is true.

**No machine state.** "The pipeline is already initialized in the current workspace"
is false for every reader but one. Write what a fresh clone must do.

## One fact, one home

A fact is written once and linked everywhere else. Before adding a paragraph, grep
for it; if it exists, link instead.

Established homes:

| Fact | Home |
|---|---|
| Standing claims, scope boundaries, what is *not* claimed | `docs/scope-and-claims.md` |
| Resource ceilings, reservations and charges | `docs/budget.md` |
| Subsystem-to-file map, pipeline shape, vocabulary | `docs/architecture.md` |
| Dataset CLI surface | `docs/dataset-pipeline.md` |
| Proposal-provider contract | `docs/assisted-review.md` |
| GPU controller procedure | `docs/training-runbook.md` |
| What a fresh clone can rebuild | `docs/reproducibility.md` |

Duplicating a command block is the most common failure. If two docs need the same
commands, one of them owns them and the other links to the section.

## Rules for decisions and records

**Name them by date:** `docs/decisions/YYYY-MM-DD-slug.md`,
`docs/archive/<subject>-YYYY-MM-DD.md`.

**Open with a status line.** Decisions carry `Status: active` or
`Status: superseded by <link>`. Records carry one sentence saying what host, commit
and date produced them, and that nothing in the file is current.

**Do not edit them afterwards.** To change a decision, write a new one and mark the
old superseded. To correct a record, write a note in the newer record. This is the
whole reason they are separate from reference docs.

**Reference docs may cite them, but must not depend on them.** A reader who never
opens `docs/archive/` should still be able to do the work.

## Evidence that lives outside the repository

`work/`, `artifacts/`, `cache/`, `models/` and the other payload roots are
gitignored (`.gitignore`). Paths into them are legitimate in commands the reader
runs, but a path cited as *evidence for a claim* is unverifiable to everyone but the
author.

When a doc points at ignored storage as evidence, mark it: **local-only evidence,
not reproducible from this repository.** Never make a documented next action live
only in ignored storage — a fresh clone must be able to follow the docs.

## Links

Use relative Markdown links, including to `../PLAN.md` and `../AGENTS.md`.
`scripts/check-docs.mjs` resolves every relative link and its `#anchor` during
`pnpm run check`, so a rename that breaks a cross-reference fails the build.

Inline code spans are not checked. A path that matters should be a link.

Every doc must be reachable from `docs/index.md`. An unreachable doc is either
missing from the index or should not exist.

## Adding a document

1. Decide which of the four kinds it is. If the answer is "both", you have two files.
2. Grep for the facts it will state. Link the ones that already have a home.
3. Put it in the right directory and add it to `docs/index.md`.
4. Run `pnpm run check`.
