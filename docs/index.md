# Documentation index

Every Markdown file in this repository is listed here. Its kind tells you how to
read it — see [documentation standards](documentation-standards.md) for what the
kinds mean and the rules each one follows.

## Entry points

| Document | What it is |
| --- | --- |
| [README](../README.md) | Run the demo, run the checks, use the library contract. |
| [AGENTS](../AGENTS.md) | Repository-wide rules for anyone, human or agent, changing this project. |
| [documentation standards](documentation-standards.md) | How docs are organized, and the rules that keep them from drifting. |

## Reference — current, edited in place

| Document | What it is |
| --- | --- |
| [PLAN](../PLAN.md) | The forward delivery plan: ownership, architecture hypotheses, dataset targets, success criteria. |
| [scope and standing claims](scope-and-claims.md) | What this project does and does not claim. The single home for every disclaimer. |
| [architecture](architecture.md) | Pipeline shape, vocabulary, diagnostic modes, export formats, subsystem-to-file map. |
| [operator workflow](operator-workflow.md) | The owner-facing narrative: what happens at each stage and where your input is needed. |
| [dataset pipeline](dataset-pipeline.md) | Command reference for the local dataset CLI and review dashboard. |
| [assisted review](assisted-review.md) | The proposal-provider contract: manifests, result schema, job lifecycle, editor rules. |
| [synthetic dataset](synthetic-dataset.md) | The deterministic renderer and job controller. |
| [training runbook](training-runbook.md) | The GPU training controller, its gates and stop conditions. |
| [local candidate](local-candidate.md) | Preparing and exercising an ignored local candidate bundle. |
| [native runtime](native-runtime.md) | Pinned upstream models, ONNX export, and the browser parity contract. |
| [reproducibility](reproducibility.md) | What a fresh clone can and cannot rebuild. |
| [budget](budget.md) | The append-only ledger of approved ceilings and recorded charges. |
| [issue #1 outstanding gates](issue-1-evidence.md) | What issue #1 has not yet established. |

### Provenance and rights

| Document | What it is |
| --- | --- |
| [artifacts](provenance/artifacts.md) | Third-party browser artifact admission: pinned identities, licenses, notices. |
| [public dataset review](provenance/public-dataset-review.md) | Rights screen of candidate public sources and admitted printed pages. |
| [reuse](provenance/reuse.md) | Which chess-reader components were reused or reviewed, and on what basis. |

## Decisions — dated, immutable, superseded rather than edited

[decisions index](decisions/index.md) lists every recorded owner decision
newest-first with its status.

## Records — dated, immutable, nothing inside is current

[archive index](archive/index.md) lists every historical evidence log, run record
and post-mortem. Live documents may cite these but never depend on them.
