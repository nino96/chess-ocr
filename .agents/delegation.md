# Delegation and token-efficient execution

Harness orchestration policy for agents working in this repository. It governs
how work is split between a lead and its workers, not what the project builds;
the repository-wide rules are in [AGENTS.md](../AGENTS.md).

- The user selects the primary model. The lead retains consequential design,
  integration, review and final validation; do not lower acceptance criteria to
  save tokens. Put exact model/effort routing in personal runtime configuration,
  not shared repository policy.
- Delegate only when a bounded independent task is likely to save total work or
  provide necessary independent evidence. Keep small edits and tightly coupled
  reasoning local. Use a capable inexpensive worker for clear lookup/check tasks,
  a stronger coding worker for bounded implementation, and the lead or a stronger
  reviewer for ambiguity, experimental design and consequential correctness.
- State the chosen model/effort and reason briefly when delegating, using supported
  runtime controls. Supply only the objective, relevant files/instructions,
  decisions, owned paths, acceptance checks and a concise return format. Prefer
  a fresh bounded context over forking the full conversation. Do not reread or
  investigate the same material in both lead and worker without a review need.
- Workers do not spawn other agents. Reuse a suitable existing worker for a
  related follow-up; do not create a roster of speculative agents. Respect the
  personal concurrency cap. Escalate a concrete reasoning gap to the lead after
  one unsuccessful bounded attempt instead of cycling through cheap retries.
- One writer per file. Workers make no Git or branch changes without lead
  coordination, and delegation never widens authority.
- Read required governing documents once per task/context and reuse a compact
  evidence summary; reread changed or missing sections when necessary. Use targeted
  searches and bounded tool output. Return findings, file references, checks and
  blockers rather than transcripts or repeated plans.
- Keep reports concise and evidence-bearing. At handoff record the next action,
  changed hashes and unresolved blockers so work resumes without reconstructing
  the session. Report measured token/quota usage only when available; neither a
  model choice nor a concurrency cap guarantees a weekly allowance or savings.
