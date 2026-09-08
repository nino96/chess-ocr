# Issue #1 outstanding gates

What issue #1 has not yet established, and what a resuming session should do
first. The measurements that closed the rest of issue #1 are the record
[issue #1 implementation evidence — 2026-09-06](archive/issue-1-evidence-2026-09-06.md).

## Outstanding acceptance gates

- No named laptop, physical macOS/Windows host or iPad is available to this
  project. OS/device setup and laptop peak memory/runtime budgets are not
  claimed complete.
- The selected GB10 container passes these two native forward probes only with
  cuDNN disabled (MobileNet max abs 1.48e-5, YOLOX 5.82e-5). The default cuDNN
  path failed and remains recorded. Backward/optimizer/recovery validation is #3.
- CPU native lock is Linux ARM64-specific; it must not be copied to other platforms.
- Original repository source is licensed under MIT; third-party packages and
  model artifacts retain their separate licenses and notices. The npm package is
  private, and model release/publication is a separate decision.
- No real-data accuracy, qualified localization, skew support or superiority over
  FENShot is claimed. Dataset and substantive training stay in #2/#3.

## Resuming

Use [the resource ledger](budget.md), preserved local raw reports and their
`codeHashes` to reuse unchanged evidence. Next owner-dependent acceptance action:
run the runtime harness on the chosen laptop and set its prospective budgets;
schedule physical OS/device gates only when available. Package publication needs
the owner's source-license decision.

The raw report identities to match against those `codeHashes` are listed in
[the record](archive/issue-1-evidence-2026-09-06.md#raw-final-report-sha-256).
They are local-only evidence, not reproducible from this repository.
