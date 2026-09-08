# Training runbook

The GPU training controller: how to initialize, start, monitor and stop a run,
what the controller refuses, how retries and audits work, and the gates a run
must pass before and after optimization.

The controller is `pnpm run training`, a thin wrapper over
`python/training_job.py`. Subcommands are `init`, `start`, `stop`, `run`,
`status`, `export`, `audit` and `ledger`.

What a trained candidate may and may not be claimed to do is in
[scope and claims](scope-and-claims.md). Reservations and charges are in
[budget](budget.md).

## Boundaries and hypothesis

Hypothesis: task heads followed by lower-rate adaptation of the pinned native
checkpoints can learn the audited synthetic piece designs and page geometry well
enough to reduce proposal corrections on separately reviewed real pages.

Training the classifier and detector on the synthetic seed tests the available
positive, negative, multiple-board, small-board, affine and projective cases. It
does not establish real localization, because a synthetic seed lacks source
diversity and several target detector conditions.

All synthetic pages retain TRAIN purpose. A deterministic 85/10/5 internal
train/development/calibration partition keeps each page, repeated position
parent and shared degradation seed connected. It exists for optimization,
checkpoint selection and diagnostic calibration only, and is never
qualification.

Both board orientations appear in the seed, and every internal partition must
contain both. Metrics report the two orientations separately. Recognition stays
image-relative and orientation stays unknown until the user selects it, unless a
later orientation model is independently validated.

Each initialized run freezes the exact recipe values it read, so revising a
recipe does not alter an active or historical run. Failed and interrupted
attempts remain charged.

## GPU-seconds

The controller reports its reservation in GPU-seconds. **One GPU-second is one
second of wall time while a scheduled training container owns the GPU.** It is
not a count of optimizer updates.

Preflight, classifier and detector segments each hold a frozen allocation, and a
retry consumes the same segment allocation rather than creating a new one. The
default `status` view reports capacity, consumed and remaining seconds per
segment; `status --history` prints the complete attempt ledger.

The allocations themselves are recorded in [budget](budget.md).

## Initializing and running

From a clean branch based on merged `origin/main`, commit the reviewed training
code first, then initialize with explicit read-only roots for the corpus and the
admitted native artifacts:

```sh
pnpm run training -- init \
  --dataset-root /absolute/path/to/chess-ocr/work/dataset/synthetic \
  --native-root /absolute/path/to/chess-ocr \
  --overlay-root /absolute/path/to/ignored/training-overlay
pnpm run training -- start
pnpm run training -- status
pnpm run training -- stop
```

`init` rejects dirty code, stale corpus or model hashes, an unavailable pinned
container, unsafe paths, and insufficient free space.

Before allocating a GPU, `start` runs the frozen container's CPU-only
input/dependency/output validation. A failure is retained without a GPU charge.
`start` returns once the bounded background supervisor is live.

Checkpoints, curves, exports and raw logs stay under ignored `work/training/`.
Do not publish them without the separate artifact rights review this repository
requires.

### Run roots

Commands without `--run-root` always address the default
`work/training/synthetic-bootstrap-v1` directory. They do not discover the newest
repaired attempt.

If initialization used a different directory, pass that same `--run-root` to
every later `status`, `start`, `stop`, `export` and `audit` command. Throughout
this document `work/training/<run-name>` stands for the run root you chose.

### Detector-only run reusing a completed classifier

A detector-only run uses `recipes/synthetic-bootstrap-v2.json` and the original
COCO detector checkpoint, and reuses a completed classifier checkpoint without
charging classifier GPU time. Initialize it **without** `--prior-run`, which is
what gives it its own fresh frozen reservation while retaining the earlier run as
evidence:

```sh
pnpm run training -- init --detector-only \
  --recipe recipes/synthetic-bootstrap-v2.json \
  --classifier-checkpoint /absolute/path/to/<prior-run>/classifier/checkpoint-010000.pt \
  --dataset-root /absolute/path/to/chess-ocr/work/dataset/synthetic \
  --native-root /absolute/path/to/chess-ocr \
  --overlay-root /absolute/path/to/ignored/training-overlay \
  --run-root work/training/<run-name>
pnpm run training -- start --run-root work/training/<run-name>
pnpm run training -- status --run-root work/training/<run-name>
```

The controller verifies the checkpoint schema, the completed schedule, the
selected-candidate evidence and its hash; exports it with native-to-ONNX parity
before detector optimization; marks the classifier complete without classifier
GPU charges; and then runs the detector schedule. It reuses the retained
development evidence instead of repeating a full CPU evaluation. Choose the new
reservation in the reviewed recipe before initialization.

A repaired attempt that supplies `--prior-run` imports every prior attempt and
charge: a new run directory cannot reset a cumulative reservation.

### Export-only retry

If a completed detector run needs only its failed export retried, use the
export-only command. It does not repeat optimization, final evaluation or
calibration:

```sh
pnpm run training -- export --run-root work/training/<run-name> --segment detector
```

### Audit and ledger

A completed run is immutable. To recompute corrected synthetic
development/calibration metrics from its selected checkpoint and ONNX, without
optimization or lifecycle changes:

```sh
pnpm run training -- audit --run-root work/training/<run-name>
pnpm run training -- ledger --training-root work/training
```

The audit includes partial and unsupported inputs as no-valid-board cases, and
writes only ignored `audits/` evidence plus its log — local-only evidence, not
reproducible from this repository. The ledger deduplicates inherited attempt
histories across retained runs, because summing each run's charged seconds would
count the same work twice.

## Proposal threshold versus acceptance threshold

A synthetic calibration can conclude that **no** nonzero-recall threshold meets
its false-positive limit. The calibrated acceptance threshold is then `1.0`,
and a hash-bound bundle abstains in automatic mode.

That outcome is a refinement and rejection blocker. It is not evidence to relax
the threshold, sweep thresholds, retrain, or change models, and it is not
evidence that the model is useless.

A separate, much lower value such as `0.01` feeds region proposals to the
deterministic nine-line grid refiner. It is **not** an accepted-board threshold.
Any low detector threshold must be explicitly identified as a proposal threshold
before grid refinement, never as calibrated board acceptance. Both values are
labeled synthetic-development-only in a prepared bundle.

## Gates and stop conditions

Before substantive optimization, verify corpus and checkpoint hashes; actual
image, label and tensor ordering; finite gradients; expected parameter updates;
a tiny known-label fit; negative-page loss; native preprocessing; and actual
stochastic checkpoint recovery.

Use the digest-pinned NVIDIA container with cuDNN disabled — its default cuDNN
path fails native parity. If backward, resume, deterministic execution, or the
complete schedule's measured projection fails, stop before the run and report the
specific blocker.

The container runs as the invoking host UID/GID, writes only through `/output`,
and resolves NumPy 2.2.4, Pillow 11.1.0, ONNX 1.17.0, ONNX Runtime 1.20.1,
OpenCV headless 4.11.0.86, safetensors 0.5.3 and timm 1.0.15 from a hash-frozen
ignored overlay built from the local wheelhouse.

### Preflight validation segment

Before `start` can allocate a GPU, the exact frozen, network-disabled container
runs a CPU-only validation segment. It verifies dependency pins, output writes,
classifier tensors and labels, and detector preprocessing. The detector check
compares an OpenCV BGR source against the pinned upstream helper using
non-square geometry, colored channel sentinels, actual ImageNet RGB mean and
standard deviation, and a maximum tensor difference of `1e-6`. A self-consistent
check that is not checkpoint-compatible does not satisfy this gate.

Failure leaves a retained validation log and consumes no GPU time. Timeout
cleanup stops the named validation container before returning control.

### Preprocessing identity

The detector graph boundary is raw letterboxed RGB float32 `0..255` with pad
`114`; the graph wrapper divides by 255, subtracts the ImageNet RGB mean, and
divides by the ImageNet RGB standard deviation. Training feeds the equivalent
normalized RGB tensor directly. Native export and browser preprocessing must
agree on that identifier.

### Optimization guards

The detector recipe clips its head and full-model gradient norm at 10.0. This is
a mechanics-stability guard: the configured 0.005 head learning rate produces
non-finite gradients without clipping, while the clipped path stays finite and
reduces loss. Adding the guard changes no seed, model family, corpus or schedule
length.

The schedule is complete only after 10,000 classifier and 9,000 detector updates
plus all frozen development and calibration evaluations. Do not shorten it in
order to call a mechanics pilot successful. One failed comparison permits one
bounded causal diagnosis — not a new seed, resolution, model family or sweep.

### Reporting

Report full curves and the effective class, design and effect exposure. For the
classifier, report exact boards, square confusion, occupied/class/color errors,
NLL, confidence coverage and confident errors. For the detector, report
multi-board recall, negative false positives, AP, IoU, and normalized box error
by size, layout and effect. Detector average precision is clamped to the valid
zero-to-one range; partial and unsupported pages count as pages with no valid
complete board, with false detections on those pages reported separately.
Compare the selected candidates with unchanged FENShot on identical inputs.

## Verifying the controller and its outputs

The audit path cannot train or alter checkpoints. To verify retained artifacts
end to end, prepare the hash-bound bundle and run the actual models through Node
WASM and every production browser engine — see
[local candidate testing](local-candidate.md) for the bundle preparation and
verification commands, and [architecture](architecture.md) for what the shared
runtime does with them.

## Delivery after the bootstrap

A candidate may prefill issue #2 reviews only as a visible proposal. The next
promotion decision requires separately reviewed real development data, the
bounded classical detector comparison, shared inner-grid refinement, and paired
end-to-end evidence. Fresh human-checked qualification stays untouched until
candidate freeze. FENShot remains the shipped default until all issue #3
quality, offline WASM, runtime and browser gates pass.

Checkpoints, ONNX files and raw evidence remain ignored. Model publication and a
public artifact mechanism require separate rights review and owner approval, as
recorded in [scope and claims](scope-and-claims.md#local-evaluation-only).
