# EMEL

## What This Is

EMEL is a deterministic C++ inference engine built around explicit Stateforward.SML orchestration, with
runtime behavior modeled as explicit actors instead of ad hoc control flow. The repo now ships
maintained GGUF generation slices, one explicit maintained trimodal embedding slice for
`augmem/TE-75M-GGUF`, one maintained Sortformer diarization GGUF slice, and one maintained Whisper
ASR GGUF slice, along with pluggable parity and benchmark tooling that compares EMEL against
external reference engines without sharing runtime state.

## Core Value

Prove real end-to-end behavior with explicit SML orchestration and parity-oriented verification
before widening API surface or model scope.

## Current State

Current milestone: none open.

Latest shipped milestone: `v1.23 I/O Loading Strategy Boundary`

Status: `v1.23` shipped on 2026-05-04 and is open for review in PR #82. The repo now treats
`src/emel/io` as the loading strategy boundary owner, `src/emel/model/tensor` as the canonical
owner of tensor load, bind, evict, and residency semantics, and `model/loader` as the orchestrator
across those public actor surfaces.

Current planning focus: start the next milestone with `$gsd-new-milestone` after PR #82.

## Previous Shipped Milestone: v1.23 I/O Loading Strategy Boundary

**Goal:** Add `src/emel/io` as the first-class owner of loading strategy and transport boundaries,
then wire `model/tensor` to an explicit I/O contract without moving tensor residency semantics out
of `model/tensor` or low-level byte strategy into `model/loader`.

**Source:** GitHub issue #60, "Add emel/io module and tensor-to-io orchestration boundary"

**Shipped:** 2026-05-04

**Delivered:**
- Added an `src/emel/io` module with Stateforward.SML actor organization and public component
  aliases that match existing EMEL machine conventions.
- Defined explicit tensor-to-I/O request, result, and error events for loading strategy handoff
  without hidden shared state.
- Allowed tensor-owned load flow to target an I/O strategy boundary while `model/tensor` remains the
  residency lifecycle owner.
- Modeled strategy policy injection and behavior selection with guards and transitions so future
  concrete strategies can land independently.
- Added tests, docs, and source-backed guardrails that prevent `model/loader` from regaining
  low-level loading strategy ownership.

**Validation:** Focused model/tensor/IO tests, domain-boundary guardrails, changed-file coverage,
lint snapshots, benchmark snapshots, docs generation, and the changed-file scoped quality gate
passed on 2026-05-04. Concrete mmap/read/copy/device/async strategies remain deferred.

**Audit:** Source-backed audit passed with 14/14 active requirements satisfied after the delegated
final source audit returned no blockers.

## Previous Shipped Milestone: v1.22 Weight Loading Ownership Cutover

**Goal:** Make `src/emel/model/tensor` the canonical owner of tensor load, bind, evict, and
residency transitions while removing `src/emel/model/weight_loader` from the primary runtime load
path.

**Source:** GitHub issue #59, "Cut over weight loading ownership from model/weight_loader to
model/tensor"

**Shipped:** 2026-05-03

**Delivered:**
- Made `src/emel/model/tensor` the canonical owner of tensor load and residency behavior.
- Rewired `model/loader` to coordinate tensor-owned bind, plan, and apply behavior through public
  tensor actor events.
- Removed the retired `model/weight_loader` source/test path from maintained runtime ownership.
- Added explicit loader tensor outcome decision states so tensor bulk success and failure no longer
  depend on local callback flag capture.
- Added guardrails and public documentation that prevent stale retired-owner prose and keep concrete
  I/O strategy implementation deferred under the future `emel/io` seam.
- Moved maintained generation benchmark, Sortformer benchmark, embedded-size probe, and paritychecker
  GGUF KV storage growth before model-loader dispatch.

**Audit:** Source-backed audit passed with 14/14 active requirements satisfied after Phase 194
closed the maintained loader tool prebind gap.

## Previous Shipped Milestone: v1.21 Quality Gate Selective Runner Optimization

**Goal:** Teach the required quality gate to use manifest-backed parity/benchmark impact
selection and safe parallelism so localized changes do less unnecessary work without weakening
coverage guarantees, lane isolation, or failure reporting.

**Source:** GitHub issue #58, "Optimize quality gates after #54 and #55 land"

**Shipped:** 2026-05-02

**Delivered:**
- Kept `scripts/quality_gates.sh` as the mandatory top-level gate entrypoint for implementation
  changes.
- Added manifest-backed parity runner impact selection from `parity_dependency_manifest/v1` data.
- Preserved manifest-backed benchmark runner impact selection from `bench_dependency_manifest/v1`
  data.
- Kept conservative full relevant fallback for missing, stale, uncertain, or failed manifest
  resolution.
- Added selected parity execution through `scripts/paritychecker.sh --runner=<name>` and retained
  selected benchmark execution through `scripts/bench.sh --suite=<runner>`.
- Parallelized independent benchmark, coverage, parity, and fuzz lanes after serial preflight and
  build.
- Added source-backed quality-gate regression coverage and closeout evidence showing a 508 second
  full scoped gate and a 19 second representative selective gate.

**Audit:** Source-backed audit passed with 14/14 active requirements satisfied.

## Previous Shipped Milestone: v1.20 SML Dependency And Namespace Migration

**Goal:** Upgrade EMEL to the current `stateforward/sml.cpp` dependency and migrate
project-owned code/docs from the legacy SML surface to `stateforward::sml` without weakening
actor-model rules or maintained parity/benchmark evidence.

**Source:** GitHub issue #56, hard cutover to the latest `stateforward/sml.cpp` surface.

**Shipped:** 2026-05-02

**Delivered:**
- Pinned the intended upstream `stateforward/sml.cpp` commit and documented dependency provenance.
- Migrated active project-owned source, tests, tools, examples, and docs to the preferred
  `stateforward` include and namespace surface.
- Proved transition tables, completion/internal transitions, unexpected-event behavior, logger
  wiring, dispatch tables, and state inspection through the migrated surface.
- Repaired contributor rules and documentation so active guidance points to
  `docs/rules/sml.rules.md` and `stateforward::sml`.
- Added maintained guardrails for unapproved active legacy SML references.
- Repaired closeout reproducibility and passed the final full quality gate with coverage above the
  required threshold.

**Audit:** Final source-backed audit passed with 12/12 active requirements satisfied after
Phase 179.

## Previous Shipped Milestone: v1.19 Benchmark Tool Pluggable Runner Refactor

**Goal:** Refactor `tools/bench` so shared benchmark orchestration owns config, asset, and report
normalization while benchmark-family execution lives behind pluggable runner boundaries that can
be built and gated independently.

**Source:** GitHub issue #55, "Refactor benchmark tool for cleaner boundaries and pluggable
runners"

**Shipped:** 2026-05-01

**Delivered:**
- Shared `tools/bench` orchestration behind `emel::bench::run_bench_cli(...)`, with
  `bench_main.cpp` reduced to a process shim.
- Deterministic `bench_runner_request/v1` and `bench_runner_result/v1` contracts for the
  process-level runner seam.
- Localized benchmark suite metadata and lookup in `bench_runner_registry.hpp` / `.cpp`.
- Per-suite `bench_runner_suite_<suite>` CMake object targets for selected maintained runner
  sources.
- Deterministic `bench_dependency_manifest/v1` records, checked-in manifest baseline,
  write/check CLI operations, and conservative quality-gate manifest consumption.
- Source-backed maintained generation and diarization behavior checks plus shared
  runner/orchestrator lane-isolation checks.

**Audit:** Final source-backed audit passed with 13/13 active requirements satisfied.

## Previous Shipped Milestone: v1.18 Parity Tool Boundary Refactor

**Goal:** Refactor `tools/paritychecker` into explicit runner, engine, asset-loading, and
dependency-manifest boundaries so new parity engines can be added locally without weakening
EMEL/reference lane isolation or existing parity behavior.

**Initial closeout:** 2026-05-01 from GitHub issue #54
**Shipped:** 2026-05-01 after reopened source-backed gap closure

**Delivered:**
- Runner-owned path, byte-loading, and maintained generation fixture resolution helpers.
- Explicit tokenizer, GBNF, kernel, Jinja, and generation parity engine adapters behind a small
  runner-facing registration surface.
- Modular `tools/paritychecker` CMake source groups shared by the executable and tests.
- `parity_dependency_manifest/v1` records with conservative missing/stale/uncertain full-gate
  semantics.
- Source-backed behavior and lane-isolation closure proving shared runner files stay free of
  lane-owned runtime objects while maintained parity tests continue to pass.
- Runner-owned CLI/config parsing through `run_parity_cli(...)`.
- Maintained generation comparison against live reference-lane output before baseline load; stored
  baselines are publication artifacts, and legacy non-current Qwen drift is reported truthfully.
- Paritychecker source ownership now goes through public GGUF/model/llama wrapper surfaces instead
  of non-kernel actor/detail includes, with broad source checks guarding the boundary.
- Maintained dependency-manifest CLI emission/check operations and quality-gate freshness
  escalation now force full parity when manifest data is missing, stale, or uncertain.

**Audit:** Final source-backed audit passed with 12/12 active requirements satisfied.

## Previous Shipped Milestone: v1.17 Text Generator Domain Alignment

**Goal:** Move the canonical generative text actor to `text/generator` so generation ownership
matches the text-domain layout and aligns with the existing `embeddings/generator` actor.

**Shipped:** 2026-04-30 after Phase 147

**Delivered:**
- Established `src/emel/text/generator/**`, `emel/text/generator/**`, and
  `emel::text::generator::sm` as the canonical generative text actor ownership contract.
- Moved the generator parent actor, initializer, prefill child, tests, CMake wiring, and public
  aliases under text-domain ownership.
- Repaired generator wrapper SML rule violations so runtime choices stay in explicit
  guard/transition orchestration.
- Preserved maintained generation parity and benchmark proof while removing paritychecker and
  benchmark dependencies on text-generator actor internals.
- Added a public `event::capture_diagnostics` actor event for source-backed diagnostics evidence.
- Closed reopened audit gaps with public graph lifecycle diagnostics, kernel-owned row-storage
  sizing, stronger domain-boundary checks, explicit generator test-surface classification,
  guard-owned route support, native-quantized route evidence, explicit compute outcome modeling,
  and guard-accepted graph validation/bind/extract callbacks.

## Previous Shipped Milestone: v1.16 ARM Whisper GGUF Parity And Performance

**Goal:** Bring up one truthful maintained ARM Whisper tiny ASR GGUF slice through speech-owned
runtime actors, exact transcript parity, and matched single-thread benchmark proof.

**Shipped:** 2026-04-28

**Delivered:**
- Removed the top-level Whisper runtime domain; Whisper runtime actors now live under speech
  encoder/decoder/tokenizer ownership while model binding stays in `model/whisper`.
- Pinned and validated the maintained Whisper tokenizer/model/audio contract before dispatch.
- Published exact transcript parity against the pinned `whisper.cpp` lane through the public
  recognizer route.
- Replaced hidden recognizer backend dispatch with explicit SML route states, transitions, guards,
  and compile-time route policy wiring.
- Removed decoder production dependency on encoder-owned Whisper detail helpers.
- Published source-backed ARM single-thread evidence where EMEL beats the matched `whisper.cpp`
  reference lane.

## Previous Shipped Milestone: v1.15 ARM Sortformer Diarization GGUF Slice

**Goal:** Bring up one truthful maintained ARM diarization slice for
`openresearchtools/diar_streaming_sortformer_4spk-v2.1-gguf`, using EMEL-owned loading,
execution, output decoding, parity proof, and benchmark publication.

**Shipped:** 2026-04-25

**Delivered:**
- Pinned Sortformer GGUF fixture contract with source URL, checksum/provenance, and architecture
  metadata truth.
- Explicit Sortformer model contract and GGUF acceptance/rejection behavior.
- Diarization request surface for deterministic mono `float32` PCM at 16,000 Hz.
- Native ARM Sortformer runtime path in `src/`, with no tool-only compute fallback.
- Deterministic `T x 4` speaker-activity probabilities and bounded `speaker_0` through
  `speaker_3` segment output.
- Lane-isolated PyTorch/NeMo parity proof, ONNX benchmark reference proof, and one maintained ARM
  benchmark publication.
- Strict Phase 93 generated record where EMEL exact-matches PyTorch/NeMo and ONNX on
  `output_dim=17`, checksum `4249677247906920305`, while beating ONNX CPU single-thread:
  EMEL `1352780166 ns/op` versus ONNX `1920646958 ns/op`.

## Previous Shipped Milestone: v1.14 Benchmark Variant Organization

**Shipped:** 2026-04-21

**Delivered:**
- Shared benchmark manifest discovery and duplicate-ID validation helpers
- Deterministic generation workload discovery from checked-in manifests
- Deterministic embedding variant discovery from checked-in manifests
- Aligned operator selectors for generation workload IDs and embedding variant IDs
- Documentation and focused regressions proving ordinary variant additions stay data-only

## Previous Shipped Milestone: v1.13 Pluggable Generative Parity Bench

**Shipped:** 2026-04-21

**Delivered:**
- Canonical `generation_compare/v1` contract for EMEL and reference lanes
- Explicit workload manifests for prompts, formatter mode, seeds, sampling, and stop conditions
- Maintained `llama_cpp_generation` backend integration on one operator-facing compare workflow
- Truthful compare verdicts and publication artifacts for reproducible cross-engine review
- Maintained single-lane publication that reports `non_comparable` instead of making a false
  parity claim

## Previous Shipped Milestone: v1.12 Pluggable Reference Parity Bench Architecture

**Shipped:** 2026-04-18

**Delivered:**
- One canonical `embedding_compare/v1` contract that keeps the EMEL lane unchanged while reference
  backend choice stays in manifest/tooling space.
- Maintained Python and C++ reference engines running through one operator-facing compare
  workflow and output schema.
- Truthful repaired C++ compare publication that preserves both maintained baseline records for the
  `liquid_cpp` text workflow.
- Refreshed requirement-traceability and Nyquist evidence so `v1.12` can close with a passing
  milestone audit.

## Previous Shipped Milestone: v1.11 TE-75M GGUF Trimodal Embedding Runtime

<details>
<summary>Shipped on 2026-04-15</summary>

**Delivered:**
- One maintained TE fixture pinned at `tests/models/TE-75M-q8_0.gguf` with explicit provenance
  and checksum tracking.
- Truthful `omniembed` model-family support instead of aliasing TE onto an existing LLM runtime.
- Explicit text, vision, and audio embedding lanes with one shared normalized embedding contract
  and supported Matryoshka truncation.
- Stored upstream-golden proof plus tiny cross-modal smoke checks integrated into the normal repo
  gate flow.

</details>

## Historical Open Closeout: v1.10 Planner Family AGENTS Hard Cutover

<details>
<summary>Implementation complete on 2026-04-05; closeout still pending</summary>

**Goal:** Hard-cut `src/emel/batch/planner` and its child planner-mode submachines over to the
`AGENTS.md` naming, layout, and SML orchestration contract without broadening the change into
unrelated machine families.

**Delivered before closeout:**
- Renamed and reorganized the planner-family files, aliases, events, and states into the canonical
  planner-owned surface.
- Brought planner and planner-mode machines into rule compliance for destination-first transitions,
  event naming, and persistent-state ownership.
- Preserved maintained batching behavior with focused planner-family proof.
- Kept the work bounded to `src/emel/batch/planner` and its child modes.

</details>

## Previous Shipped Milestone: v1.9 Liquid LFM2.5-1.2B Thinking ARM Slice

<details>
<summary>Shipped on 2026-04-02</summary>

**Goal:** Prove one truthful maintained LiquidAI `LFM2.5-1.2B-Thinking-GGUF` ARM slice through the
existing EMEL generator, paritychecker, and benchmark workflow, with `Q4_K_M` as the maintained
truth anchor and without broadening into generic Liquid-family support.

**Delivered:**
- One official `LFM2.5-1.2B-Thinking-Q4_K_M.gguf` fixture with reproducible provenance under
  `tests/models/`
- One explicit Liquid request-conditioning contract derived from the official primary chat template
- Explicit `lfm2` runtime bring-up on the maintained generator path
- Maintained parity, regression protection, and benchmark publication for the same slice

</details>

## Requirements

### Active

- v1.22 cuts weight-loading ownership from `src/emel/model/weight_loader` into
  `src/emel/model/tensor` while preserving existing model-loading behavior.
- v1.22 updates `src/emel/model/loader` so bulk model loading orchestrates tensor-owned behavior
  instead of treating `load_weights_fn` as the long-term architecture seam.
- v1.22 retires or explicitly bounds the old `model/weight_loader` path so the codebase does not
  retain a second tensor-residency owner under a new name.
- v1.22 prepares, but does not implement, the future `emel/io` strategy layer below tensor
  ownership.

### Validated

- ✓ v1.21 keeps `scripts/quality_gates.sh` mandatory while selecting only impacted parity and
  benchmark runners when dependency-manifest evidence is trustworthy.
- ✓ v1.21 fails closed to the affected full relevant parity or benchmark runner set when manifest
  impact resolution is missing, stale, uncertain, malformed, or failed.
- ✓ v1.21 runs independent heavy quality-gate lanes in safe parallel groups with ordered logs,
  deterministic status reporting, and timing output.
- ✓ v1.21 closed with source-backed evidence for focused regressions, a full scoped gate, and a
  representative selective-runner speedup without weakening required validation lanes.
- ✓ v1.20 upgraded EMEL to the current `stateforward/sml.cpp` dependency surface and migrated
  active project-owned code/docs to `stateforward::sml` naming.
- ✓ v1.20 proved the migrated SML orchestration surfaces with live behavior tests and maintained
  legacy-reference guardrails.
- ✓ v1.20 preserved actor-model rules, maintained runtime behavior, and quality-gate evidence
  through a passing source-backed closeout audit.
- ✓ v1.19 refactored `tools/bench` around shared CLI/config/report orchestration while keeping
  benchmark-family execution behind explicit runner boundaries.
- ✓ v1.19 added localized benchmark runner discovery, registration, and per-suite build targets so
  maintained runner additions no longer require broad static orchestrator wiring.
- ✓ v1.19 added deterministic benchmark dependency manifests and conservative quality-gate
  fallback behavior for missing, stale, or uncertain manifest data.
- ✓ v1.19 preserved maintained generation and diarization benchmark behavior while adding
  source-backed checks for shared runner lane isolation and actor-boundary cleanliness.
- ✓ v1.18 refactored `tools/paritychecker` so shared runner orchestration owns asset/config
  loading, CLI parsing, and result normalization while mode-specific parity behavior lives behind
  explicit engine adapters.
- ✓ v1.18 preserves existing tokenizer, GBNF, kernel, Jinja, and generation parity behavior while
  maintaining EMEL/reference model, tokenizer, runtime, cache, and output ownership isolation by
  lane.
- ✓ v1.18 adds maintained per-runner dependency manifest emission/checking and conservative
  quality-gate escalation when manifest data is missing, stale, or uncertain.
- ✓ v1.0 proved one canonical `Llama-68M-Chat-v1-Q2_K.gguf` slice end to end through
  `tools/paritychecker` with real GGUF/model loading, bounded generation, and subprocess coverage.
- ✓ v1.1 added one truthful canonical generation benchmark in `tools/bench`, published through the
  existing compare, snapshot, and docsgen workflow.
- ✓ v1.2 shipped an EMEL-owned flash-attention path plus hard cutover to `emel::tensor::sm` on the
  canonical Llama generation slice.
- ✓ v1.3 shipped optimized AArch64 flash execution and maintained optimized-vs-shared attribution
  on the canonical ARM workload.
- ✓ v1.4 shipped EMEL-owned vectorized AArch64 `q2_K/q3_K/q6_K x q8_K` hot-path kernels and
  maintained `1/10/100/1000` parity on the canonical ARM slice.
- ✓ v1.5 closed the canonical ARM quantized-path contract at
  `native_quantized=8 approved_dense_f32_by_contract=4 disallowed_fallback=0 explicit_no_claim=0`
  and restored checked-in flash publication so stored evidence matches live proof.
- ✓ v1.6 documented one official canonical `Qwen3-0.6B-Q8_0.gguf` fixture and bound the maintained
  Qwen path to one explicit GGUF-derived conditioning contract with structured chat-message input
  and no implicit raw fallback.
- ✓ v1.6 added native `src/emel` `q8_0` runtime support plus explicit canonical `qwen3` topology
  handling on the shipped generator path.
- ✓ v1.6 proved maintained canonical Qwen parity against `llama.cpp` and protected the prior
  Llama anchor with stored compare coverage on `1/10/100/1000`, `--dump`, and `--attribution`.
- ✓ v1.6 published one truthful canonical Qwen benchmark compare/docs path aligned with the same
  parity-backed formatter/runtime contract.
- ✓ v1.7 extracted an explicit `src/emel/generator/prefill` orchestration machine for prefill
  slots, snapshot, compute dispatch, and handoff.
- ✓ v1.7 kept prefill request-scoped orchestration data on typed runtime/internal events instead of
  generator context phase fields.
- ✓ v1.7 collapsed the old top-level prefill routing matrix into explicit request-scoped prefill
  compute contracts.
- ✓ v1.8 published one truthful maintained Qwen3 executable-size comparison on final linked
  executables for the canonical `hello` -> first-token path.
- ✓ v1.9 documented one official maintained `LFM2.5-1.2B-Thinking-Q4_K_M.gguf` fixture and bound
  the maintained Liquid path to explicit executable metadata truth plus one explicit structured
  chat contract.
- ✓ v1.9 added explicit `lfm2` model/runtime support for one maintained Liquid ARM slice through
  the shipped generator path.
- ✓ v1.9 proved maintained Liquid parity against `llama.cpp`, preserved additive maintained Qwen
  coverage, and published one truthful Liquid benchmark/docs path aligned with the parity-backed
  slice.
- ✓ v1.9 repaired its missing closeout artifacts, validation layer, and milestone bookkeeping so
  the milestone could pass audit and archive cleanly.
- ✓ v1.11 Phase 49 added a repo-owned `embeddings/generator` actor that initializes through the
  maintained conditioner/tokenizer seam and returns normalized TE text embeddings on
  `TE-75M-q8_0.gguf`.
- ✓ v1.11 Phase 49 proved supported Matryoshka truncation at `768/512/256/128`, explicit invalid
  dimension rejection, and callback/error coverage while keeping global quality gates green at
  `90.2%` line coverage.
- ✓ v1.11 Phase 50 derives the maintained TE image contract from the declared
  `mobilenetv4_conv_medium.e180_r384_in12k` encoder family and runs the real `image_encoder.*`
  convolutional tower plus shared `image_projection.*` head natively in `src/`.
- ✓ v1.11 Phase 50 proves normalized `1280`-dimensional image embeddings, supported truncation,
  and explicit malformed-image rejection on the maintained in-memory RGBA path while keeping
  repo-wide quality gates green.
- ✓ v1.11 Phase 51 derives the maintained TE audio frontend from the declared
  `efficientat_mn20_as` encoder family and runs the real `audio_encoder.features.*` tower plus
  shared `audio_projection.*` head natively in `src/`.
- ✓ v1.11 Phase 51 proves normalized `1280`-dimensional audio embeddings, supported truncation,
  and explicit malformed-audio rejection on the maintained mono PCM contract while keeping
  repo-wide quality gates green.
- ✓ v1.11 Phase 52 proves text, image, and audio all land on one deterministic
  `embeddings/generator` result contract with shared normalization, shared truncation behavior,
  and uniform invalid-dimension rejection while keeping repo-wide quality gates green.
- ✓ v1.11 Phase 53 proves the maintained TE anchors against stored upstream text/image/audio
  golden vectors, keeps tiny cross-modal smoke checks in repo-owned doctests, and preserves
  WordPiece parity across the TE and BERT GGUF vocab surfaces while repo-wide quality gates stay
  green.
- ✓ v1.12 added one canonical pluggable reference-backend contract so the EMEL-owned compare lane
  stays unchanged while Python and C++ reference engines emit the same compare record schema.
- ✓ v1.12 brought maintained Python and C++ reference backends up through one operator-facing
  compare workflow with explicit backend identity, fixture identity, and reproducibility metadata.
- ✓ v1.12 repaired the lossy multi-record C++ publication path and backfilled the missing
  traceability / Nyquist closeout evidence so the rerun milestone audit passed.
- ✓ v1.13 added one canonical `generation_compare/v1` contract, manifest-pinned generation
  workloads, a maintained `llama_cpp_generation` reference backend, one operator-facing
  generation compare workflow, and closeout-proof regression coverage for the maintained LFM2
  compare slice.
- ✓ v1.13 repaired EMEL/reference JSONL lane isolation, added real selected single-lane
  non-comparable publication, and backfilled requirement/Nyquist evidence for a no-blocker audit.
- ✓ v1.14 added a shared benchmark variant registry contract with deterministic ordering and
  hard-fail duplicate-ID validation.
- ✓ v1.14 cut generation benchmarks over to manifest discovery so maintained workload additions
  are data-owned.
- ✓ v1.14 cut embedding benchmarks over to variant discovery so maintained embedding cases are
  data-owned across EMEL and Python-golden lanes.
- ✓ v1.14 proved and documented the ordinary data-only add path for both benchmark families.
- ✓ v1.15 pinned one maintained Sortformer diarization GGUF fixture and accepted only its explicit
  model/tensor/profile contract.
- ✓ v1.15 added a diarization-owned mono 16 kHz PCM request surface, native feature preparation,
  native encoder/executor execution, and deterministic probability/segment output.
- ✓ v1.15 proved EMEL against lane-isolated PyTorch/NeMo and ONNX references on the maintained
  fixture and closed the performance contract by beating ONNX Runtime CPU single-thread in the
  strict Phase 93 generated record.
- ✓ v1.16 moved Whisper runtime ownership out of top-level model-family domains and into
  speech-owned encoder, decoder, tokenizer, and recognizer actors.
- ✓ v1.16 proved exact recognizer-backed Whisper transcript parity against the pinned
  `whisper.cpp` lane and kept EMEL/reference runtime state isolated.
- ✓ v1.16 repaired hidden recognizer route dispatch, decoder ownership leaks, and benchmark
  evidence stability before merge.
- ✓ v1.17 moved the canonical generative text actor, initializer, and prefill child machines under
  `src/emel/text/generator/**` with canonical namespace `emel::text::generator`.
- ✓ v1.17 updated tests, tools, benchmark/parity surfaces, and domain-boundary checks so no stale
  top-level generator ownership remains.
- ✓ v1.17 closed source-backed SML/detail/test-surface audit gaps with public lifecycle events,
  kernel-owned row sizing, component-private test classification, explicit route/outcome modeling,
  and guard-accepted graph validation/bind/extract callbacks.

### Out of Scope

- New public embedding C ABI or broad CLI API commitments
- Remote HTTP or service-hosted reference engines
- Public plugin SDK or third-party backend distribution outside the repo
- Broad new `src/` runtime support added only to satisfy a reference backend
- Shared model, tokenizer, cache, or runtime objects between the EMEL lane and any reference lane
- New model/runtime support added solely to demonstrate the benchmark registry
- Broader performance tuning beyond the shipped maintained Sortformer ONNX single-thread closure
- Decode extraction, sampler redesign, new generation model support, or public generation C ABI
  expansion during the parity tool boundary refactor
- New benchmark semantics, new benchmark families, or improved performance claims solely to prove
  runner pluggability
- A public third-party benchmark plugin SDK or distribution format beyond a repo-owned runner
  contract
- Hash-only quality-gate skipping that bypasses conservative build-graph dependency manifests

## Context

This remains a brownfield repository with an existing codebase map under `.planning/codebase/`.
The repo stays governed by `AGENTS.md` and `docs/rules/sml.rules.md`. `v1.21` is the latest
shipped milestone, optimizing the mandatory quality gate with manifest-backed selective runners,
conservative fallback, and parallel lane reporting. The current maintained state includes repo-owned
EMEL generation, embedding, diarization, and Whisper ASR lanes plus pluggable parity and benchmark
tooling that publishes through canonical compare/benchmark contracts without shared runtime state.
`v1.18` and `v1.19` provide the parity and benchmark dependency manifests that v1.21 now consumes
from the top-level quality-gate orchestration. `v1.21` shipped from issue #58 and did not weaken
mandatory validation or change benchmark/parity semantics. `v1.22` shipped from issue #59 and made
`model/tensor` the canonical owner of tensor load, bind, evict, and residency behavior while
keeping concrete I/O strategy work deferred. `v1.23` shipped from issue #60 and added the missing
`emel/io` orchestration boundary under tensor-owned residency without implementing concrete mmap,
read/copy, staged, chunked, or asynchronous strategy machines.

## Constraints

- **Architecture**: Follow `docs/rules/sml.rules.md` and `AGENTS.md`; keep runtime behavior choice
  explicit and avoid hiding route selection in helper branching.
- **Isolation**: Keep the EMEL lane repo-owned and separate from all reference-engine runtime
  state, objects, and execution dependencies.
- **Reproducibility**: Preserve truthful compare artifacts with backend identity, fixture
  identity, and enough metadata to reproduce results.
- **Comparability**: Record formatter, tokenization, seed, and sampling metadata explicitly so
  benchmark or parity claims never hide apples-to-oranges drift.
- **Lifecycle**: Keep active milestone requirements in `.planning/REQUIREMENTS.md` when a
  milestone is open, and archive shipped requirements under `.planning/milestones/`.
- **Diarization scope**: Any future diarization widening must start from a new milestone and keep
  the shipped `v1.15` Sortformer slice truthful before broadening into general speech, ASR,
  streaming-service, or media-ingestion support.
- **Text generator scope**: Moving `generator` to `text/generator` must be an ownership refactor;
  it must not introduce new sampling semantics, model-family support, or performance claims.
- **Benchmark refactor scope**: `v1.19` moves `tools/bench` boundaries only; it must preserve
  maintained benchmark intent unless a behavior change is separately approved.
- **SML migration scope**: `v1.20` must preserve the RTC actor model and no-queue invariant while
  changing dependency pins, includes, namespaces, documentation, and checks. It must not use the
  namespace migration as a reason to broaden runtime semantics.
- **Quality gate optimization scope**: `v1.21` may reduce unnecessary parity and benchmark runner
  execution only through conservative dependency-manifest impact resolution inside the mandatory
  gate. It must not introduce permissive skips, hash-only trust, hidden runner suppression, or
  unclear failure logs.
- **Weight-loading ownership scope**: `v1.22` is an ownership and orchestration cutover. It must not
  introduce asynchronous loading, a new `emel/io` implementation, backend-specific loading logic in
  `model/loader`, or a renamed shadow owner for tensor residency.
- **I/O boundary scope**: `v1.23` creates the `emel/io` module and tensor-to-I/O contract only. It
  must not implement concrete mmap, read/copy, staged/chunked, device-specific, or cooperative async
  loading strategies; those belong in follow-on milestones such as issue #61.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Start v1.23 from GitHub issue #60 as the `emel/io` boundary milestone | v1.22 moved tensor residency ownership into `model/tensor`; the next architecture step is the explicit I/O strategy seam beneath tensor-owned residency before concrete mmap or staged strategy work lands | ✓ Shipped |
| Start v1.22 from GitHub issue #59 as the weight-loading ownership cutover | `model/tensor` owns individual tensor lifecycle state while `model/weight_loader` still owns bulk residency transition planning; the next runtime architecture milestone should remove that split before adding future I/O strategy work | ✓ Shipped |
| Start v1.21 from GitHub issue #58 as quality-gate selective runner optimization | v1.18 and v1.19 added parity and benchmark dependency manifests; the next milestone should cash in that structure at the mandatory gate-orchestration level without weakening conservative fallback behavior | ✓ Shipped |
| Start v1.20 from GitHub issue #56 as an SML dependency and namespace migration | The repo already sources `stateforward/sml.cpp` but still used legacy SML includes, namespaces, and contributor guidance; the next milestone should align code and docs with the current upstream naming before more actor work accumulates on the old surface | ✓ Shipped |
| Start v1.19 from GitHub issue #55 as a benchmark runner boundary refactor | `tools/bench` has accumulated broad runner build wiring and static case registration; the next milestone should make benchmark runners pluggable, independently buildable, and conservatively gateable without weakening lane isolation | ✓ Shipped |
| Start v1.18 from GitHub issue #54 as a paritychecker boundary refactor | The existing parity tool has grown mode-specific implementation, asset loading, and build wiring in places that make future engines harder to add and harder to gate conservatively | ✓ Shipped |
| Move the canonical generator actor to `text/generator` in v1.17 | The existing generator is a generative text actor, and placing it under the text domain aligns it with text tokenizer/formatter ownership and the established `embeddings/generator` top-level actor pattern | ✓ Implementation complete |
| Start v1.15 as one maintained ARM Sortformer diarization GGUF slice | The user asked for ARM support for `openresearchtools/diar_streaming_sortformer_4spk-v2.1-gguf`; the model card defines a diarization contract, not generation or embedding behavior | ✓ Shipped |
| Require native EMEL-owned Sortformer execution for the EMEL lane | Tool-only Python, ONNX, NeMo, llama.cpp, or ggml compute would not satisfy ARM support in `src/` and would violate lane-isolation expectations | ✓ Shipped |
| Keep v1.15 input/output contracts narrow | Mono 16 kHz PCM, `T x 4` probabilities, and bounded four-speaker segments are enough to prove this slice without committing media decode, ASR, or broad streaming APIs | ✓ Shipped |
| Use PyTorch/NeMo as parity reference and ONNX as benchmark reference for Sortformer | PyTorch/NeMo is the model-card reference lane, while ONNX gives a reproducible CPU single-thread engine for bench parity and performance scrutiny | ✓ Shipped |
| Close v1.15 only after EMEL beats ONNX CPU single-thread without weakening gates | A slower native lane would not satisfy the milestone performance contract, and stale or synthetic gates would not survive closeout scrutiny | ✓ Shipped |
| Organize generation and embedding benchmark variants before adding more variants | The existing pluggable surfaces work, but hard-coded lists and code-owned cases make every new variant touch unrelated code and increase determinism risk | ✓ Shipped |
| Promote the deferred `v1.12` generative follow-on into `v1.13` instead of inventing a separate tool family | The shipped embedding compare architecture already proves the lane-isolated pluggable pattern; the next milestone should reuse it for generation with a narrow reproducibility contract | ✓ Shipped |
| Treat generative comparability drift as explicit contract data, not an implicit failure mode | Cross-engine generation can diverge because of formatter, tokenization, or sampling differences, so the workflow must publish why two runs are or are not comparable | ✓ Shipped |
| Keep the new parity/benchmark reference architecture pluggable but lane-isolated | The user wants easy comparison against different inference engines without letting reference runtimes leak into the EMEL lane | ✓ Shipped |
| Treat Python and C++ backends as equal citizens under one canonical comparison contract | The repo already has both styles of reference evidence, so the milestone should unify them instead of favoring one language-specific lane | ✓ Shipped |
| Start decomposition with `generator/prefill` instead of splitting the whole generator at once | Prefill was the clearest request-phase boundary and the largest current duplication source | ✓ Good |
| Collapse the prefill compute-routing matrix before broader extraction | File movement alone would have just relocated the cartesian product instead of reducing it | ✓ Good |
| Keep prefill request-scoped data on typed runtime/internal events | This preserved explicit behavior modeling and avoided context phase flags | ✓ Good |
| Defer decode extraction until the prefill pattern is proven | Decode also owns sampling, rendering, and loop control, so it was the riskier first cut | ✓ Good |
| Defer `attention::any` / `sm_any` extraction until after prefill collapse | Attention mode is only one axis of the duplication and should not hide unresolved top-level routing | ⚠ Revisit |
| Keep v1.9 fixed to one official Liquid Thinking GGUF slice | The repo needs one truthful maintained Liquid anchor before any broader family claims | ✓ Good |
| Use GGUF/config metadata as the maintained truth source for Liquid | Official prose and executable metadata disagree on context length, so docs must follow executable truth | ✓ Good |
| Scope v1.10 to the planner family only | The user asked for planner and planner-submachine compliance, not a broader machine-family rewrite | ✓ Good |
| Pin the first maintained TE slice to `TE-75M-q8_0.gguf` | It keeps the first `omniembed` milestone on the narrowest truthful quant/runtime surface with minimal quality loss | ✓ Locked |
| Treat TE support as explicit `omniembed` model-family work | The Hugging Face GGUF API reports `gguf.architecture=omniembed`; aliasing would overstate support | ✓ Locked |
| Prove TE behavior with upstream golden embeddings and tiny cross-modal smoke checks | The obvious existing generation parity lane does not apply cleanly to `omniembed`, so v1.11 needs its own deterministic proof seam | ✓ Locked |
| Keep generic media decoding, vector search, and public API expansion out of v1.11 | The milestone should stop at one truthful maintained embedding slice, not a broad multimodal platform | ✓ Locked |
| Define the TE proof corpus as deterministic in-memory payload contracts | Phase 47 must anchor one reproducible text/image/audio triplet set before runtime work broadens into encoder execution | ✓ Locked |
| Free `text/encoders` for embedding producers and move tokenizer families under `text/tokenizers` | It keeps tokenizer implementation concerns separate from embedding-capable text producers and preserves room for future hidden-state embedding dispatch | ✓ Locked |
| Keep `embeddings/generator` as the milestone embedding orchestrator | It gives v1.11 an explicit embedding contract without forcing a full generator-domain move in the same milestone | ✓ Locked |
| Validate `omniembed` as a modality-family contract instead of freezing TE-only internals into model acceptance | The user wants general architecture support, and later modality lanes need stable text/image/audio family bindings plus Matryoshka metadata | ✓ Locked |
| Treat `*/forward` as a future reuse seam, not a required modality domain | Every modality has internal forward computation, but a public `forward` domain should only exist when multiple top-level contracts share that hidden-state path | ✓ Locked |
| Vendor the `mdbr-leaf-ir` WordPiece vocab as the maintained text-token truth for TE tests | `TE-75M-q8_0.gguf` omits the tokenizer metadata needed for the text lane, so Phase 49 must pin the upstream vocab asset explicitly instead of inventing one | ✓ Locked |
| Keep Phase 52 on shared proof over the existing embedding actor instead of another orchestration split | The runtime already had one shared publication/truncation surface, so the truthful missing work was contract proof, not more machine churn | ✓ Locked |
| Keep WPM stored-vocab lookup compatible with both raw/`##` and `▁` word-start conventions | The maintained TE vocab and the existing BERT GGUF parity fixture store WordPiece pieces differently, so Phase 53 proof could not regress either truth surface | ✓ Locked |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `$gsd-transition`):
1. Requirements invalidated? -> Move to Out of Scope with reason
2. Requirements validated? -> Move to Validated with phase reference
3. New requirements emerged? -> Add to Active
4. Decisions to log? -> Add to Key Decisions
5. "What This Is" still accurate? -> Update if drifted

**After each milestone** (via `$gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check -> still the right priority?
3. Audit Out of Scope -> reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-04 after archiving v1.23 from GitHub issue #60*
