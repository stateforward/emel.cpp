# Project Milestones: EMEL

## v1.15 ARM Sortformer Diarization GGUF Slice (Shipped: 2026-04-25)

**Phases completed:** 24 phases, 24 plans

**Delivered:** EMEL now ships one maintained ARM Sortformer diarization GGUF slice with
EMEL-owned loading, native runtime execution, deterministic probability/segment output,
PyTorch/NeMo parity proof, ONNX CPU single-thread benchmark reference proof, and source-backed
closeout evidence.

**Key accomplishments:**

- Pinned the maintained `openresearchtools/diar_streaming_sortformer_4spk-v2.1-gguf` fixture,
  model contract, and mono 16 kHz PCM request/output contract.

- Built the maintained EMEL `src/` pipeline through feature preparation, encoder, executor,
  speaker probabilities, and bounded segment publication without external compute fallbacks.

- Added lane-isolated PyTorch/NeMo parity and ONNX CPU single-thread benchmark references that
  exact-match EMEL on `output_dim=17` and checksum `4249677247906920305`.

- Profiled and optimized the maintained runtime until EMEL beat ONNX CPU single-thread in the
  strict Phase 93 generated record: EMEL `1352780166 ns/op` versus ONNX `1920646958 ns/op`.

- Refreshed source-backed milestone audit evidence and final scoped quality gates with timing
  recorded at `246s`.

**Audit:** Passed with `16/16` requirements satisfied, no UAT/verification gaps, and source-backed
traceability from pinned fixture through loader, runtime, parity, benchmark, and docs entrypoints.

**Known deferred items at close:** 5 old non-phase items acknowledged and deferred; see
`.planning/STATE.md` `Deferred Items`.

---

## v1.14 Benchmark Variant Organization (Shipped: 2026-04-21)

**Phases completed:** 4 phases, 4 plans, 0 tasks

**Delivered:** EMEL benchmark variants for generation and embeddings are now registry/data-owned:
adding ordinary maintained variants no longer requires editing unrelated runner, compare, or test
enumeration code.

**Key accomplishments:**

- Added a shared benchmark manifest registry helper with deterministic discovery and duplicate-ID
  validation.

- Cut generation benchmark workload enumeration over to discovered manifests while preserving
  workload ID, case-name, and compare-group filters.

- Cut embedding benchmark case identity over to checked-in variant manifests consumed by the EMEL
  lane and Python-golden reference lane.

- Added operator-facing `--variant-id` selection for embedding compare wrappers, aligned with the
  existing generation `--workload-id` path.

- Documented data-only generation and embedding add paths, including the files ordinary variant
  additions must not touch.

**Audit:** Passed with `12/12` requirements satisfied and no blocking gaps.

---

## v1.13 Pluggable Generative Parity Bench (Shipped: 2026-04-21)

**Phases completed:** 8 phases, 8 plans, 0 tasks

**Delivered:** EMEL now ships one pluggable generative compare workflow that keeps the EMEL lane
isolated while running a maintained `llama_cpp_generation` reference lane through the shared
`generation_compare/v1` contract, with truthful comparable and non-comparable publication.

**Key accomplishments:**

- Defined one canonical `generation_compare/v1` JSONL contract for prompts, generated outputs,
  verdict metadata, and timing across EMEL and reference lanes.

- Added checked-in generation workload and prompt manifests that pin fixture identity, formatter
  mode, seed, sampling, stop conditions, and comparability intent.

- Integrated a maintained local `llama_cpp_generation` reference backend behind manifest-selected
  wrapper tooling without leaking reference objects into the EMEL runtime lane.

- Published `scripts/bench_generation_compare.sh` and `generation_compare_summary/v1` verdicts for
  exact match, bounded drift, non-comparable, missing, and error outcomes.

- Repaired JSONL lane isolation so EMEL-only and reference-only runs no longer prepare the other
  lane's fixture state.

- Added a maintained single-lane LFM2 workload that flows through the operator wrapper and
  truthfully publishes `non_comparable/single_lane_emel_workload` with an empty reference record
  file.

- Backfilled requirement traceability and Nyquist validation evidence for Phases 69 through 76,
  producing a `tech_debt` audit with no blocking gaps.

**Known deferred items at close:** 5 open non-phase items acknowledged and deferred; see
`.planning/STATE.md` `Deferred Items`.

**Technical debt:** Metadata mismatch tests sample representative fields directly, and the audit
turn reused the post-review Phase 75 full quality-gate pass instead of rerunning the full gate.

---

## v1.12 Pluggable Reference Parity Bench Architecture (Shipped: 2026-04-18, Closeout Repaired: 2026-04-20)

**Delivered:** EMEL now ships one pluggable embedding compare architecture that keeps the EMEL
lane isolated while running Python and C++ reference engines through one canonical comparison
contract, with repaired multi-record C++ publication and refreshed closeout evidence.

**Phases completed:** 7 phases, 7 plans, 0 tasks

**Key accomplishments:**

- Defined one canonical `embedding_compare/v1` contract shared by EMEL and reference lanes without
  changing the EMEL-owned runtime lane.

- Added maintained Python reference backends for deterministic stored-golden TE parity and
  explicit live-environment failure reporting.

- Moved the maintained Liquid C++ reference workflow behind the same manifest-driven backend
  contract used by Python backends.

- Published one operator-facing compare workflow that emits machine-readable JSONL lane records,
  dumped vectors, and `compare_summary.json`.

- Repaired the lossy C++ summary path so both maintained baseline records survive shared-group
  publication.

- Backfilled the missing requirement-traceability and Nyquist evidence for the shipped closeout.
- Repaired the archived Phase `67` proof-path drift so the rerun milestone audit is self-consistent
  after archival.

**What's next:** Define the next milestone and create a fresh requirements set. The immediate
follow-on work could broaden reference backends, extend the compare architecture beyond
embeddings, or formalize plugin/remote-backend scope if that is still desired.

---

## v1.11 TE-75M GGUF Trimodal Embedding Runtime (Shipped: 2026-04-15)

**Phases completed:** 3 phases, 3 plans, 0 tasks

**Key accomplishments:**

- (none recorded)

---

## v1.9 Liquid LFM2.5-1.2B Thinking ARM Slice (Shipped: 2026-04-02)

**Phases completed:** 8 phases, 9 plans, 0 tasks

**Key accomplishments:**

- Documented one official maintained `LFM2.5-1.2B-Thinking-Q4_K_M.gguf` fixture with executable
  metadata truth anchored on `lfm2` and `128000` context.

- Added explicit `lfm2` model and runtime support for one maintained Liquid ARM slice through the
  shipped generator path.

- Proved maintained Liquid parity against `llama.cpp` while preserving additive maintained Qwen
  coverage.

- Published one maintained Liquid benchmark/docs path aligned with the parity-backed fixture and
  contract.

- Reconstructed missing closeout artifacts and validation coverage so the milestone could pass
  audit and archive cleanly.

**What's next:** Start the next milestone. The likely follow-on work is broader Liquid coverage,
richer Liquid request support, or Liquid performance work on top of the shipped maintained slice.

---

## v1.8 Truthful Qwen3 E2E Embedded Size (Shipped: 2026-04-02)

**Delivered:** EMEL now publishes one truthful maintained Qwen3-0.6B end-to-end executable-size
comparison between an EMEL-owned runner and a matched `llama.cpp` reference executable, backed by
runtime smoke proof and generated README evidence.

**Phases completed:** 6 phases, 8 plans, 0 tasks

**Key accomplishments:**

- Locked one exact maintained workload boundary on `tests/models/Qwen3-0.6B-Q8_0.gguf`,
  structured `hello`, and first-token generation.

- Corrected the EMEL probe into a truthful final executable measurement and removed redundant
  fallback-vocab bloat that had inflated the binary to 56 MB.

- Kept the published comparator set narrow to EMEL and one matched `llama.cpp` reference
  executable with shared smoke proof.

- Refreshed the stored snapshot and generated README to publish the corrected executable-size
  values: `4,073,016` raw bytes for EMEL versus `3,334,264` for the reference row.

- Backfilled the missing v1.8 proof chain and closed the milestone with a passing `8/8`
  requirement audit.

---

## v1.7 Generator Prefill Submachine Decomposition (Shipped: 2026-03-30)

**Delivered:** The generator now ships an explicit generator-owned `prefill` child machine with
request-scoped prefill contracts, a materially smaller top-level generator surface, and maintained
Llama/Qwen proof preserved across the refactor.

**Phases completed:** 3 phases, 6 plans, 0 tasks

**Key accomplishments:**

- Collapsed the repeated top-level prefill routing matrix into explicit request-scoped prefill
  compute contracts.

- Extracted `src/emel/generator/prefill` as a generator-domain child machine with its own typed
  `run` event and explicit contract/result states.

- Kept prefill orchestration data on typed runtime/internal events instead of generator context
  phase flags.

- Reduced the parent generator surface materially and published the split `generator_prefill`
  architecture docs.

- Re-ran maintained generator, paritychecker, and compare proof on the extracted prefill boundary,
  with unrelated broad benchmark drift explicitly waived for milestone closeout.

**What's next:** Decide whether to continue Issue `#41` with decode extraction or revisit deeper
generator-family decomposition after the prefill pattern.

---

## v1.6 Qwen3-0.6B Parity And Benchmark (Shipped: 2026-03-30)

**Delivered:** EMEL now ships one truthful canonical `Qwen3-0.6B-Q8_0.gguf` maintained slice with
an explicit GGUF-derived formatter contract, native `src/emel` `q8_0` runtime support, stored
generation parity, and benchmark publication aligned to the same operator-facing Qwen workflow.

**Phases completed:** 5 phases, 12 plans, 6 tasks

**Key accomplishments:**

- Locked one official canonical Qwen3 fixture and primary-template-only formatter contract on the
  maintained paritychecker and benchmark surfaces

- Added native `q8_0` runtime support for the canonical Qwen blocker tensors in `src/emel`
- Brought the shipped generator path up on the canonical `qwen3` slice without broad family claims
- Proved maintained stored-baseline parity across `1/10/100/1000` while keeping the prior Llama
  anchor protected

- Published one truthful canonical Qwen benchmark compare, snapshot, and docs path with explicit
  formatter/runtime evidence

**What's next:** Define the next milestone before broadening Qwen scope, widening request surfaces,
or hardening benchmark-gate policy.

---

## v1.5 Full ARM Quantized Path (Shipped: 2026-03-27)

**Delivered:** The canonical CPU-hosted Llama-68M ARM slice now ships an explicit maintained
quantized-path contract, zero supported disallowed fallback, and restored checked-in flash
attribution/publication across paritychecker, compare snapshots, and generated benchmark docs.

**Phases completed:** 5 phases, 10 plans, 0 tasks

**Key accomplishments:**

- The canonical ARM slice now has a shared stage-by-stage quantized-path audit
- Unsupported quantized branches now publish explicit no-claim behavior
- The shipped generator runtime, paritychecker, and regression surfaces now prove the canonical
  `8/4/0/0` runtime contract with zero supported disallowed fallback

- Benchmark compare output, stored snapshots, and generated docs now publish the same runtime
  contract without overstating approved dense-f32-by-contract seams

- Canonical flash-attention dispatch and checked-in benchmark publication were restored together so
  maintained live proof and stored evidence match again

**What's next:** Define the next milestone before widening scope beyond the canonical CPU-hosted
Llama-68M ARM slice or changing benchmark-gate policy.

---

## v1.4 Full Vectorized Quantized Kernels (Shipped: 2026-03-25)

**Delivered:** The canonical CPU-hosted Llama-68M ARM slice now ships EMEL-owned vectorized
`q2_K/q3_K/q6_K x q8_K` kernels, maintained runtime attribution, full `1/10/100/1000` parity
proof, and refreshed benchmark publication against the preserved v1.3 scalar baseline.

**Phases completed:** 5 phases, 11 plans, 0 tasks

**Key accomplishments:**

- Replaced the maintained scalar `q2_K`, `q3_K`, and `q6_K` row helpers with EMEL-owned
  vectorized AArch64 kernels on the canonical operand path.

- Closed the maintained quantized hot-path contract with alloc-free q2/q3/q6 dispatch and no
  dequantize-to-f32 fallback.

- Exposed shipped q2/q3/q6 optimized-versus-shared runtime attribution without widening the
  actor or API surface.

- Restored maintained parity across `1`, `10`, `100`, and `1000` tokens on the canonical ARM
  workload.

- Refreshed maintained benchmark compare output and docs with quantized attribution and preserved
  v1.3 baseline context.

**What's next:** Define the next milestone before broadening beyond the canonical CPU-hosted
Llama-68M ARM slice or widening benchmark-gate policy.

---

## v1.3 ARM Flash Optimizations (Shipped: 2026-03-22)

**Delivered:** Optimized AArch64 flash execution now ships on the canonical Llama-68M ARM slice,
with parity and benchmark surfaces publishing optimized-vs-shared attribution and maintained docs
preserving the prior ARM baseline while showing a measured short-case improvement.

**Phases completed:** 3 phases, 7 plans, 0 tasks

**Key accomplishments:**

- The maintained AArch64 flash request now has a native backend execution path.
- Phase 14 closed kernel-local proof for correctness, scratch reuse, and zero shared fallback.
- The shipped generator seam now reports optimized-vs-shared flash path selection.
- Paritychecker and benchmark compare output now publish optimized-vs-shared ARM flash
  attribution on the maintained workload.

- Maintained benchmark publication now preserves the prior ARM baseline and documents a `1.140x`
  short-case speedup.

**What's next:** Define the next milestone before broadening scope beyond the canonical
CPU-hosted Llama-68M ARM slice.

---

## v1.2 Flash Attention (Shipped: 2026-03-22)

**Delivered:** The canonical Llama-68M generation slice now runs through an EMEL-owned
flash-attention path, paritychecker proves it on the normal surface, the shipped runtime is
hard-cut over to `emel::tensor::sm`, and benchmark docs publish maintained flash evidence with a
preserved pre-flash baseline.

**Phases completed:** 5 phases, 13 plans, 13 tasks

**Key accomplishments:**

- Added a real EMEL-owned flash-attention kernel path with backend-owned workspace reuse.
- Adopted flash attention in the shipped generator runtime with deterministic unsupported-request
  failure proof.

- Made paritychecker fetch and publish upstream reference identity while proving flash execution on
  both maintained workloads.

- Hard-cut graph and generator tensor lifecycle orchestration over to `emel::tensor::sm` with an
  alloc-free dispatch proof.

- Published canonical benchmark proof comments, a preserved pre-flash artifact, and generated docs
  showing a `9.126x` short-case improvement over the prior EMEL baseline.

**What's next:** Define the next milestone before widening scope beyond the canonical
CPU-hosted Llama-68M slice.

---

## v1.1 EMEL Llama-68M Generation Benchmark (Shipped: 2026-03-11)

**Delivered:** One truthful canonical Llama-68M generation benchmark in `tools/bench`, using a
shared native EMEL decode backend and the existing compare, snapshot, and docsgen workflow.

**Phases completed:** 4 phases, 10 plans

**Key accomplishments:**

- Added the canonical generation benchmark case for the shipped Llama-68M slice.
- Replaced the circular reference-backed decode seam with a shared native EMEL backend.
- Published stable EMEL-vs-`llama.cpp` compare output through the maintained compare surface.
- Integrated benchmark snapshots and generated benchmark docs into the existing operator workflow.

**What's next:** Add EMEL-owned flash attention to the same canonical generation slice.

---

## v1.0 EMEL Llama-68M Generation Slice (Shipped: 2026-03-08)

**Delivered:** The first parity-checked canonical Llama-68M generation slice through the existing
EMEL runtime and `tools/paritychecker`.

**Phases completed:** 7 phases, 15 plans

**Key accomplishments:**

- Implemented the real GGUF/model-loading path for the shipped canonical fixture.
- Wired bounded generation end to end through the existing EMEL runtime.
- Added parity-oriented subprocess success and failure coverage for the maintained workload.
- Established the narrow paritychecker-first acceptance boundary used by later milestones.

**What's next:** Build a truthful benchmark surface on top of the proven generation slice.

---
