# Roadmap

## Archived Milestones

- [x] [v1.0: EMEL Llama-68M Generation Slice](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.0-ROADMAP.md) - shipped 2026-03-08 with 7 phases and 15 plans; proved one canonical Llama-68M generation parity slice in `tools/paritychecker/`.
- [x] [v1.1: EMEL Llama-68M Generation Benchmark](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.1-ROADMAP.md) - shipped 2026-03-11 with 4 phases and 10 plans; added one truthful canonical Llama-68M generation benchmark in `tools/bench`, native EMEL decode benchmarking, compare output, and snapshot/docs integration.
- [x] [v1.2: Flash Attention](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.2-ROADMAP.md) - shipped 2026-03-22 with 5 phases and 13 plans; added an EMEL-owned flash-attention path to the canonical Llama-68M slice, hard-cut runtime tensor lifecycle through `emel::tensor::sm`, and published maintained benchmark evidence over a preserved pre-flash baseline.
- [x] [v1.3: ARM Flash Optimizations](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.3-ROADMAP.md) - shipped 2026-03-22 with 3 phases and 7 plans; delivered optimized AArch64 flash execution, maintained runtime/parity attribution, and preserved-baseline benchmark publication for the canonical ARM Llama-68M slice.
- [x] [v1.4: Full Vectorized Quantized Kernels](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.4-ROADMAP.md) - shipped 2026-03-25 with 5 phases and 11 plans; delivered EMEL-owned vectorized q2/q3/q6 kernels, full maintained `1/10/100/1000` parity proof, and quantized benchmark attribution on the canonical ARM slice.
- [x] [v1.5: Full ARM Quantized Path](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.5-ROADMAP.md) - shipped 2026-03-27 with 5 phases and 10 plans; closed the maintained ARM quantized-path contract and restored canonical flash publication.
- [x] [v1.6: Qwen3-0.6B Parity And Benchmark](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.6-ROADMAP.md) - shipped 2026-03-30 with 5 phases and 12 plans; brought one canonical Qwen3 slice up through the maintained generator, parity, and benchmark surfaces.
- [x] [v1.7: Generator Prefill Submachine Decomposition](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.7-ROADMAP.md) - shipped 2026-03-30 with 3 phases and 6 plans; extracted generator-owned prefill orchestration while preserving maintained proof.
- [x] [v1.8: Truthful Qwen3 E2E Embedded Size](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.8-ROADMAP.md) - shipped 2026-04-02 with 6 phases and 8 plans; published one truthful maintained Qwen3 executable-size comparison on the canonical first-token workload.
- [x] [v1.9: Liquid LFM2.5-1.2B Thinking ARM Slice](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.9-ROADMAP.md) - shipped 2026-04-02 with 8 phases and 9 plans; brought one maintained Liquid slice up through the generator, paritychecker, and benchmark workflow.

## Current Milestone

### v1.11 TE-75M GGUF Trimodal Embedding Runtime (Reopened For Audit Gap Closure)

This milestone was reopened on 2026-04-14 after the milestone audit found one partial runtime seam
and missing structured closeout evidence for requirements previously marked complete. The closure
work stayed inside the shipped TE-75M `omniembed` slice: no new modality scope, no public API
expansion, and no broader model-family claims.
The reopened audit-gap chain, the inserted ARM-throughput follow-ons, the same-host external ARM
baseline comparison, the focused image-tower lowering and kernelization passes, the generator
rule-clean refactor, the maintained q5 truth reconciliation, and the refreshed validation /
bookkeeping sweep are now complete. The live ledger, maintained q8/q5 TE scope, validation
artifacts, and refreshed milestone audit now agree. `v1.11` is ready for lifecycle completion;
the only remaining caution is the existing warning-only benchmark snapshot review policy under
`scripts/quality_gates.sh`.

## Phases

**Phase Numbering:**
- `v1.11` originally ran from Phase `47` through Phase `53`.
- Gap-closure work continues from the next free phase number, so the reopen starts at Phase `54`.
- Remaining audit-gap work continues from the next free phase number, so the follow-on closure
  starts at Phase `57`.

- [x] **Phase 47: TE Truth Anchor** - Pin the maintained TE fixture, provenance, and narrow proof inputs.
- [x] **Phase 48: Omniembed Model Contract** - Add truthful `omniembed` GGUF/model acceptance and execution bindings.
- [x] **Phase 49: Text Embedding Lane** - Bring up the maintained TE text path with normalized and truncatable embeddings.
- [x] **Phase 50: Vision Embedding Lane** - Add the maintained in-memory image path and TE vision encoder runtime.
- [x] **Phase 51: Audio Embedding Lane** - Add the maintained in-memory audio path and TE audio encoder runtime.
- [x] **Phase 52: Shared Embedding Session** - Unify modality output, truncation, and deterministic request/error contracts.
- [x] **Phase 53: TE Proof And Regression** - Add golden-baseline verification and tiny cross-modal smoke protection.
- [x] **Phase 54: Omniembed Execution Contract Runtime Cutover** - Cut the live embedding runtime over to the explicit Phase 48 execution-contract seam. (completed 2026-04-15)
- [x] **Phase 55: Embedding Lane Traceability Backfill** - Restore structured requirement traceability for the text, vision, audio, and shared-session closure phases. (completed 2026-04-15)
- [x] **Phase 56: Proof And Nyquist Closeout** - Backfill proof traceability and Nyquist validation so `v1.11` can re-audit cleanly. (completed 2026-04-15)
- [x] **Phase 57: Embedding Generator Rule Compliance And Error Proof** - Remove runtime branching from the maintained embedding actor actions and lock exact contract-drift error classification. (completed 2026-04-15)
- [x] **Phase 58: Embedding Generator Benchmark Publication** - Add one maintained benchmark path that exercises the full TE embedding request flow through `src/emel/embeddings/generator`. (completed 2026-04-15)
- [x] **Phase 58.1: ARM Embedding Generator Throughput Optimization** - Convert the maintained embedding benchmark into steady-state ARM throughput evidence with explicit `prepare` / `encode` / `publish` timing. (completed 2026-04-15)
- [x] **Phase 58.1.1: Liquid AI Multimodal Reference Throughput And Parity** - Add the maintained Liquid multimodal reference runner and pinned build lane for external image/audio comparison. (completed 2026-04-15)
- [x] **Phase 59: Validate v1.11 And Repair Closeout Bookkeeping** - Update milestone state/roadmap bookkeeping and rerun the audit to clean closeout. (completed 2026-04-15)
- [x] **Phase 59.4: Kernelize image throughput hot path into AArch64 architecture kernels (INSERTED)** - Move the remaining image hot path from `src/emel/embeddings/generator/detail.hpp` into prepacked architecture-shaped kernels under `src/emel/kernel/aarch64`. (completed 2026-04-15)
- [x] **Phase 59.3: Continue profiling and optimizing image performance (INSERTED)** - Continue ROI-ranked profiling and optimization of maintained image performance before refreshed audit and closeout handling resume. (completed 2026-04-16)
- [x] **Phase 59.2: Refactor Embedding Generator Hidden Control Flow Into Explicit Guards And Transitions (INSERTED)** - Remove remaining hidden runtime behavior and outcome selection from the maintained embedding generator before more optimization and closeout work. (completed 2026-04-15)
- [x] **Phase 59.1: Optimize For High Throughput On ARM (INSERTED)** - Raise steady-state ARM embeddings throughput on the maintained generator path using ROI-ranked profiling and the agreed external baseline matrix.
- [x] **Phase 59.1.1: Run llama.cpp Supported ARM Throughput Baseline Comparison (INSERTED)** - Run the agreed external baseline set on the same ARM host and publish the gap to EMEL by modality and stage. (completed 2026-04-15)
- [x] **Phase 59.1.1.1: Optimize Image Tower ARM Kernel Lowering (INSERTED)** - Attack the profiled single-thread image-tower hotspots in ROI order on the maintained generator path before milestone closeout resumes. (completed 2026-04-15)
- [x] **Phase 60: Reconcile Maintained TE Quant Scope And Proof Truth** - Make `FIX-02` truthful again by aligning maintained TE quant claims, runtime acceptance, and proof coverage after the q5 follow-on work. (completed 2026-04-16)
- [x] **Phase 61: Refresh Validation And Closeout Audit** - Finish the remaining phase-artifact and validation sweep after `59.3` and `60`, then rerun bookkeeping and the milestone audit. (completed 2026-04-16)

## Phase Details

### Phase 54: Omniembed Execution Contract Runtime Cutover
**Goal**: Make the live TE embedding runtime consume the explicit `omniembed` execution contract
instead of rebinding directly from raw model metadata and tensor names.
**Depends on**: Phase 53
**Requirements**: MOD-02, EMB-01
**Gap Closure**: Closes the `MOD-02` audit gap and the `48 -> 49/50/51` integration seam flagged
by the milestone audit.
**Success Criteria** (what must be TRUE):
  1. The live embedding runtime consumes the Phase 48 `execution_contract` as its enforcement seam
     for the maintained text, image, and audio bindings.
  2. Loader/runtime validation rejects drift between the `omniembed` contract and the embedding
     runtime before requests reach modality execution.
  3. Shared-session behavior still returns one consistent normalized embedding contract across
     modalities after the cutover.
**Plans**: 54-01

### Phase 55: Embedding Lane Traceability Backfill
**Goal**: Rebuild the structured closeout evidence for the text, image, audio, and shared-session
requirements that the milestone audit marked orphaned.
**Depends on**: Phase 54
**Requirements**: TXT-01, TXT-02, VIS-01, VIS-02, AUD-01, AUD-02, EMB-02
**Gap Closure**: Closes the orphaned requirement gaps from phases `49` through `52`.
**Success Criteria** (what must be TRUE):
  1. Phases `49`, `50`, `51`, and `52` expose `requirements-completed` frontmatter that the audit
     workflow can extract directly.
  2. The corresponding `VERIFICATION.md` artifacts contain explicit requirement coverage sections
     for each mapped REQ-ID.
  3. A re-audit can trace each text, vision, audio, and shared-session requirement across
     `REQUIREMENTS.md`, `SUMMARY.md`, and `VERIFICATION.md` without orphan detection.
**Plans**: 55-01

### Phase 56: Proof And Nyquist Closeout
**Goal**: Complete the final proof traceability and validation artifacts required for a clean
`v1.11` re-audit and milestone closeout.
**Depends on**: Phase 55
**Requirements**: PRF-01, PRF-02
**Gap Closure**: Closes the orphaned proof requirements from Phase `53` and the missing Nyquist
validation coverage across phases `47` through `53`.
**Success Criteria** (what must be TRUE):
  1. Phase `53` exposes structured proof requirement coverage and `requirements-completed`
     frontmatter for `PRF-01` and `PRF-02`.
  2. Phases `47` through `53` all have audit-visible Nyquist validation artifacts.
  3. `v1.11` re-audits without orphaned requirements or missing-validation findings.
**Plans**: 56-01

### Phase 57: Embedding Generator Rule Compliance And Error Proof
**Goal**: Make the maintained TE embedding actor rule-clean against the repo's SML contract and
close the remaining partial proof on contract-drift error classification.
**Depends on**: Phase 56
**Requirements**: None (gap-closure phase)
**Gap Closure**: Closes the runtime rule-cleanliness blocker and the partial
`MOD-02` / `EMB-01` initialization-error proof gap from the milestone audit.
**Success Criteria** (what must be TRUE):
  1. `src/emel/embeddings/generator/actions.hpp` no longer contains runtime `if` branching for
     image/audio preparation, text/image/audio execution, or embedding publication.
  2. The missing-family initialization regression locks the exact maintained error classification
     for contract drift instead of only asserting a non-`none` failure.
  3. The maintained embedding actor remains RTC-compliant and passes focused verification after the
     rule cleanup.
**Plans**: 57-01

### Phase 58: Embedding Generator Benchmark Publication
**Goal**: Publish one maintained benchmark surface for the TE slice that measures the real
embedding-generator request flow instead of ad hoc helper-local execution.
**Depends on**: Phase 57
**Requirements**: None (gap-closure phase)
**Gap Closure**: Closes the audit finding that `v1.11` lacks a maintained benchmark exercising the
full `src/emel/embeddings/generator` path.
**Success Criteria** (what must be TRUE):
  1. `tools/bench` contains at least one maintained TE benchmark case that drives initialize plus a
     real text, image, or audio embedding request through `src/emel/embeddings/generator`.
  2. The benchmark surface identifies the exact maintained TE fixture and request contract used for
     the published measurement.
  3. The benchmark path is auditable as embedding-generator evidence rather than a helper-only or
     subsystem-local timing scaffold.
**Plans**: 58-01

### Phase 58.1: ARM Embedding Generator Throughput Optimization (INSERTED)

**Goal**: Convert the maintained embedding benchmark from a cold-path proof into steady-state ARM
throughput work by landing the requested kernel-path cleanup and phase timing evidence through the
real embedding generator state machine.
**Requirements**: None (gap-closure phase)
**Depends on:** Phase 58
**Gap Closure**: Closes the post-benchmark ARM performance gap by routing hot compute through the
maintained ARM kernels, removing repeated fp16 decode and feature-buffer ping-pong, and exposing
auditable `prepare` / `encode` / `publish` timing for the maintained embedding-generator bench.
**Success Criteria** (what must be TRUE):
  1. The maintained embedding generator reuses compute-friendly bound weights instead of decoding
     fp16 inside inner loops, and steady-state ARM requests use the maintained ARM matmul kernels
     on the hot path where supported.
  2. The image and audio towers no longer copy full feature buffers back and forth between stages
     when a buffer swap is sufficient.
  3. The maintained embedding benchmark can report steady-state `prepare`, `encode`, and `publish`
     timing for text, image, and audio requests through `src/emel/embeddings/generator` without
     helper-local ad hoc timing hooks or hidden control flow.
**Plans**: 58.1-01

### Phase 58.1.1: Liquid AI Multimodal Reference Throughput And Parity (INSERTED)

**Goal**: Add a maintained reference lane against Liquid AI's `benchmarks-llama.cpp` multimodal
stack so ARM image/audio throughput can be compared against a real C/C++ vision/audio tower
implementation instead of Python or text-only references.
**Requirements**: None (gap-closure phase)
**Depends on:** Phase 58.1
**Gap Closure**: Closes the missing multimodal reference-lane gap by adding a maintained Liquid
fork benchmark/parity surface for image/audio tower throughput and deterministic output anchors.
**Success Criteria** (what must be TRUE):
  1. `tools/bench` can build a dedicated reference runner against `Liquid4All/benchmarks-llama.cpp`
     and its `libmtmd` image/audio path without touching the EMEL lane.
  2. The reference runner reports separate `prepare`, `encode`, and `publish` timing for image and
     audio tower inputs, plus deterministic output metadata useful for parity anchoring.
  3. The maintained workflow to configure/build the Liquid reference lane is captured in-repo so
     the comparison can be rerun without ad hoc local setup.
**Plans**: 58.1.1-01

### Phase 59: Validate v1.11 And Repair Closeout Bookkeeping
**Goal**: Bring milestone bookkeeping back in sync with the completed work and rerun the milestone
audit to clean closeout.
**Depends on**: Phase 58.1.1
**Requirements**: None (gap-closure phase)
**Gap Closure**: Closes the stale `.planning/STATE.md` contradictions and the blocked milestone
closeout flow from the rerun audit.
**Success Criteria** (what must be TRUE):
  1. `.planning/STATE.md`, `.planning/ROADMAP.md`, and related milestone bookkeeping agree on the
     real `v1.11` status instead of simultaneously reporting reopened, pending, and archived
     states.
  2. The new rule-clean embedding proof and maintained benchmark evidence are visible to the
     milestone audit.
  3. `$gsd-audit-milestone` passes or narrows to non-blocking tech debt, enabling clean milestone
     completion.
**Plans**: 59-01

### Phase 59.4: Kernelize image throughput hot path into AArch64 architecture kernels (INSERTED)

**Goal:** Move the maintained ARM image hot path out of
`src/emel/embeddings/generator/detail.hpp` and into `src/emel/kernel/aarch64` while keeping the
public `embeddings/generator` request path and explicit model behavior unchanged.
**Requirements**:
1. The maintained image lane must stay on the public `embeddings/generator` event path while the
   ARM hot numeric loops move into `src/emel/kernel/aarch64`.
2. Prepack remains one-time only; no dispatch-time allocation or hidden control flow may be
   introduced in `detail.hpp`.
3. The phase must republish maintained image throughput evidence and hotspot ownership after the
   kernel move.
**Depends on:** Phase 59
**Plans:** 59.4-01

Plans:
- [x] **59.4-01** - Move the maintained ARM image pointwise and depthwise hot loops into
  `src/emel/kernel/aarch64`, rerun maintained image evidence, and confirm the hotspot is now
  kernel-owned.

### Phase 59.3: Continue profiling and optimizing image performance (INSERTED)

**Goal:** Continue ROI-ranked profiling and optimization of maintained single-thread image
performance on the public `embeddings/generator` path and carry the gains onto the approved q5
edge slice.
**Requirements**:
1. Keep all throughput claims on the maintained image request path through
   `src/emel/embeddings/generator`, not on helper-local or tool-only compute.
2. Reprofile the image lane after each landed optimization and keep the next hotspot explicit
   instead of folding runtime behavior into `detail.hpp`.
3. Republish both q8 and q5 maintained image evidence so the edge slice inherits the image
   optimization wins truthfully.
**Depends on:** Phase 59
**Plans:** 59.3-01

Plans:
- [x] **59.3-01** - Reprofile the maintained image lane, keep the fused/direct image path on the
  real benchmark surface, and republish q8/q5 image evidence plus the next ROI hotspot.

### Phase 60: Reconcile Maintained TE Quant Scope And Proof Truth

**Goal:** Make the maintained TE quant-scope contract truthful again after the live repo started
accepting `TE-75M-q5_0.gguf` while `v1.11` still claims a q8-only maintained slice.
**Depends on:** Phase 59.3
**Requirements**: FIX-02
**Gap Closure**: Closes the partial `FIX-02` requirement, the maintained quant-gating flow break,
and the Phase `47` policy -> live runtime integration contradiction from the rerun audit.
**Success Criteria** (what must be TRUE):
  1. `REQUIREMENTS.md`, `tests/models/README.md`, maintained tests, and the live TE runtime all
     agree on the exact maintained TE quant scope instead of mixing q8-only claims with q5
     acceptance.
  2. The maintained TE workflow either rejects unapproved sibling quant artifacts again or
     explicitly promotes the supported sibling slice with truthful proof/benchmark coverage; no
     implied broad TE support remains.
  3. `FIX-02` can be re-audited as satisfied from requirements, summary, and verification evidence
     without contradicting the live runtime behavior.
**Plans**: 60-01

Plans:
- [x] **60-01** - Reconcile the maintained TE quant contract around the approved q8/q5 fixtures,
  add q5 proof/manifest coverage, and republish the maintained benchmark lane truthfully.

### Phase 61: Refresh Validation And Closeout Audit

**Goal:** Finish the remaining inserted-phase validation and bookkeeping sweep after the pending
image-performance and quant-scope closure work, then rerun the root milestone audit on a truthful
ledger.
**Depends on:** Phase 60
**Requirements**: None (gap-closure phase)
**Gap Closure**: Closes the milestone closeout flow break, the stale Phase `59` bookkeeping claim,
and the remaining missing-validation debt called out by the rerun audit.
**Success Criteria** (what must be TRUE):
  1. Phase `59.3` and the new closure work have the summary/verification artifacts required by the
     milestone audit.
  2. The inserted follow-on phases that remain in milestone scope have audit-visible validation or
     explicit resolved rationale where the workflow expects it.
  3. `ROADMAP.md`, `STATE.md`, `REQUIREMENTS.md`, and `.planning/v1.11-MILESTONE-AUDIT.md` agree
     on the live milestone status, and a rerun audit passes or narrows to non-blocking tech debt.
**Plans**: 61-01

Plans:
- [x] **61-01** - Backfill the remaining validation artifacts, rerun the full maintained quality
  gate surface, reconcile the planning ledger, and republish the root milestone audit.

### Phase 59.2: Refactor Embedding Generator Hidden Control Flow Into Explicit Guards And Transitions (INSERTED)

**Goal:** Remove the remaining hidden runtime behavior and outcome selection from the maintained
`embeddings/generator` actor so the model is explicit through guards and transitions instead of
bool-returning helpers or generic helper routing.
**Requirements**:
1. Runtime success/error choice for prepare and encode phases must be visible in explicit guards
   and transitions, not hidden behind bool-returning helpers called from actions.
2. Runtime behavior choice for the maintained embedding-generator lane must not be hidden in
   generic `detail.hpp` helper routing or ad hoc fallback logic.
3. The maintained TE embedding actor must remain on the public `process_event(...)` path and stay
   verification-clean after the refactor.
**Depends on:** Phase 59
**Plans:** 59.2-01

Plans:
- [x] **59.2-01** - Add a failing generator-rule regression, refactor the maintained
  embedding-generator model to make the scoped runtime choices explicit in guards/transitions, and
  rerun maintained verification.

### Phase 59.1: Optimize For High Throughput On ARM (INSERTED)

**Goal:** Raise steady-state ARM embeddings throughput on the maintained `embeddings/generator`
path using ROI-ranked profiling and the agreed `llama.cpp`-supported comparison matrix.
**Requirements**:
1. Keep all throughput claims on the real steady-state `embeddings/generator` path and its
   maintained `prepare` / `encode` / `publish` timing split.
2. Attack the highest-ROI stages first: image `encode`, then audio `prepare`, then text only if
   the data justifies it.
3. Anchor external comparison methodology to `Arctic S`, `EmbeddingGemma 300M`, `LFM2-VL-450M`,
   and `Ultravox 1B` without widening the current EMEL runtime scope.
**Depends on:** Phase 59
**Plans:** 59.1-01

Plans:
- [x] **59.1-01** - Optimize image and audio first on the maintained ARM benchmark, then publish
  final evidence against the agreed external baseline matrix.

### Phase 59.1.1: Run llama.cpp supported ARM throughput baseline comparison (INSERTED)

**Goal:** Run the agreed `llama.cpp`-supported baseline matrix on the same ARM hardware as the
maintained TE embedding benchmark so EMEL throughput can be compared against a real external
reference per modality and stage.
**Requirements**:
1. Keep the EMEL lane and reference lane split cleanly: no shared runtime state, no shared model
   objects, and no helper-local shortcuts.
2. Use the locked external baseline set only: `Arctic S`, `EmbeddingGemma 300M`, `LFM2-VL-450M`,
   and `Ultravox 1B`.
3. Publish steady-state ARM evidence with the same `prepare` / `encode` / `publish` framing used
   by the maintained EMEL benchmark, plus an explicit note where the reference lane cannot expose
   an exact stage match.
**Depends on:** Phase 59.1
**Gap Closure**: Closes the remaining external-baseline execution gap so the ARM throughput work
can be judged against real `llama.cpp`-supported embedding and multimodal references instead of
internal-only evidence.
**Plans:** 59.1.1-01

Plans:
- [x] **59.1.1-01** - Extend the maintained reference lane, run the approved same-host baseline
  matrix, and publish the EMEL gap with explicit output-contract caveats.

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 47. TE Truth Anchor | 1/1 | Complete | 2026-04-13 |
| 48. Omniembed Model Contract | 1/1 | Complete | 2026-04-14 |
| 49. Text Embedding Lane | 1/1 | Complete | 2026-04-14 |
| 50. Vision Embedding Lane | 1/1 | Complete | 2026-04-14 |
| 51. Audio Embedding Lane | 1/1 | Complete | 2026-04-14 |
| 52. Shared Embedding Session | 1/1 | Complete | 2026-04-14 |
| 53. TE Proof And Regression | 1/1 | Complete | 2026-04-14 |
| 54. Omniembed Execution Contract Runtime Cutover | 1/1 | Complete | 2026-04-15 |
| 55. Embedding Lane Traceability Backfill | 1/1 | Complete | 2026-04-15 |
| 56. Proof And Nyquist Closeout | 1/1 | Complete | 2026-04-15 |
| 57. Embedding Generator Rule Compliance And Error Proof | 1/1 | Complete | 2026-04-15 |
| 58. Embedding Generator Benchmark Publication | 1/1 | Complete | 2026-04-15 |
| 58.1. ARM Embedding Generator Throughput Optimization | 1/1 | Complete | 2026-04-15 |
| 58.1.1. Liquid AI Multimodal Reference Throughput And Parity | 1/1 | Complete | 2026-04-15 |
| 59. Validate v1.11 And Repair Closeout Bookkeeping | 1/1 | Complete | 2026-04-15 |
| 59.4. Kernelize image throughput hot path into AArch64 architecture kernels | 1/1 | Complete | 2026-04-15 |
| 59.3. Continue profiling and optimizing image performance | 1/1 | Complete | 2026-04-16 |
| 59.2. Refactor Embedding Generator Hidden Control Flow Into Explicit Guards And Transitions | 1/1 | Complete | 2026-04-15 |
| 59.1. Optimize For High Throughput On ARM | 1/1 | Complete | 2026-04-15 |
| 59.1.1. Run llama.cpp Supported ARM Throughput Baseline Comparison | 1/1 | Complete | 2026-04-15 |
| 59.1.1.1. Optimize Image Tower ARM Kernel Lowering | 1/1 | Complete | 2026-04-15 |
| 60. Reconcile Maintained TE Quant Scope And Proof Truth | 1/1 | Complete | 2026-04-16 |
| 61. Refresh Validation And Closeout Audit | 1/1 | Complete | 2026-04-16 |

### Phase 59.1.1.1: Optimize image tower ARM kernel lowering (INSERTED)

**Goal:** Raise single-thread ARM image throughput on the maintained `embeddings/generator` path by
replacing the current per-pixel generic lowering in the vision tower with higher-throughput
image-shaped kernel/lowering work, then reprofile the real benchmark surface.
**Requirements**:
1. Keep all work on the maintained single-thread image request path through
   `src/emel/embeddings/generator`, not on helper-local or tool-only compute.
2. Attack the profiled highest-ROI hotspots first: universal inverted-block pointwise `1x1`
   lowering, then edge-residual spatial conv lowering, then shared f32 kernel packing cleanup only
   if the updated profile still justifies it.
3. Reprofile the maintained image benchmark after each landed step and publish updated stage and
   hotspot attribution before any lower-ROI cleanup.
**Depends on:** Phase 59.1.1
**Gap Closure**: Closes the newly profiled image-tower throughput gap where most single-thread ARM
time is still spent in many tiny generic f32 matmul invocations instead of image-shaped lowering
or kernel paths.
**Plans:** 59.1.1.1-01

Plans:
- [x] **59.1.1.1-01** - Batch the image tower pointwise `1x1` lowering through the maintained f32
  matmul kernel path, rebenchmark, and reprofile before taking any second optimization.
