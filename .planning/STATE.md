---
gsd_state_version: 1.0
milestone: v1.14
milestone_name: Benchmark Variant Organization
status: ready_for_phase_planning
stopped_at: started v1.14 benchmark variant organization
last_updated: "2026-04-21T09:36:00-05:00"
last_activity: 2026-04-21
progress:
  total_phases: 4
  completed_phases: 0
  total_plans: 4
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-21)

**Core value:** Prove real end-to-end behavior with explicit SML orchestration and parity-oriented
verification before widening API surface or model scope.
**Current focus:** v1.14 Benchmark Variant Organization.

## Current Position

Phase: Not started (roadmap defined)
Plan: -
Status: Milestone requirements and roadmap are defined; ready to start Phase 77.
Last activity: 2026-04-21

Progress: [----------] 0%

## Deferred Items

Items acknowledged and deferred at milestone close on 2026-04-21:

| Category | Item | Status |
|----------|------|--------|
| quick_task | 260401-ejm-add-non-blocking-benchmark-binary-size-c | missing |
| todo | 2026-04-02-move-eager-quant-prepack-into-generator-initializer.md | pending |
| todo | 2026-04-02-optimize-lfm2-5-q4-prefill-kernel.md | pending |
| todo | 2026-04-02-optimize-lfm2-5-q6-prefill-kernel.md | pending |
| todo | 2026-04-02-reuse-q8-rhs-across-lfm2-5-prefill-matmuls.md | pending |

## Performance Metrics

**Latest shipped milestone:**

- Milestone: v1.13 Pluggable Generative Parity Bench
- Completed phases: `8/8`
- Completed plans: `8/8`
- Audit status: `tech_debt` with no blocking gaps

**Current planning shape:**

- Active milestone: `v1.14 Benchmark Variant Organization`
- Latest shipped milestone: `v1.13`
- Next action: `$gsd-discuss-phase 77`
- Current blocker: none

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- `v1.13` promotes the deferred generation-scope follow-on from `v1.12` into the next active
  milestone, reusing the same lane-isolated pluggable compare architecture instead of inventing a
  separate tool family.

- Phase `69` established `generation_compare/v1` as the maintained shared JSONL contract for EMEL
  and C++ reference generation lanes while preserving the text benchmark output path.

- Phase `70` moved maintained generation workload truth into checked-in prompt/workload manifests
  and added explicit comparability labeling for single-lane workloads.

- Phase `71` put the maintained llama.cpp generation reference lane behind
  `tools/bench/reference_backends/llama_cpp_generation.json` and
  `scripts/bench_generation_reference_llama_cpp.sh` without widening `src/`.

- Phase `72` published `scripts/bench_generation_compare.sh` and
  `generation_compare_summary/v1` verdicts for `exact_match`, `bounded_drift`,
  `non_comparable`, `missing`, and `error`.

- Phase `73` added a closeout-proof regression that runs the operator-facing generation compare
  wrapper end to end on the maintained LFM2 workload manifest and reran `./scripts/quality_gates.sh`.

- Milestone `v1.14` starts from the observation that both generation and embedding compare
  workflows are modular, but adding new maintained variants still touches too much unrelated code.

- `v1.14` should make generation workloads and embedding benchmark cases data/registry-owned with
  deterministic discovery, validation, and proof that variant additions stay local.

- The new generation compare workflow must record prompt, formatter, seed, sampling, and stop
  metadata explicitly so cross-engine results remain reproducible and truthful.

- `v1.12` will treat parity and benchmark reference engines as pluggable backends under one
  canonical comparison contract instead of keeping separate Python-only and C++-only tool lanes.

- The new comparison architecture must keep the EMEL lane fully isolated from reference-engine
  model, tokenizer, cache, and runtime objects even when backend selection becomes pluggable.

- Phase `62` fixed the shared `embedding_compare/v1` contract as the single compare record schema
  for EMEL and reference lanes.

- Phase `63` established the maintained Python backend truth as stored upstream TE goldens with an
  optional live backend path for future direct Python engine checks.

- Phase `64` put the existing Liquid/mtmd C++ lane behind the same manifest-driven backend
  contract and kept its asset setup in the wrapper script.

- Phase `65` published the unified compare driver and wrapper workflow, including machine-readable
  compare summaries and focused test coverage.

- `v1.11` is pinned to `tests/models/TE-75M-q8_0.gguf` for the first maintained TE slice.
- TE support will land as explicit `omniembed` model-family work rather than an alias of the
  existing LLM runtime paths.

- `omniembed` model acceptance now validates modality-family prefixes and shared embedding
  metadata through its own architecture family rather than through the decoder contract.

- Phase 49 now proves the maintained TE text lane end to end through a repo-owned
  `embeddings/generator` actor rather than through generation-only runtime surfaces.

- Phase 49 vendors the upstream `mdbr-leaf-ir` WordPiece vocab as maintained tokenizer truth for
  TE tests because the pinned GGUF fixture omits tokenizer metadata.

- Phase 50 derives the maintained image preprocessing contract from the declared
  `mobilenetv4_conv_medium.e180_r384_in12k` encoder family instead of inventing CLIP-style GGUF
  preprocessing metadata.

- Phase 50 keeps the image lane on one documented in-memory RGBA contract and runs the real
  `image_encoder.*` tower plus shared projection head natively in `src/`.

- Phase 51 keeps the audio lane on one documented mono `float32` PCM contract, derives the
  EfficientAT `mn20_as` frontend from the declared encoder family, and runs the real
  `audio_encoder.features.*` tower plus shared projection head natively in `src/`.

- Phase 52 keeps the shared contract proof on the existing `embeddings/generator` actor and
  validates uniform normalization, truncation, and invalid-dimension rejection across text, image,
  and audio.

- Phase 53 keeps the TE proof on stored upstream goldens and tiny canonical smoke checks while the
  WPM encoder remains compatible with both the maintained TE vocab and the existing BERT GGUF
  parity fixture.

- The first milestone stops at synchronous text/image/audio embedding extraction and one shared
  output contract.

- Proof for TE will use stored upstream golden embeddings and tiny cross-modal smoke checks rather
  than the current generation-parity path.

- Phase 47 defines the canonical proof anchors as deterministic in-memory payload contracts:
  `red-square` for text/image and `pure-tone-440hz` for text/audio.

- Phase numbering continues from prior milestone history, so `v1.11` starts at Phase 47.
- The locked directory direction is `text/tokenizers/...`, `text/encoders/...`,
  `vision/encoders/...`, `audio/encoders/...`, and `embeddings/generator/...`.

- `*/forward/...` is not a required milestone domain; it becomes justified only when a modality
  has more than one top-level contract reusing the same hidden-state execution path.

- Phase 57 moved the last embedding-generator runtime outcome choice into explicit guards and
  transitions, and it locked maintained contract drift to `model_invalid`.

- Phase 58 established a maintained `tools/bench/embedding_generator_bench_runner` surface that
  exercises initialize plus real TE embedding requests through `embeddings/generator`.

- Phase 58.1 converted that publication lane into steady-state ARM throughput evidence with
  explicit `prepare` / `encode` / `publish` timing.

- Phase 58.1.1 added a maintained Liquid multimodal reference lane without introducing any
  llama.cpp dependency into the EMEL benchmark lane.

- Phase 59.1 is reserved for the urgent next pass on steady-state ARM embedding throughput before
  milestone archival resumes.

- Phase 59.1 uses the agreed external comparison set of `Arctic S`, `EmbeddingGemma 300M`,
  `LFM2-VL-450M`, and `Ultravox 1B` to keep future ARM throughput claims honest without widening
  the current EMEL runtime scope.

- Phase 59.1.1 runs that agreed external baseline set on the same ARM host and publishes the gap
  between EMEL and the reference lanes with explicit output-contract caveats.

- Phase 59.1.1.1 completed after the first ROI-ranked pass because maintained single-thread image
  throughput moved from `~1.51 embeddings/s` to `~4.27 embeddings/s`, and the next separate image
  hotspot shifted from per-pixel pointwise lowering to depthwise work.

- Phase 59.2 completed the maintained actor-level refactor that removed the old detail-side
  prepare/encode error-wrapper seam and replaced it with explicit route-readiness guards and
  transitions.

- Phase 59.4 completed the architecture-kernel ownership move for the maintained image pointwise
  and depthwise hot loops, putting the remaining image hotspot into `src/emel/kernel/aarch64`
  instead of leaving those long-lived hot loops in `embeddings/generator/detail.hpp`.

- Phase `59.3` executed and completed, republishing maintained q8/q5 image evidence at roughly
  `~10.5 embeddings/s` and narrowing the next image ROI to AArch64 pointwise, then depthwise.

- The 2026-04-16 rerun milestone audit reset `FIX-02` to pending because the live repo now
  accepted `TE-75M-q5_0.gguf` while `v1.11` still documented a q8-only maintained slice.

- Phase `60` executed and completed, reconciling maintained TE quant-scope truth around the
  approved q8/q5 fixtures and restoring `FIX-02` to satisfied.

- Phase `61` executed and completed, backfilling the remaining validation artifacts, rerunning the
  full maintained quality-gate surface, and refreshing the root milestone audit to a passing
  ledger.

### Roadmap Evolution

- Started milestone `v1.13` on 2026-04-20 with continued numbering, so the new roadmap begins at
  Phase `69`.

- Completed Phase `69` on 2026-04-20, landing the shared generation compare contract plus repo
  gate verification and leaving Phase `70` as the next active target.

- Completed Phase `70` on 2026-04-20, landing manifest-driven workload provenance and explicit
  single-lane comparability truth and leaving Phase `71` as the next active target.

- Completed Phases `71` through `73` on 2026-04-20, landing the manifest-selected maintained
  llama.cpp generation backend, the operator-facing compare workflow, verdict publication, and
  closeout-proof regression evidence.

- Completed Phases `74` through `76` on 2026-04-21, repairing JSONL lane isolation, adding real
  selected single-lane non-comparable publication, and backfilling requirement/Nyquist evidence.

- Archived milestone `v1.13` on 2026-04-21 after a no-blocker audit with accepted tech debt.

- Started milestone `v1.12` on 2026-04-17 with continued numbering, so the new roadmap begins at
  Phase `62`.

- Completed Phases `62` through `65` on 2026-04-17/18, leaving the milestone ready for audit.
- Completed Phase `66` on 2026-04-18, repairing the maintained multi-record C++ compare
  publication path.

- Completed Phase `67` on 2026-04-18, backfilling the reopened `v1.12` traceability and Nyquist
  evidence.

- `v1.10` remains ready for closeout.
- `v1.11` reopened at Phase `54` through `56` to close the milestone-audit gaps without deleting
  the shipped Phase `47` through `53` history.

- Phase `58.1` inserted after Phase `58`: ARM Embedding Generator Throughput Optimization
  (URGENT)

- Phase `58.1.1` inserted after Phase `58.1`: Liquid AI Multimodal Reference Throughput And
  Parity (URGENT)

- Phase `59` closed the stale state-ledger contradiction and reran the milestone audit to `passed`.
- Phase `59.1` inserted after Phase `59`: optimize for high throughput on ARM (URGENT)
- Phase `59.1` executed and completed, restoring the milestone to 100% phase/plan completion.
- Phase `59.1.1` inserted after Phase `59.1`: Run llama.cpp supported ARM throughput baseline
  comparison (URGENT)

- Phase `59.1.1` executed and completed, publishing same-host ARM reference evidence for `Arctic
  S`, `EmbeddingGemma 300M`, `LFM2-VL-450M`, and `Ultravox 1B`.

- Phase `59.1.1.1` inserted after Phase `59.1.1`: Optimize image tower ARM kernel lowering
  (URGENT)

- Phase `59.1.1.1` executed and completed, publishing the maintained single-thread image-tower
  lowering pass and the shifted hotspot profile.

- Phase `59.2` inserted after Phase `59`: Refactor Embedding Generator Hidden Control Flow Into
  Explicit Guards And Transitions (URGENT)

- Phase `59.2` executed and completed, removing the scoped helper-latched prepare/encode outcome
  seam from the maintained embedding generator.

- Phase `59.3` inserted after Phase `59`: Continue profiling and optimizing image performance
  (URGENT)

- Phase `59.4` inserted after Phase `59`: Kernelize image throughput hot path into AArch64
  architecture kernels (URGENT)

- Phase `59.4` executed and completed, moving the maintained image pointwise and depthwise ARM
  hot loops into `src/emel/kernel/aarch64` and publishing the refreshed image-only throughput and
  ownership evidence.

- Phase `60` added after the rerun audit: Reconcile Maintained TE Quant Scope And Proof Truth.
- Phase `61` added after the rerun audit: Refresh Validation And Closeout Audit.

### Pending Todos

- 2026-04-02 - Move eager quant prepack into generator initializer
- 2026-04-02 - Reuse q8 RHS across LFM2.5 prefill matmuls
- 2026-04-02 - Optimize LFM2.5 q4 prefill kernel
- 2026-04-02 - Optimize LFM2.5 q6 prefill kernel

### Blockers/Concerns

- `v1.12` starts before lifecycle-archiving `v1.11`, so any archive/cleanup step should remain a
  deliberate separate action from the new parity-bench work.

- The full repo gate for `v1.12` still tolerates warning-only benchmark snapshot regressions, so
  the compare-workflow milestone closes on a green gate with the existing warning note rather than
  on a benchmark-snapshot clean slate.

- `v1.10` still needs formal closeout and archival after its implementation-complete planner work.
- Benchmark snapshot and benchmark-marker issues remain warning-only under current gate policy, so
  published performance claims should continue to use a deliberate benchmark-review step rather
  than the raw gate tail alone.

- The latest maintained ARM evidence is now recorded in Phase `59.1`, with isolated per-modality
  steady-state runs rather than a contended all-modalities sample.

- The same-host external baseline evidence is now recorded in Phase `59.1.1`; the main remaining
  caveat is that the multimodal reference outputs are token-grid tower embeddings rather than the
  exact final normalized embedding contract used by EMEL.

- The highest remaining single-thread ARM ROI is still the image pointwise kernel now living in
  `src/emel/kernel/aarch64`; the latest maintained q5 image evidence is about `~10.5 embeddings/s`,
  so future image work still has clear ROI even though it is no longer a milestone blocker.

- `v1.11` lifecycle completion is still a separate step because this repo currently has a heavily
  dirty worktree; archival/cleanup should be deliberate rather than automatic.

## Session Continuity

Last session: 2026-04-18T00:20:00Z
Stopped at: completed the pluggable reference parity bench architecture, verified the unified
Python/C++ compare workflows, and left the milestone ready for audit
Resume file: None
