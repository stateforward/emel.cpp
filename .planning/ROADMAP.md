# Roadmap

## Archived Milestones

- [x] [v1.0: EMEL Llama-68M Generation Slice](.planning/milestones/v1.0-ROADMAP.md)
- [x] [v1.1: EMEL Llama-68M Generation Benchmark](.planning/milestones/v1.1-ROADMAP.md)
- [x] [v1.2: Flash Attention](.planning/milestones/v1.2-ROADMAP.md)
- [x] [v1.3: ARM Flash Optimizations](.planning/milestones/v1.3-ROADMAP.md)
- [x] [v1.4: Full Vectorized Quantized Kernels](.planning/milestones/v1.4-ROADMAP.md)
- [x] [v1.5: Full ARM Quantized Path](.planning/milestones/v1.5-ROADMAP.md)
- [x] [v1.6: Qwen3-0.6B Parity And Benchmark](.planning/milestones/v1.6-ROADMAP.md)
- [x] [v1.7: Generator Prefill Submachine Decomposition](.planning/milestones/v1.7-ROADMAP.md)
- [x] [v1.8: Truthful Qwen3 E2E Embedded Size](.planning/milestones/v1.8-ROADMAP.md)
- [x] [v1.9: Liquid LFM2.5-1.2B Thinking ARM Slice](.planning/milestones/v1.9-ROADMAP.md)
- [x] [v1.11: TE-75M GGUF Trimodal Embedding Runtime](.planning/milestones/v1.11-ROADMAP.md)
  - Shipped 2026-04-15 with maintained TE trimodal embedding runtime support, refreshed closeout
    evidence, and a passing milestone audit.
- [x] [v1.12: Pluggable Reference Parity Bench Architecture](.planning/milestones/v1.12-ROADMAP.md)
  - Shipped 2026-04-18, reopened narrowly for archived closeout-proof repair on 2026-04-19, and
    returned to a passing rerun audit on 2026-04-20.
- [x] [v1.13: Pluggable Generative Parity Bench](.planning/milestones/v1.13-ROADMAP.md)
  - Shipped 2026-04-21 with a maintained generative compare contract, workload manifests,
    `llama_cpp_generation` reference lane, truthful comparable/non-comparable publication, and a
    no-blocker audit.
- [x] [v1.14: Benchmark Variant Organization](.planning/milestones/v1.14-ROADMAP.md)
  - Shipped 2026-04-21 with deterministic data-owned benchmark variant discovery for generation
    and embedding variants.
- [x] [v1.15: ARM Sortformer Diarization GGUF Slice](.planning/milestones/v1.15-ROADMAP.md)
  - Shipped 2026-04-25 with one maintained native Sortformer diarization GGUF slice, PyTorch/NeMo
    parity, ONNX CPU single-thread benchmark reference, EMEL-over-ONNX performance closure, and a
    passing source-backed milestone audit.
- [x] [v1.16: ARM Whisper GGUF Parity And Performance](.planning/milestones/v1.16-ROADMAP.md)
  - Initial archive created 2026-04-26, reopened the same day because bounded transcript drift was
    not acceptable for an E2E milestone, then reopened again on 2026-04-27 after a source-backed
    audit rerun found artifact, benchmark-publication, runtime-surface, and evidence-ledger gaps;
    a later 2026-04-28 source-backed audit found the public recognizer route still hid behavior
    behind a runtime backend function-pointer table; Phase 126 repaired that blocker, but a later
    source-backed audit found the decoder runtime still executed through encoder-owned detail
    code. Phase 127 closed that ownership blocker and the latest source-backed audit passed.

## Current Milestone

**v1.16: ARM Whisper GGUF Parity And Performance**

Goal: Maintained Whisper tiny E2E path with no top-level Whisper runtime domain,
speech-domain encoder/decoder/tokenizer ownership, exact transcript parity against the pinned
`whisper.cpp` lane, ARM single-thread performance where EMEL beats the matched `whisper.cpp`
reference, and a source-backed closeout audit.

Current gap closure: Phases 123-127 repaired the public recognizer E2E path, cut maintained
parity/performance evidence over to that path, replaced hidden `runtime_backend`
function-pointer dispatch with explicit recognizer SML route graph behavior, and removed the
decoder actor's production dependency on `speech/encoder/whisper/detail.hpp`. The latest
source-backed audit found no blockers, but it identified non-blocking tech debt. Phases 128-129
closed that debt before archive/tag confirmation: Phase 128 stabilized benchmark evidence and
superseded prose, and Phase 129 removed stale encoder-owned decoder helper duplication.

## Phases

- [x] **Phase 94: Whisper Starting Point Backfill** - archived in
  `.planning/milestones/v1.16-phases/94-whisper-starting-point-backfill/`.
- [x] **Phase 95: Whisper Fixture And Contract Matrix** - archived in
  `.planning/milestones/v1.16-phases/95-whisper-fixture-and-contract-matrix/`.
- [x] **Phase 96: Native Quant Variant Kernels** - archived in
  `.planning/milestones/v1.16-phases/96-native-quant-variant-kernels/`.
- [x] **Phase 97: Whisper Audio Frontend And Encoder** - archived in
  `.planning/milestones/v1.16-phases/97-whisper-audio-frontend-and-encoder/`.
- [x] **Phase 98: Whisper Decoder And Transcript Runtime** - archived in
  `.planning/milestones/v1.16-phases/98-whisper-decoder-and-transcript-runtime/`.
- [x] **Phase 99: whisper.cpp Parity Lane** - archived in
  `.planning/milestones/v1.16-phases/99-whispercpp-parity-lane/`.
- [x] **Phase 100: Single-Thread CPU Benchmark Harness** - archived in
  `.planning/milestones/v1.16-phases/100-single-thread-cpu-benchmark-harness/`.
- [x] **Phase 101: ARM Profiling And Optimization** - archived in
  `.planning/milestones/v1.16-phases/101-arm-profiling-and-optimization/`.
- [x] **Phase 102: Whisper Closeout Evidence** - archived in
  `.planning/milestones/v1.16-phases/102-whisper-closeout-evidence/`.
- [x] **Phase 103: Speech Recognizer Domain Cleanup** - Complete on 2026-04-26. Moved top-level
  `src/emel/whisper/**` runtime actors into speech-domain ownership so Whisper remains a model
  family, not a custom public domain. Focused Whisper tests pass, and no `emel/whisper` include
  path remains.
- [x] **Phase 104: Speech Tokenizer And Decode Policy Contract** - Complete on 2026-04-26. Pinned
  and checksum-verified `tests/models/tokenizer-tiny.json`, added
  `speech/tokenizer/whisper`, and changed the EMEL compare/benchmark runner to detokenize through
  that speech tokenizer contract instead of publishing numeric transcript placeholders.
- [x] **Phase 105: Whisper Exact Transcript Parity Closure** - Complete as transition/evidence
  work only. Recorded normalized-GGUF bridge exact match, direct pinned-artifact failure, and
  handoff of final exact parity/closeout to Phases 107/108.
- [x] **Phase 106: Reopened Whisper Evidence Repair** - Complete on 2026-04-27. Closed reopened
  artifact and planning-state
  gaps for reopened Phases 103-105, but the 2026-04-27 audit rerun found Phase 109 is still
  required to supply phase-local verification and validation artifacts.
- [x] **Phase 107: Speech Tokenizer And Decode Policy Hardening** - Complete on 2026-04-27.
  Enforced the pinned
  `tokenizer-tiny.json` contract before maintained dispatch and model Whisper ASR decode policy as
  an explicit speech-domain contract; Phase 115 later superseded stale recognizer-route evidence.
- [x] **Phase 108: Pinned Whisper Artifact Parity Closeout** - Complete on 2026-04-27. Replaced
  the default bench-only normalized bridge with the user-approved Option B contract: source-owned
  legacy Whisper `lmgg` conversion in `src/emel/model/whisper`, exact transcript parity through
  the maintained EMEL speech encoder/decoder/tokenizer surface, and exact parity evidence;
  Phase 116 supplied final closeout after later audit blockers.
- [x] **Phase 109: Reopened Whisper Artifact Evidence Closure** - Complete on 2026-04-27.
  Backfilled source-backed Phase 106 verification and validation artifacts and reconciled
  REOPEN-01/SPEECH-01 evidence without falsely claiming tokenizer, parity, benchmark, or closeout
  scope.
- [x] **Phase 110: Maintained Whisper Benchmark Publication Repair** - Complete on 2026-04-27.
  Connected the single-thread Whisper benchmark EMEL lane to the pinned Phase 99 source model path,
  matched deterministic reference policy, and made benchmark publication fail on model, transcript,
  iteration, warmup, or missing transcript contradictions.
- [x] **Phase 111: Speech Recognizer SML Rule Readiness Repair** - Superseded on 2026-04-27.
  Phase 115 corrected this as historical recognizer-route evidence; Phase 114 owns current
  runtime-surface truth.
- [x] **Phase 112: Reopened Whisper Closeout Rerun** - Superseded on 2026-04-27. Phase 116 owns
  final closeout after runtime-surface and evidence repairs.
- [x] **Phase 113: Recursive Whisper ARM Profile And Optimize Closure** - Superseded on
  2026-04-27. The stale plan was retired; active runtime-surface, evidence, and final closeout
  gaps moved to Phases 114-116.
- [x] **Phase 114: Whisper Runtime Surface Contract Repair** - Complete on 2026-04-27. Defined the
  maintained runtime surface as the speech encoder/decoder/tokenizer Whisper path, updated
  compare/benchmark metadata, and verified exact `[C]` parity.
- [x] **Phase 115: Whisper Evidence Truth Repair** - Complete on 2026-04-27. Corrected or
  superseded false Phase 103, 107, 108, 111, 112, and stale Phase 113 artifacts so active
  milestone evidence matches live source.
- [x] **Phase 116: Whisper Final Closeout Rerun** - Complete on 2026-04-27. Reran source-backed
  closeout; a later source-backed audit found remaining compare-gate, harness-boundary, policy,
  and Nyquist-ledger blockers.
- [x] **Phase 117: Whisper Compare Failure Contract Repair** - Complete on 2026-04-27. Maintained
  Whisper transcript drift now exits nonzero while exact match remains the only successful compare
  status.
- [x] **Phase 118: Whisper Public Runtime Harness Boundary Repair** - Complete on 2026-04-27.
  Moved maintained parity/benchmark proof off actor detail APIs and reconciled speech-owned
  runtime, tokenizer, decode-policy, parity, and performance evidence through public event
  interfaces.
- [x] **Phase 119: Whisper Final Source-Backed Closeout Rerun** - Complete on 2026-04-27.
  Corrected the Phase 113 Nyquist ledger and reran final closeout after Phases 117-118.
- [x] **Phase 120: Whisper Decode Policy And Transcript Runtime Repair** - Complete on
  2026-04-27. Wired the speech-owned ASR decode-policy contract into decoder runtime behavior and
  removed the hardcoded public decoder `token:<id>` transcript surface so `POLICY-01` and
  `TOK-02` are source-backed.
- [x] **Phase 121: Whisper Baseline Nyquist Validation Backfill** - Complete on 2026-04-27. Added
  archived-baseline validation artifacts for preserved baseline Phases 94-102 without giving them
  active runtime credit beyond their archived scope.
- [x] **Phase 122: Whisper Final Gap Closeout Rerun** - Complete on 2026-04-27. Reran
  source-backed closeout after Phases 120-121, cited stable warmed benchmark evidence, updated
  the audit, and previously closed `CLOSE-01` before later public-recognizer audits superseded
  that closeout.
- [x] **Phase 123: Whisper Public Recognizer Runtime Wiring** - Complete. The maintained
  Whisper model, tokenizer, decode-policy, encoder, decoder, and transcript publication path now
  has a public recognizer route through `emel::speech::recognizer::sm`.
- [x] **Phase 124: Whisper Recognizer Compare And Benchmark Cutover** - Complete on 2026-04-28.
  The maintained EMEL compare and single-thread benchmark lanes now drive the public recognizer
  actor and publish recognizer-backed parity/performance records.
- [x] **Phase 125: Whisper Final Recognizer Closeout Rerun** - Complete on 2026-04-28. Reran
  source-backed audit and full relevant quality gates after recognizer-backed parity and benchmark
  evidence; the later behavior-selection audit was closed by Phase 126.
- [x] **Phase 126: Whisper Recognizer Explicit Route Graph Repair** - Complete on 2026-04-28.
  Replaced hidden `runtime_backend` function-pointer route selection/execution with explicit
  recognizer SML route states, guards, transitions, compile-time route policy wiring, and
  rule-focused regression tests.
- [x] **Phase 127: Whisper Decoder Ownership Gap Closure** - Complete on 2026-04-28. Rewired the
  decoder actor to use decoder-owned detail for sequence/logit/timestamp-policy runtime
  execution, added a source-level ownership regression, and reran source-backed closeout gates.
- [x] **Phase 128: Whisper Benchmark And Closeout Evidence Cleanup** - Complete on 2026-04-28.
  Stabilized the default Whisper single-thread benchmark closeout path with a 20-iteration sample
  and repaired superseded Phase 122/125 closeout prose so the evidence ledger matches Phase 127
  truth.
- [x] **Phase 129: Whisper Detail Helper Deduplication Cleanup** - Complete on 2026-04-28.
  Removed unused duplicate decoder/timestamp helper code from encoder detail while preserving
  decoder-owned runtime execution and SML behavior-selection rules.

## Phase Details

### Phase 94: Whisper Starting Point Backfill

**Goal:** Preserve the audited v1.16 starting point and make the pre-reopen Whisper work
traceable from archived artifacts.
**Requirements:** Archived v1.16 baseline requirements.
**Success Criteria**:
1. Archived SUMMARY and VERIFICATION artifacts remain available.
2. Starting-point claims are treated as historical evidence, not reopened closeout proof.

### Phase 95: Whisper Fixture And Contract Matrix

**Goal:** Preserve the archived Whisper fixture and model-contract matrix evidence.
**Requirements:** Archived v1.16 fixture and contract requirements.
**Success Criteria**:
1. Archived fixture/contract evidence remains available for reference.
2. Reopened phases verify maintained-path claims against live source before closeout.

### Phase 96: Native Quant Variant Kernels

**Goal:** Preserve archived native quant-kernel evidence for the maintained Whisper slice.
**Requirements:** Archived v1.16 kernel requirements.
**Success Criteria**:
1. Archived kernel proof remains available.
2. Reopened parity work does not replace native kernels with whole-tensor fallback paths.

### Phase 97: Whisper Audio Frontend And Encoder

**Goal:** Preserve archived Whisper audio frontend and encoder evidence.
**Requirements:** Archived v1.16 ASR encoder requirements.
**Success Criteria**:
1. Archived encoder proof remains available.
2. Reopened work keeps runtime behavior routed through maintained speech recognizer actors.

### Phase 98: Whisper Decoder And Transcript Runtime

**Goal:** Preserve archived Whisper decoder and transcript runtime evidence.
**Requirements:** Archived v1.16 decoder requirements.
**Success Criteria**:
1. Archived decoder proof remains available.
2. Reopened transcript work is verified against maintained tokenizer and parity contracts.

### Phase 99: whisper.cpp Parity Lane

**Goal:** Preserve the archived isolated `whisper.cpp` parity lane as the reference-side
comparison path for reopened closeout.
**Requirements:** Archived v1.16 parity lane requirements.
**Success Criteria**:
1. Reference-side `whisper.cpp` evidence remains available.
2. Reopened EMEL parity proof does not depend on reference-owned runtime objects.

### Phase 100: Single-Thread CPU Benchmark Harness

**Goal:** Preserve archived single-thread CPU benchmark harness evidence.
**Requirements:** Archived v1.16 benchmark requirements.
**Success Criteria**:
1. Archived benchmark proof remains available.
2. Reopened closeout reruns full relevant gates before archival.

### Phase 101: ARM Profiling And Optimization

**Goal:** Preserve archived ARM profiling and optimization evidence for the Whisper slice.
**Requirements:** Archived v1.16 performance requirements.
**Success Criteria**:
1. Archived profiling proof remains available.
2. Reopened changes preserve or improve maintained runtime performance.

### Phase 102: Whisper Closeout Evidence

**Goal:** Preserve the initial v1.16 closeout evidence while acknowledging it is superseded by
the reopened exact-transcript parity requirement.
**Requirements:** Archived v1.16 closeout requirements.
**Success Criteria**:
1. Archived closeout evidence remains available.
2. Reopened audit does not treat bounded-drift closeout evidence as sufficient.

### Phase 103: Speech Recognizer Domain Cleanup

**Goal:** Verify and document the removal of the top-level Whisper runtime domain and the
speech-domain ownership of Whisper recognizer actors.
**Requirements:** REOPEN-01, SPEECH-01.
**Success Criteria**:
1. No `src/emel/whisper/**` runtime domain or `emel/whisper` include path remains.
2. Whisper runtime actors and fused ASR detail live under speech recognizer ownership while model
   binding stays in `model/whisper`.
3. Source-backed verification and Nyquist validation artifacts exist for the reopened cleanup.

### Phase 104: Speech Tokenizer And Decode Policy Contract

**Goal:** Verify and document the speech tokenizer contract introduced for Whisper transcript
publication.
**Requirements:** Historical tokenizer contract evidence only; active TOK-01, TOK-02, and
POLICY-01 remain Phase 107.
**Success Criteria**:
1. The maintained `tokenizer-tiny.json` asset exists with a pinned checksum.
2. Transcript publication uses speech tokenizer/detokenizer machinery rather than fixture text or
   numeric token placeholders.
3. Verification documents that pre-dispatch checksum enforcement and explicit decode-policy
   hardening remain Phase 107 scope.

### Phase 105: Whisper Exact Transcript Parity Closure

**Goal:** Represent Phase 105 truthfully as transition/evidence cleanup that records the
normalized-GGUF bridge exact match, the direct pinned-artifact failure, and the handoff of final
exact parity/closeout to Phases 107/108.
**Requirements:** None for active traceability; PARITY-01 and CLOSE-01 remain Phase 108.
**Success Criteria**:
1. Phase 105 SUMMARY, VERIFICATION, and VALIDATION artifacts exist and claim no active
   requirements.
2. Bridge exact-match evidence is recorded separately from direct pinned-artifact failure with
   command lines and SHA values.
3. ROADMAP.md and STATE.md no longer claim Phase 105 closed exact pinned-artifact parity or made
   v1.16 ready.
4. No source runtime, tool compute, script, test, or kernel changes are attributed to Phase 105
   transition closure.

### Phase 106: Reopened Whisper Evidence Repair

**Goal:** Close audit artifact and planning-state gaps for reopened Phases 103-105.
**Requirements:** Historical REOPEN-01 and SPEECH-01 evidence attempt; active traceability now
maps reopened evidence closure to Phase 109.
**Success Criteria**:
1. ROADMAP.md and STATE.md no longer contradict the actual reopened v1.16 readiness state.
2. Phase 103 and Phase 104 have source-backed VERIFICATION and VALIDATION artifacts.
3. Phase 105's superseded or remaining scope is represented truthfully before closeout continues.
4. REOPEN-01 and SPEECH-01 completion claims are superseded by the Phase 109 artifact closure
   plan after the 2026-04-27 audit rerun found missing phase-local evidence.

### Phase 107: Speech Tokenizer And Decode Policy Hardening

**Goal:** Enforce the maintained tokenizer asset and make Whisper ASR decode policy explicit in
the speech recognizer path.
**Requirements:** TOK-02, plus historical TOK-01/POLICY-01 hardening evidence now superseded by
Phase 111 for remaining SML rule-readiness.
**Success Criteria**:
1. The pinned `tokenizer-tiny.json` checksum is enforced before maintained compare/runtime
   dispatch.
2. Prompt sequence, language/task roles, timestamp mode, and suppression behavior are represented
   as an explicit speech-domain decode policy contract.
3. Recognizer initialization no longer allocates route buffers or child machines during SML
   dispatch.
4. Exact transcript parity remains Phase 108 scope.

### Phase 108: Pinned Whisper Artifact Parity Closeout

**Goal:** Prove exact transcript parity through the maintained EMEL runtime path against the pinned
Phase 99 `whisper.cpp` audio/model pair.
**Requirements:** PARITY-01; CLOSE-01 final closeout rerun now maps to Phase 116.
**Success Criteria**:
1. User approval for Option B is recorded as the final v1.16 closeout contract.
2. The EMEL lane consumes the pinned source model path through a source-owned legacy Whisper
   conversion path in `src/emel/model/whisper`.
3. Exact transcript parity is proven without a bench-only normalized-GGUF bridge being presented
   as direct pinned-artifact parity.
4. Full closeout quality gates and source-backed audit evidence from Phase 108 are superseded by
   Phases 112 and 113 because later audit reruns found remaining closeout and performance
   blockers.

**Completion Evidence**:
- `build/whisper_compare/summary.json` records `comparison_status=exact_match`, EMEL transcript
  `[C]`, reference transcript `[C]`, source model SHA
  `9ade048c9d3692b411572a9a8ad615766168e62fb1d4c234973825a377c71984`, and
  `model_normalization: {}`.
- `EMEL_QUALITY_GATES_SCOPE=full EMEL_QUALITY_GATES_BENCH_SUITE=whisper_compare
  scripts/quality_gates.sh` passed on 2026-04-27 with 90.6% line coverage, 55.6% branch
  coverage, paritychecker, fuzz, Whisper compare, and docsgen.

### Phase 109: Reopened Whisper Artifact Evidence Closure

**Goal:** Close the Phase 106 artifact gap found by the milestone audit with phase-local,
source-backed verification and validation for the reopened evidence repair work.
**Requirements:** REOPEN-01, SPEECH-01.
**Gap Closure:** Closes audit gaps for missing `106-VERIFICATION.md`, missing
`106-VALIDATION.md`, and the requirement traceability contradiction for REOPEN-01 and SPEECH-01.
**Success Criteria**:
1. Phase 106 has source-backed verification evidence for the reopened bounded-drift blocker,
   speech-domain ownership, and the absence of stale top-level Whisper runtime paths.
2. Phase 106 has Nyquist validation evidence with executable commands and rule-compliance notes.
3. REQUIREMENTS.md, ROADMAP.md, STATE.md, and the next milestone audit agree that REOPEN-01 and
   SPEECH-01 are either pending or source-backed by the assigned closure phase.
4. No tokenizer, parity, benchmark, or closeout requirement is falsely claimed by the artifact
   backfill.

**Completion Evidence**:
- `106-VERIFICATION.md` and `106-VALIDATION.md` now exist with source-backed evidence.
- `109-VERIFICATION.md` and `109-VALIDATION.md` passed for REOPEN-01 and SPEECH-01.

### Phase 110: Maintained Whisper Benchmark Publication Repair

**Goal:** Repair the maintained single-thread Whisper benchmark publication lane so EMEL and the
reference consume the same pinned Phase 99 source model contract and mismatches cannot publish as
`ok`.
**Requirements:** Historical CLOSE-01 publication repair; active CLOSE-01 now maps to Phase 116.
**Gap Closure:** Closes audit integration and flow gaps for benchmark model-path truth and
benchmark summary masking.
**Success Criteria**:
1. `scripts/bench_whisper_single_thread.sh` defaults the EMEL lane to the pinned Phase 99 source
   model path, not `tests/models/model-tiny-q80.gguf`.
2. The EMEL benchmark lane reaches the same source-owned legacy Whisper conversion path used by
   the maintained parity compare lane before GGUF binding and speech recognizer dispatch.
3. `tools/bench/whisper_benchmark.py` hard-fails claimed publication when EMEL/reference model
   SHA or transcript differs.
4. Focused benchmark tests and the Whisper benchmark wrapper prove the repaired publication
   contract.

**Completion Evidence**:
- `build/whisper_compare_tools/whisper_benchmark_tests` passed with 6 test cases and 86
  assertions.
- `build/whisper_benchmark/benchmark_summary.json` records `status: ok`, transcript `[C]` for
  both lanes, and model SHA
  `9ade048c9d3692b411572a9a8ad615766168e62fb1d4c234973825a377c71984` for both lanes.

### Phase 111: Speech Recognizer SML Rule Readiness Repair

**Goal:** Resolve SML/detail rule risks in the maintained Whisper recognizer path by making
runtime readiness decisions explicit in guards/transitions instead of detail helper outputs called
from dispatch-critical actions or guards.
**Requirements:** TOK-01, POLICY-01.
**Gap Closure:** Closes audit rule-readiness blockers for tokenizer readiness validation and
Whisper execution-contract acceptance.
**Success Criteria**:
1. Tokenizer checksum and control-token readiness outcomes are represented by explicit
   speech-recognizer guards/transitions without using detail helper output to decide what happens
   next.
2. Whisper execution-contract acceptance is represented by explicit guards/transitions before
   action execution mutates route state.
3. Actions only execute already-selected behavior paths and do not call helpers whose outputs
   select success/error or routing outcomes.
4. Focused SML introspection and recognizer lifecycle tests prove the repaired graph and
   unexpected-event behavior.

**Completion Evidence**:
- `build/audit-native/emel_tests_bin --no-breaks --source-file='*tests/speech/*'` passed with 6
  test cases and 1072 assertions.
- Recognizer route context no longer stores initialize-event model/tokenizer payload views.

### Phase 112: Reopened Whisper Closeout Rerun

**Goal:** Re-run closeout after Phases 109-111 repair artifact, benchmark, and rule-readiness
blockers.
**Requirements:** Historical CLOSE-01 attempt; active CLOSE-01 and PERF-03 now map to Phase 116.
**Gap Closure:** Superseded by Phase 116 after later runtime-surface and evidence-ledger repairs.
**Success Criteria**:
1. Full closeout quality gates pass with `EMEL_QUALITY_GATES_SCOPE=full` and
   `EMEL_QUALITY_GATES_BENCH_SUITE=whisper_compare`.
2. The maintained Whisper compare and single-thread benchmark summaries both record the pinned
   Phase 99 source model contract truthfully.
3. A source-backed milestone audit rerun reports all requirements satisfied, all integration flows
   passed, and no missing Nyquist validation artifacts.
4. ROADMAP.md, REQUIREMENTS.md, STATE.md, and the milestone audit agree whether v1.16 is ready to
   archive or still blocked.

**Completion Evidence**:
- Superseded by Phase 116 final closeout evidence.

### Phase 113: Recursive Whisper ARM Profile And Optimize Closure

**Goal:** Restore the v1.16 performance contract by recursively profiling and optimizing the
maintained ARM Whisper runtime until EMEL beats the matched single-thread CPU `whisper.cpp`
reference lane, then rerun source-backed closeout.
**Requirements:** No active requirements; historical/stale attempt only.
**Gap Closure:** Superseded by Phases 114-116.
**Status Note:** The stale implementation plan was retired and not executed.
**Success Criteria**:
1. Phase 113 summary and verification record supersession.
2. `CLOSE-01` and `PERF-03` remain owned by Phase 116.
3. The stale plan no longer directs implementation work.

### Phase 114: Whisper Runtime Surface Contract Repair

**Goal:** Make the maintained Whisper ASR runtime surface source-backed, domain-boundary clean, and
extendable by reconciling the top-level recognizer claim with the live encoder/decoder/tokenizer
runtime path.
**Requirements:** SPEECH-01, TOK-01, TOK-02, POLICY-01, PARITY-01.
**Gap Closure:** Closes audit gaps for public recognizer E2E wiring, tokenizer/policy/parity
claims that are currently satisfied only by the benchmark runner, and the domain extensibility
decision for the maintained Whisper slice.
**Success Criteria**:
1. The maintained v1.16 ASR runtime surface is explicit and source-backed as the speech
   encoder/decoder actor pair plus tokenizer policy.
2. The chosen runtime surface preserves `scripts/check_domain_boundaries.sh` and does not add
   forbidden generic-recognizer or model-family runtime roots.
3. The compare and benchmark EMEL lanes exercise the chosen maintained runtime surface rather than
   a contradictory tool-local route.
4. Tokenizer checksum validation, decode policy, model binding, transcript publication, and exact
   `[C]` parity evidence are verified through the chosen runtime surface with focused tests.

### Phase 115: Whisper Evidence Truth Repair

**Goal:** Repair the milestone evidence ledger so completed phase artifacts match live source and
no phase verification claims missing recognizer-route files, forbidden paths, or recognizer
dispatch behavior that is not present.
**Requirements:** Historical evidence repair only; no active requirement ownership.
**Gap Closure:** Closes audit phase-artifact and Nyquist gaps for invalid Phase 103, 107, 108,
111, 112 evidence and the stale Phase 113 plan/context.
**Success Criteria**:
1. Phase 103 evidence no longer claims old recognizer-internal actor paths or a model-family
   kernel root.
2. Phase 107 and Phase 111 evidence no longer claims missing recognizer-route files or symbols.
3. Phase 108 evidence names the selected speech encoder/decoder/tokenizer runtime surface.
4. Phase 112 validation and Phase 113 plan/context are superseded or corrected so ROADMAP,
   REQUIREMENTS, STATE, and audit evidence agree with live benchmark and runtime truth.

### Phase 116: Whisper Final Closeout Rerun

**Goal:** Close v1.16 with a source-backed final rerun after runtime-surface and evidence repairs.
**Requirements:** CLOSE-01, PERF-03.
**Gap Closure:** Closes the remaining closeout ledger, benchmark performance, Nyquist validation,
and milestone audit gaps.
**Success Criteria**:
1. Phase 114 and Phase 115 have SUMMARY, VERIFICATION, and VALIDATION artifacts with executable
   evidence.
2. The maintained compare summary records exact transcript parity on the pinned Phase 99
   model/audio pair through the selected runtime surface.
3. The maintained single-thread benchmark summary records `status: ok`, `reason: ok`, matching
   model/transcript truth, and EMEL mean process wall time strictly below the matched
   `whisper.cpp` reference mean.
4. Full relevant quality gates pass with the Whisper benchmark suite and a domain-boundary check.
5. ROADMAP.md, REQUIREMENTS.md, STATE.md, and the source-backed milestone audit agree that
   `CLOSE-01` and `PERF-03` are complete before archival.

### Phase 117: Whisper Compare Failure Contract Repair

**Goal:** Make maintained Whisper transcript drift fail the compare and quality-gate path instead
of publishing a successful `bounded_drift` result.
**Requirements:** REOPEN-01.
**Gap Closure:** Closes audit gaps `COMPARE-DRIFT-GATE` and `Parity failure flow`.
**Success Criteria**:
1. `tools/bench/whisper_compare.py` returns failure for any non-exact transcript comparison,
   including `bounded_drift`.
2. Focused tests prove exact match still passes and transcript mismatch fails with a deterministic
   machine-readable reason.
3. `scripts/bench_whisper_compare.sh` and the `whisper_compare` quality-gate suite fail when the
   maintained EMEL/reference transcripts diverge.
4. Updated verification evidence maps the no-drift gate back to `REOPEN-01`.

**Completion Evidence**:
- `build/whisper_compare_tools/whisper_benchmark_tests` passed with 9 test cases and 130
  assertions.
- `scripts/bench_whisper_compare.sh --skip-reference-build --skip-emel-build` still reports
  `status=exact_match reason=ok`.
- Changed-file scoped `scripts/quality_gates.sh` with
  `EMEL_QUALITY_GATES_BENCH_SUITE=whisper_compare` passed.

### Phase 118: Whisper Public Runtime Harness Boundary Repair

**Status:** Complete.

**Goal:** Repair the maintained Whisper parity/benchmark EMEL lane so proof code drives only
public actor event interfaces and does not reach actor `detail.hpp` helpers directly.
**Requirements:** SPEECH-01, TOK-02, POLICY-01, PARITY-01, PERF-03.
**Gap Closure:** Closes audit gaps `HARNESS-DETAIL-API`, `POLICY-FIELD-WIRING`, and the
`Public actor interface flow`.
**Success Criteria**:
1. `tools/bench/whisper_emel_parity_runner.cpp` no longer directly includes or calls actor
   `detail.hpp` or `detail.cpp` helpers for model binding, encoder/decoder buffer sizing,
   tokenizer policy, or transcript decode.
2. The maintained EMEL lane drives the speech-owned Whisper encoder/decoder/tokenizer runtime
   through public event interfaces and `process_event(...)` only.
3. Whisper ASR decode-policy evidence is either fully wired into maintained behavior or narrowed
   in source-backed requirements and verification to the fields that actually affect behavior.
4. Exact `[C]` parity and EMEL-faster single-thread benchmark evidence still pass after the
   public-interface repair.

**Completion Evidence**:
- `tools/bench/whisper_emel_parity_runner.cpp` now uses public Whisper model/speech `any.hpp`
  surfaces plus encoder/decoder `process_event(...)`; a focused doctest guards against direct
  detail-header regressions.
- The maintained decode policy is source-backed as `english` / `transcribe` /
  `timestamp_tokens` / `suppress_translate=true` with 3 prompt tokens; no-timestamps behavior is
  explicitly not claimed for the `[C]` lane.
- `scripts/bench_whisper_compare.sh --skip-reference-build --skip-emel-build` reports
  `status=exact_match reason=ok`.
- `EMEL_WHISPER_BENCH_WARMUPS=0 EMEL_WHISPER_BENCH_ITERATIONS=1
  scripts/bench_whisper_single_thread.sh --skip-reference-build --skip-emel-build` reports
  `benchmark_status=ok reason=ok`, with EMEL mean `62020084 ns` and reference mean `66998708 ns`.
- Changed-file scoped `scripts/quality_gates.sh` with
  `EMEL_QUALITY_GATES_BENCH_SUITE=whisper_compare` passed.

### Phase 119: Whisper Final Source-Backed Closeout Rerun

**Status:** Complete.

**Goal:** Rerun final v1.16 closeout after the compare-failure and public-harness repairs, with
truthful Nyquist and benchmark evidence.
**Requirements:** CLOSE-01.
**Gap Closure:** Closes the remaining audit closeout blocker and Phase 113 Nyquist-ledger gap.
**Success Criteria**:
1. Phase 113 is no longer counted as Nyquist-compliant without a validation artifact; either add
   the missing validation evidence or document the superseded exception truthfully in the final
   audit.
2. Full relevant quality gates include the maintained compare proof and the single-thread Whisper
   benchmark proof needed for `CLOSE-01`.
3. ROADMAP.md, REQUIREMENTS.md, STATE.md, phase artifacts, benchmark summaries, and the milestone
   audit agree on the final v1.16 status.
4. `$gsd-audit-milestone` passes without source-backed maintained-path contradictions.

**Completion Evidence**:
- Phase 113 now has `113-VALIDATION.md` documenting the superseded retirement truth without
  crediting it for runtime implementation.
- Full closeout quality gate passed with `EMEL_QUALITY_GATES_SCOPE=full` and
  `EMEL_QUALITY_GATES_BENCH_SUITE=whisper_compare`.
- The final maintained compare summary reports `status=exact_match reason=ok` with EMEL and
  reference transcripts `[C]`.
- The final maintained benchmark summary reports `status=ok reason=ok`, EMEL mean `59049750 ns`,
  and reference mean `63237291 ns`.
- The 2026-04-27T21:02 audit reported `status: passed`; a later source-backed rerun at
  2026-04-27T21:45 found remaining decode-policy, tokenizer-backed decoder transcript, and
  preserved-baseline Nyquist gaps now planned in Phases 120-122.

### Phase 120: Whisper Decode Policy And Transcript Runtime Repair

**Goal:** Make the maintained Whisper decoder runtime consume a speech-owned ASR decode-policy
contract and make transcript publication tokenizer-owned end to end.
**Requirements:** TOK-02, POLICY-01.
**Gap Closure:** Closes audit gaps for decode-policy wiring and the public decoder
`token:<id>` transcript surface.
**Success Criteria**:
1. The decoder public event carries the speech-owned ASR decode-policy contract or a narrowed
   policy-owned runtime payload, not just prompt tokens.
2. Timestamp and suppression behavior choices are modeled by explicit SML guards/states or
   explicit already-chosen action paths, not hardcoded detail-side behavior selection.
3. The decoder no longer publishes a hardcoded `token:<id>` transcript surface; transcript text is
   either removed from decoder output or produced only through `speech/tokenizer/whisper`.
4. Focused decoder/tokenizer/benchmark tests prove exact `[C]` parity still passes, transcript
   drift still fails, and policy fields in compare JSON describe behavior actually consumed by the
   decoder runtime.

### Phase 121: Whisper Baseline Nyquist Validation Backfill

**Goal:** Backfill truthful Nyquist validation artifacts for preserved baseline Phases 94-102.
**Requirements:** Closeout support only; active requirement closure remains Phase 122.
**Gap Closure:** Closes audit phase-artifact gap `NYQUIST-MISSING-BASELINE`.
**Success Criteria**:
1. Phases 94-102 each have a `*-VALIDATION.md` artifact with executable evidence or an explicit
   archived-baseline validation scope.
2. Validation artifacts do not re-credit superseded baseline claims for final runtime, parity,
   benchmark, tokenizer, or closeout requirements now owned by later phases.
3. Each validation artifact records rule-compliance notes and no unresolved manual-only blocker.
4. A focused artifact scan proves all v1.16 phase directories have SUMMARY, VERIFICATION, and
   VALIDATION artifacts where required by the active Nyquist workflow.

### Phase 122: Whisper Final Gap Closeout Rerun

**Status:** Complete.

**Goal:** Rerun v1.16 source-backed closeout after decode-policy/transcript and baseline
validation repairs.
**Requirements:** CLOSE-01.
**Gap Closure:** Closes final audit blockers for `POLICY-01`, `TOK-02`, `CLOSE-01`, and
preserved-baseline Nyquist coverage.
**Success Criteria**:
1. Phase 120 and Phase 121 have SUMMARY, VERIFICATION, and VALIDATION artifacts.
2. Maintained compare exact-matches `[C]` and proves policy JSON fields match runtime behavior.
3. Maintained single-thread benchmark evidence uses the default warmed multi-iteration wrapper or
   immutable archived benchmark evidence, not the volatile zero-warmup one-iteration citation.
4. Full relevant quality gates, domain-boundary checks, and source-backed milestone audit rerun
   pass before `CLOSE-01` is re-marked complete.

**Completion Evidence**:
- `scripts/check_domain_boundaries.sh` passed and forbidden-root grep returned no matches.
- `scripts/bench_whisper_compare.sh --skip-reference-build --skip-emel-build` reported
  `status=exact_match reason=ok`.
- `scripts/bench_whisper_single_thread.sh --skip-reference-build --skip-emel-build` reported
  `benchmark_status=ok reason=ok` with EMEL mean `70709972 ns` below reference mean
  `81716555 ns`.
- Full closeout quality gate passed with
  `EMEL_QUALITY_GATES_SCOPE=full` and
  `EMEL_QUALITY_GATES_BENCH_SUITE=whisper_compare:whisper_single_thread`; coverage was line
  `90.8%`, branch `55.6%`.
- The 2026-04-27 audit reported `status: passed`; the 2026-04-28 source-backed rerun supersedes
  that result with a public-recognizer bypass blocker now planned in Phases 123-125.

### Phase 123: Whisper Public Recognizer Runtime Wiring

**Status:** Complete.

**Goal:** Make the maintained Whisper E2E ASR path run through the public speech recognizer actor
instead of tool-local encoder/decoder/tokenizer orchestration.
**Requirements:** SPEECH-01, TOK-01, TOK-02, POLICY-01.
**Gap Closure:** Closes audit gaps `PUBLIC-RECOGNIZER-BYPASS`,
`RECOGNIZER-BACKEND-DISABLED`, recognizer tokenizer checksum, recognizer decode policy, and
recognizer transcript publication.
**Success Criteria**:
1. `emel::speech::recognizer::sm` can initialize the maintained Phase 99 Whisper model/tokenizer
   route through explicit guards/transitions without generic recognizer Whisper leakage.
2. Recognizer dispatch validates the pinned tokenizer asset identity and ASR decode-policy support
   before selected execution.
3. Recognizer recognition dispatch owns the maintained encoder, decoder, and tokenizer-backed
   transcript publication flow using public child actor events and `process_event(...)`.
4. Focused recognizer tests prove the Phase 99 model/audio/tokenizer path succeeds through public
   recognizer events and no longer asserts that the maintained Whisper route is unsupported.

**Completion Evidence**:
- Generic recognizer tests pass with a model-family-free backend route and no `whisper` matches in
  `src/emel/speech/recognizer` or `tests/speech/recognizer`.
- A focused Whisper fixture test proves `emel::speech::recognizer::sm` initializes the maintained
  model/tokenizer route and runs recognition through public recognizer events.
- `scripts/check_domain_boundaries.sh` passed, and forbidden-root grep returned no matches.
- Changed-file scoped `scripts/quality_gates.sh` passed with
  `EMEL_QUALITY_GATES_BENCH_SUITE=whisper_compare`; changed-file coverage was line `100.0%`,
  branch `64.5%`.
- Phase 124 cut compare and benchmark proof tools over from the old bypass runner to this
  recognizer route.

### Phase 124: Whisper Recognizer Compare And Benchmark Cutover

**Goal:** Make maintained parity and performance proof consume the public recognizer runtime path.
**Requirements:** REOPEN-01, PARITY-01, PERF-03.
**Gap Closure:** Closes audit flows for public recognizer E2E parity, public recognizer
benchmarking, no-drift compare gating, and recognizer-backed performance publication.
**Success Criteria**:
1. `tools/bench/whisper_emel_parity_runner.cpp` drives `emel::speech::recognizer::sm` through
   public recognizer events instead of constructing encoder/decoder actors or calling tokenizer
   decode directly.
2. `tools/bench/whisper_compare.py` and `tools/bench/whisper_benchmark.py` publish EMEL backend
   and runtime-surface metadata that identifies the public recognizer lane.
3. Recognizer-backed compare exact-matches `[C]` against the pinned `whisper.cpp` Phase 99
   model/audio pair and still hard-fails transcript drift.
4. Recognizer-backed single-thread CPU benchmark records EMEL faster than the matched
   `whisper.cpp` reference lane without using tool-only compute fallback or reference-owned EMEL
   lane state.

**Completion Evidence**:
- `tools/bench/whisper_emel_parity_runner.cpp` now initializes and recognizes through
  `emel::speech::recognizer::sm` with `speech/recognizer_routes/whisper::backend()`.
- Runner source grep found no direct `encoder::whisper`, `decoder::whisper`,
  `speech/encoder/whisper`, `speech/decoder/whisper`, or `decode_token_ids` references.
- Compare and benchmark summaries publish `backend_id=emel.speech.recognizer.whisper` and
  `runtime_surface=speech/recognizer+speech/recognizer_routes/whisper`.
- Recognizer-backed compare passed with `status=exact_match reason=ok`.
- Recognizer-backed single-thread benchmark passed with `benchmark_status=ok reason=ok`; the
  quality-gate artifact records EMEL mean `58,263,986 ns` versus reference mean `60,507,152 ns`.
- Changed-file scoped `scripts/quality_gates.sh` passed with
  `EMEL_QUALITY_GATES_BENCH_SUITE=whisper_compare:whisper_single_thread`; focused recognizer,
  Whisper recognizer, benchmark tool, domain-boundary, generic leak, and forbidden-root checks
  passed.

### Phase 125: Whisper Final Recognizer Closeout Rerun

**Goal:** Close v1.16 only after recognizer-backed source evidence satisfies the maintained E2E
runtime, parity, benchmark, and closeout contracts.
**Requirements:** CLOSE-01.
**Gap Closure:** Closes audit closeout blockers caused by the prior bypass-lane parity and
benchmark evidence.
**Success Criteria**:
1. Phases 123 and 124 have SUMMARY, VERIFICATION, and VALIDATION artifacts with executable
   source-backed evidence.
2. Full relevant quality gates pass with the recognizer-backed Whisper compare and single-thread
   benchmark suites.
3. `scripts/check_domain_boundaries.sh`, forbidden-root grep, focused recognizer tests,
   recognizer-backed compare, and recognizer-backed benchmark all pass.
4. `$gsd-audit-milestone` reports no source-backed maintained-path contradictions and
   REQUIREMENTS.md, ROADMAP.md, STATE.md, and phase artifacts agree that v1.16 is ready to
   archive.

**Completion Evidence**:
- The Phase 125 runtime evidence remains useful, but its original closeout claim was superseded
  by the Phase 126 explicit route graph repair and then by the Phase 127 decoder ownership gap.
- Full-scope `scripts/quality_gates.sh` passed with
  `EMEL_QUALITY_GATES_SCOPE=full` and
  `EMEL_QUALITY_GATES_BENCH_SUITE=whisper_compare:whisper_single_thread`; all 12 test shards,
  paritychecker, fuzz smoke, recognizer-backed compare, recognizer-backed benchmark, and docsgen
  completed.
- Latest full-gate compare summary records `exact_match`, `ok`, backend
  `emel.speech.recognizer.whisper`, runtime surface
  `speech/recognizer+speech/recognizer_routes/whisper`, transcript `[C]`.
- Latest full-gate benchmark summary records `ok`, `ok`, backend
  `emel.speech.recognizer.whisper`, runtime surface
  `speech/recognizer+speech/recognizer_routes/whisper`, transcript `[C]`, EMEL mean
  `59,106,792 ns`, and reference mean `59,958,847 ns`.
- The latest source-backed audit finds no remaining direct-runner bypass, domain-boundary leak,
  forbidden root, parity, benchmark, or explicit route graph contradiction, but it does find the
  decoder ownership blocker now assigned to Phase 127.

### Phase 126: Whisper Recognizer Explicit Route Graph Repair

**Goal:** Remove hidden recognizer backend dispatch so Whisper route support, readiness, encode,
decode, and detokenize behavior are selected and executed through explicit SML states, guards, and
transitions.
**Requirements:** CLOSE-01.
**Gap Closure:** Closes the 2026-04-28 audit blocker where `event::runtime_backend` function
pointers and `ctx.backend->...` calls hide route behavior outside the recognizer transition graph.
**Success Criteria**:
1. `src/emel/speech/recognizer/**` no longer defines, stores, binds, or invokes a runtime backend
   function-pointer table for route support, readiness, encode, decode, or detokenize behavior.
2. The maintained Whisper recognizer route remains model-family-free at the generic public
   recognizer boundary while route support/readiness choices are observable through explicit
   recognizer `sm.hpp` states/transitions and `guards.hpp` predicates.
3. The maintained compare and single-thread benchmark still drive `emel::speech::recognizer::sm`,
   exact-match `[C]`, and publish recognizer-backed parity/performance metadata after the refactor.
4. Rule-focused tests and `scripts/check_sml_behavior_selection.sh` fail before the repair and pass
   after the repair for the maintained recognizer path.
5. A final source-backed audit, domain-boundary check, recognizer tests, compare, benchmark, and
   full relevant quality gate pass before `CLOSE-01` is marked complete again.

**Completion Evidence:**
- `src/emel/speech/recognizer/**` no longer defines or stores `runtime_backend`,
  `initialize.backend`, or `ctx.backend` route calls.
- `src/emel/speech/recognizer/sm.hpp` routes support/readiness and encode/decode/detokenize
  phases through explicit route-policy guard/action transition rows.
- Generic recognizer leak grep for `whisper` returned no matches.
- `scripts/check_sml_behavior_selection.sh src/emel/speech/recognizer src/emel/speech/recognizer_routes/whisper src/emel/speech/encoder/whisper src/emel/speech/decoder/whisper src/emel/speech/tokenizer/whisper`
  passed.
- `EMEL_QUALITY_GATES_SCOPE=full EMEL_QUALITY_GATES_BENCH_SUITE='whisper_compare:whisper_single_thread' scripts/quality_gates.sh`
  passed on 2026-04-28 with 12/12 test shards, coverage line `90.8%`, branch `55.6%`,
  paritychecker, fuzz smoke, Whisper compare, Whisper benchmark, and docs generation.
- Latest benchmark evidence records EMEL mean `58,911,208 ns` versus reference mean
  `60,982,694 ns`.

### Phase 127: Whisper Decoder Ownership Gap Closure

**Goal:** Move maintained Whisper decoder sequence, logits, and timestamp-policy execution out of
`src/emel/speech/encoder/whisper/detail.hpp` so the public recognizer-to-decoder path is
source-backed by decoder-owned code or an explicitly appropriate shared kernel-owned surface.
**Requirements:** SPEECH-01, POLICY-01, CLOSE-01.
**Gap Closure:** Closes the 2026-04-28 source-backed audit blocker where
`src/emel/speech/decoder/whisper/actions.hpp` includes encoder detail code and calls
`emel::speech::encoder::whisper::detail::run_decoder_sequence`.
**Success Criteria**:
1. `src/emel/speech/decoder/whisper/actions.hpp` no longer includes
   `emel/speech/encoder/whisper/detail.hpp`, aliases `encoder::whisper::detail`, or calls decoder
   runtime helpers from the encoder namespace.
2. Decoder logits, timestamp-aware token selection, generation stopping, and decode-sequence
   execution live under decoder-owned implementation or a justified kernel-owned shared surface,
   without duplicating behavior or hiding route/runtime selection outside SML guards and
   transitions.
3. The public recognizer route still dispatches encoder, decoder, and detokenizer actors through
   public events; compare evidence exact-matches `[C]`; benchmark evidence keeps EMEL faster than
   the matched single-thread `whisper.cpp` reference.
4. Source-backed checks pass: `scripts/check_sml_behavior_selection.sh` over recognizer, Whisper
   route, encoder, decoder, and tokenizer paths; `scripts/check_domain_boundaries.sh`; forbidden
   root grep; and a focused grep proving decoder no longer depends on encoder detail.
5. A final source-backed audit and full relevant quality gate pass before `SPEECH-01`,
   `POLICY-01`, and `CLOSE-01` are marked complete again.

**Completion Evidence:**
- `src/emel/speech/decoder/whisper/actions.hpp` and
  `src/emel/speech/decoder/whisper/guards.hpp` no longer include or alias
  `emel/speech/encoder/whisper/detail.hpp`.
- `rg -n "encoder/whisper/detail|encoder::whisper::detail" src/emel/speech/decoder/whisper`
  returned no matches.
- `tests/speech/decoder/whisper/lifecycle_tests.cpp` includes a regression that checks
  production decoder files for encoder-detail dependency leaks and decoder-owned decode
  entrypoints.
- Focused decoder tests passed with 5 test cases and 1431 assertions; the focused recognizer
  test passed with 1 test case and 356 assertions.
- `scripts/check_sml_behavior_selection.sh` over recognizer, Whisper route, encoder, decoder,
  and tokenizer paths passed.
- `scripts/check_domain_boundaries.sh` passed, and forbidden model-family root grep over `src`,
  `tests`, and `CMakeLists.txt` returned no matches.
- Scoped quality gate passed on 2026-04-28 with changed-source coverage line `98.5%`, branch
  `58.2%`, exact Whisper compare transcript `[C]`, and Whisper single-thread benchmark status
  `ok`.
- Latest benchmark evidence records EMEL mean `58,537,483 ns` versus reference mean
  `60,435,595 ns`.

### Phase 128: Whisper Benchmark And Closeout Evidence Cleanup

**Status:** Complete.

**Goal:** Close the non-blocking milestone audit debt around closeout evidence stability and stale
historical closeout prose.
**Requirements:** Tech debt cleanup only; active requirements remain satisfied.
**Gap Closure:** Closes audit tech debt for the noisy default 3-iteration Whisper benchmark
wrapper and superseded Phase 122/125 "no blockers remain" wording.
**Success Criteria**:
1. The default Whisper single-thread benchmark closeout path is stable enough for milestone
   evidence, either by using the current 20-iteration sample by default for closeout or by making
   the closeout wrapper require an explicit stable iteration count.
2. Benchmark tests prove transcript, model, reference, and performance-regression contradictions
   still fail instead of being hidden by the more stable sampling policy.
3. Phase 122 and Phase 125 closeout artifacts are clearly marked superseded by Phases 126-127 and
   the latest tech-debt audit, without erasing their historical evidence.
4. The updated audit/roadmap/state ledger identifies Phase 127 as the active closeout truth and
   records Phase 128 as evidence-cleanup only.

**Completion Evidence:**
- `scripts/bench_whisper_single_thread.sh` and `tools/bench/whisper_benchmark.py` now default to
  20 measured iterations, with an explicit 20,000 ppm process-wall tolerance in benchmark
  summaries.
- `tools/bench/whisper_benchmark_tests.cpp` covers both default paths and preserves hard-fail
  regression coverage for transcript, model, reference, and material performance contradictions.
- Default closeout wrapper evidence passed with `benchmark_status=ok`, exact `[C]` transcripts,
  20 iterations, EMEL mean `60,189,787 ns`, and reference mean `60,736,881 ns`.
- Phase 122 and Phase 125 artifacts now include supersession notices pointing final closeout
  truth to the later source-backed chain.

### Phase 129: Whisper Detail Helper Deduplication Cleanup

**Status:** Complete.

**Goal:** Remove stale duplicate decoder/timestamp helper code from encoder detail without
reintroducing decoder production dependencies on encoder-owned implementation.
**Requirements:** Tech debt cleanup only; active requirements remain satisfied.
**Gap Closure:** Closes audit tech debt for unused duplicate decoder/timestamp helpers in
`src/emel/speech/encoder/whisper/detail.hpp`.
**Success Criteria**:
1. Duplicate decoder/timestamp helpers are removed from encoder detail or relocated to an
   explicitly appropriate shared surface, with no behavior-selection logic hidden in detail
   helpers.
2. Decoder production code still includes and aliases only decoder-owned detail or the approved
   shared surface, never `speech/encoder/whisper/detail.hpp`.
3. Regression tests cover the ownership boundary and the maintained recognizer-backed `[C]`
   compare path still passes.
4. Source-backed checks pass: `scripts/check_sml_behavior_selection.sh` over recognizer, Whisper
   route, encoder, decoder, and tokenizer paths; `scripts/check_domain_boundaries.sh`; forbidden
   root grep; and the decoder ownership grep.

**Completion Evidence:**
- `src/emel/speech/encoder/whisper/detail.hpp` no longer contains decoder token constants,
  `decode_policy_runtime`, decoder workspace sizing, decoder cross-cache/logit helpers,
  timestamp-aware token selection, or `run_decoder_sequence`.
- Decoder timestamp helper tests now live in `tests/speech/decoder/whisper/lifecycle_tests.cpp`;
  encoder detail tests include a source regression that blocks decoder helper names from
  returning to encoder detail.
- Focused encoder Whisper tests passed with 15 test cases and 2166 assertions; focused decoder
  Whisper tests passed with 7 test cases and 1436 assertions.
- `ctest` passed for `emel_tests_speech` and `emel_tests_whisper`.
- SML behavior-selection scan, domain-boundary script, forbidden-root grep, encoder-detail helper
  removal grep, and decoder production ownership grep passed.
- Scoped quality gate passed with encoder-detail coverage line `100.0%`, branch `50.0%`, exact
  Whisper compare transcript `[C]`, and Whisper single-thread benchmark status `ok`.
