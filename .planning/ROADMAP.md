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
    audit rerun found artifact, benchmark-publication, and rule-readiness gap phases were still
    required before archive.

## Current Milestone

**v1.16: ARM Whisper GGUF Parity And Performance**

Goal: Maintained Whisper tiny E2E path with no top-level Whisper runtime domain, speech-domain
recognizer/tokenizer ownership, exact transcript parity against the pinned `whisper.cpp` lane, and
a source-backed closeout audit.

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
  `tokenizer-tiny.json` contract before maintained dispatch, model Whisper ASR decode policy as an
  explicit speech-domain contract, and removed dispatch-time recognizer allocation; Phase 111 now
  owns the remaining SML rule-readiness gap.
- [x] **Phase 108: Pinned Whisper Artifact Parity Closeout** - Complete on 2026-04-27. Replaced
  the default bench-only normalized bridge with the user-approved Option B contract: source-owned
  legacy Whisper `lmgg` conversion in `src/emel/model/whisper`, exact transcript parity through
  the maintained EMEL speech recognizer lane, and exact parity evidence; Phase 112 now owns the
  final closeout rerun after audit blockers.
- [ ] **Phase 109: Reopened Whisper Artifact Evidence Closure** - Planned gap closure. Backfill
  phase-local verification and validation for the reopened artifact repair claims assigned to
  Phase 106, and reconcile the audit ledger so REOPEN-01 and SPEECH-01 are source-backed by the
  assigned closure phase.
- [ ] **Phase 110: Maintained Whisper Benchmark Publication Repair** - Planned gap closure.
  Connect the single-thread Whisper benchmark EMEL lane to the pinned Phase 99 source model path
  through the source-owned conversion path and make benchmark publication fail on model or
  transcript mismatches.
- [ ] **Phase 111: Speech Recognizer SML Rule Readiness Repair** - Planned gap closure. Move
  tokenizer readiness and Whisper execution-contract acceptance decisions out of detail-driven
  action/guard helper outputs and into explicit SML guards/transitions.
- [ ] **Phase 112: Reopened Whisper Closeout Rerun** - Planned gap closure. Rerun the full
  closeout gate and source-backed milestone audit after Phases 109-111 close the artifact,
  benchmark, and rule-readiness blockers.

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
2. Whisper runtime actors live under speech recognizer ownership while model and kernel ownership
   stay in `model/whisper` and `kernel/whisper`.
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
**Requirements:** PARITY-01; CLOSE-01 final closeout rerun now maps to Phase 112.
**Success Criteria**:
1. User approval for Option B is recorded as the final v1.16 closeout contract.
2. The EMEL lane consumes the pinned source model path through a source-owned legacy Whisper
   conversion path in `src/emel/model/whisper`.
3. Exact transcript parity is proven without a bench-only normalized-GGUF bridge being presented
   as direct pinned-artifact parity.
4. Full closeout quality gates and source-backed audit evidence from Phase 108 are superseded by
   Phase 112 because the 2026-04-27 audit rerun found remaining closeout blockers.

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

### Phase 110: Maintained Whisper Benchmark Publication Repair

**Goal:** Repair the maintained single-thread Whisper benchmark publication lane so EMEL and the
reference consume the same pinned Phase 99 source model contract and mismatches cannot publish as
`ok`.
**Requirements:** CLOSE-01.
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

### Phase 112: Reopened Whisper Closeout Rerun

**Goal:** Re-run closeout after Phases 109-111 repair artifact, benchmark, and rule-readiness
blockers.
**Requirements:** CLOSE-01.
**Gap Closure:** Closes final audit contradiction and milestone readiness gaps.
**Success Criteria**:
1. Full closeout quality gates pass with `EMEL_QUALITY_GATES_SCOPE=full` and
   `EMEL_QUALITY_GATES_BENCH_SUITE=whisper_compare`.
2. The maintained Whisper compare and single-thread benchmark summaries both record the pinned
   Phase 99 source model contract truthfully.
3. A source-backed milestone audit rerun reports no unsatisfied requirements, no integration
   blockers, and no missing Nyquist validation artifacts.
4. ROADMAP.md, REQUIREMENTS.md, STATE.md, and the milestone audit agree that v1.16 is ready to
   archive.
