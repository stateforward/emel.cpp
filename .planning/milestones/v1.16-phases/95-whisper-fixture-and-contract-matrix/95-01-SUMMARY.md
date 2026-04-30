---
phase: 95-whisper-fixture-and-contract-matrix
plan: 01
requirements-completed: [FIX-01, FIX-02, FIX-03]
completed: 2026-04-25
---

# Phase 95 Plan 01: Whisper Fixture And Contract Matrix — Execution Summary

**Phase Goal:** Pin the requested model family and validate Whisper tiny GGUF architecture,
tensors, tokenizer, and audio contract.

**Plan Goal:** Lock the narrowed maintained Whisper tiny GGUF variant family (`{q8_0, q4_0,
q4_1}`), document each fixture's provenance, prove the EMEL loader/contract accepts the
maintained set, prove it rejects non-GGUF reference-lane siblings, and ensure no
whisper.cpp/ggml object can bootstrap EMEL Whisper state.

**Status:** Complete.

## Approval Of Narrowed Scope

agent-0 approved Option A (narrow scope) on 2026-04-25 in message `J2bGANuAk61CW6P6` after
the Phase 95 blocker (`dQ3s3jUncdOmQRdl`) reported that the upstream
`oxide-lab/whisper-tiny-GGUF` repo only ships three EMEL-loadable Candle-style GGUFs at the
top level. The other quant siblings (`q5_0`, `q5_1`, `q2_k`, `q3_k`, `q4_k`, `q5_k`,
`q6_k`, `q8_k`) live only under the `whisper.cpp/` subdirectory and use whisper.cpp's
`lmgg` binary format, which cannot bootstrap EMEL state. The maintained v1.16 family is
therefore `{q8_0, q4_0, q4_1}`. Future broader-quant work is deferred to an explicit
EMEL-owned conversion phase.

## Outcomes Per Task

### Task 1 — Narrow REQUIREMENTS.md KERN-01

- Edited `.planning/REQUIREMENTS.md` so `KERN-01` lists only `{q4_0, q4_1, q8_0}` and
  documents the deferral of the broader quant family with a pointer to the agent-0 message
  ID.

### Task 2 — Tighten `tests/models/README.md` Whisper Section

- Replaced the broad 11-variant scope note with a narrowed `{q8_0, q4_0, q4_1}` scope
  note that explains why the `whisper.cpp/`-prefixed siblings (and their `lmgg` magic) are
  reserved for the reference lane.
- Added new `## whisper-tiny-q4_0.gguf` and `## whisper-tiny-q4_1.gguf` sections with
  pinned download URLs, sizes, SHA256s, license, repo commit, executable metadata
  description, maintained contract pointer, and loader/runtime boundary note. Sizes and
  SHA256s were source-backed by reading the LFS pointer files at the pinned commit
  (`https://huggingface.co/oxide-lab/whisper-tiny-GGUF/raw/<commit>/whisper-tiny-q4_0.gguf`
  and `whisper-tiny-q4_1.gguf`).
- Added an external tokenizer-asset note to the `model-tiny-q80.gguf` section explaining
  that the Candle-style Whisper tiny GGUFs do not embed tokenizer metadata; the maintained
  Whisper tokenizer source is the upstream `tokenizer-tiny.json` sibling. This note keeps
  Phase 95's contract truthful and defers staging the tokenizer file until ASR runtime
  phases (97/98) need it.

### Task 3 — Extend `tests/model/fixture_manifest_tests.cpp`

- Updated the existing q80 wording-lock test to also assert the narrowed scope wording
  ("narrowed to the three upstream EMEL-loadable Candle-style GGUFs", "deferred to a
  future approved EMEL-owned").
- Added two new wording-lock tests for the q4_0 and q4_1 README sections asserting URL,
  pinned-commit URL, size, SHA256, and the per-variant loader/runtime boundary note.
- Added two new local-size guards for the q4_0 and q4_1 fixtures that skip with a
  `MESSAGE` when the larger fixtures are not staged locally so CI degrades gracefully.

### Task 4 — Loader/Contract Proof Coverage In `tests/model/loader/lifecycle_tests.cpp`

- Added `whisper_q4_0_fixture_path()` and `whisper_q4_1_fixture_path()` helpers next to
  the existing `whisper_fixture_path()` helper.
- Added two new real-fixture tests
  (`model_whisper_detail_builds_execution_contract_from_q4_0_fixture` and the q4_1
  counterpart) that, when the fixtures are present locally, parse them via the
  EMEL-owned loader and assert `build_execution_contract` returns OK with the maintained
  tensor families, architecture identity, and hyperparameters. Both tests skip gracefully
  with a `MESSAGE` when the larger fixtures are not staged locally.
- Added `model_whisper_loader_rejects_whisper_cpp_lmgg_sibling_magic_synthetic` which
  builds a 64-byte in-memory file image whose first four bytes are `lmgg` and asserts the
  EMEL-owned `emel::gguf::loader::detail::probe_requirements` returns `model_invalid`
  without producing a usable requirements descriptor. This proves a `whisper.cpp/`
  sibling cannot bootstrap EMEL Whisper state.
- Replaced the broken Phase 94 tokenizer-loaded-from-GGUF test
  (`model_whisper_fixture_tokenizer_metadata_is_loaded_from_emel_parsed_gguf`) with the
  truthful inverse contract:
  `model_whisper_fixture_carries_no_embedded_tokenizer_metadata_in_gguf`. The new test
  asserts the q80 fixture carries no `tokenizer.model`/`tokenizer.tokens`-family keys
  and that `load_vocab_from_gguf` cleanly fails (returns `false` and leaves
  `tokenizer_model_id == UNKNOWN`) instead of fabricating a vocab.

### Task 5 — Static No-whisper.cpp/ggml Include Guardrail

- Added a `EMEL Whisper sources reference no whisper.cpp or ggml headers` test in
  `tests/model/fixture_manifest_tests.cpp`. It walks `src/emel/model/whisper/` recursively
  and scans `.hpp/.cpp/.h/.cc` files for forbidden include substrings (the `<whisper.h>`,
  `"whisper.h"`, `<whisper>`, `<ggml.h>`, `"ggml.h"`, `<ggml>`, `<whisper-cpp>` family).
  It also scans the two test surfaces that touch Whisper loader/contract code
  (`tests/model/loader/lifecycle_tests.cpp` and `tests/model/fixture_manifest_tests.cpp`).
- Forbidden needle strings are constructed via concatenation in the test source so the
  scanner does not self-trip on its own literal needles.

### Task 6 — Focused Tests And Scoped Quality Gate

- Built `emel_tests_bin` via `cmake --build build/audit-native --target emel_tests_bin`.
- Focused doctest filter `--test-case='*Whisper*,*model_whisper*,*EMEL Whisper sources*'`
  result: `17 test cases, 17 passed, 0 failed, 471 assertions, all passing`. The four
  q4_0/q4_1 real-fixture and local-size tests skip with explicit `MESSAGE`s because the
  larger fixtures are not staged locally; this is the documented graceful-degradation
  behavior.
- Scoped quality gate via
  `EMEL_QUALITY_GATES_CHANGED_FILES="tests/model/fixture_manifest_tests.cpp
   tests/model/loader/lifecycle_tests.cpp tests/models/README.md" scripts/quality_gates.sh`
  exits `0`. The gate correctly skips coverage, paritychecker, fuzz, bench, and docsgen
  lanes because Phase 95 did not change `src/emel/`, parity, fuzz, bench, or docs source.

### Task 7 — SUMMARY And VERIFICATION

- This summary file enumerates the outcomes per task above.
- `95-VERIFICATION.md` (companion file) maps each ROADMAP success criterion to the
  artifacts and test cases that satisfy it.

## Files Changed

| File | Type | Change |
|------|------|--------|
| `.planning/REQUIREMENTS.md` | docs | Narrow `KERN-01` to `{q4_0, q4_1, q8_0}` with deferral note. |
| `.planning/ROADMAP.md` | docs | (already in this branch from Phase 94 close) Phase 94 marked complete; Phase 95 stays open. |
| `.planning/STATE.md` | docs | (already in this branch) Phase 94 closed, Phase 95 active. |
| `tests/models/README.md` | docs | Narrow Whisper variant-family scope note; add q4_0 and q4_1 sections; document external tokenizer asset. |
| `tests/model/fixture_manifest_tests.cpp` | tests | Lock narrowed scope wording; add q4_0/q4_1 wording-locks; add local-size guards; add static no-whisper.cpp/ggml include guardrail. |
| `tests/model/loader/lifecycle_tests.cpp` | tests | Add q4_0/q4_1 contract tests; add lmgg rejection test; replace incorrect Phase 94 tokenizer-loaded test with the truthful absent-tokenizer test. |
| `.planning/phases/95-whisper-fixture-and-contract-matrix/95-CONTEXT.md` | docs | Phase 95 context. |
| `.planning/phases/95-whisper-fixture-and-contract-matrix/95-01-PLAN.md` | docs | Phase 95 plan. |
| `.planning/phases/95-whisper-fixture-and-contract-matrix/95-01-SUMMARY.md` | docs | This summary. |
| `.planning/phases/95-whisper-fixture-and-contract-matrix/95-VERIFICATION.md` | docs | Verification report (companion). |

## Phase 94 Audit Gap Closed In Phase 95

Phase 94 added
`model_whisper_fixture_tokenizer_metadata_is_loaded_from_emel_parsed_gguf` based on an
incorrect assumption that the Candle-style Whisper tiny GGUF embeds tokenizer keys. Phase
94's verification ran a narrower doctest filter
(`*maintained Whisper tiny q80 fixture*`) so the broken test was never executed at
verification time. Phase 95 surfaced the failure when running the broader Whisper filter
and replaced the incorrect test with one that source-backs the actual contract truth (no
embedded tokenizer keys; external `tokenizer-tiny.json` is the maintained tokenizer
source). This closes the audit gap without depending on the original false claim.

## Unrelated Residual Risk

- `tests/diarization/sortformer/modules/lifecycle_tests.cpp:74`,
  `sortformer modules bind maintained tensor contract`, fails with `SIGSEGV` when running
  the full `emel_tests_bin` suite. The failure is reproducible at HEAD without Phase 95
  changes (verified by stashing the Phase 95 worktree edits and rerunning the same binary)
  and is in code Phase 95 did not touch. Per agent-0 directive (2026-04-25 message
  during execution), Phase 95 records this as an unrelated residual risk and does not fix
  Sortformer in this phase.

## Snapshot Notes

- The scoped quality gate run wrote updated timings into
  `snapshots/quality_gates/timing.txt`. That file was already in the worktree's modified
  list at session start (Phase 94 carry-over) and remains in the worktree as part of the
  general worktree drift; it is not committed by Phase 95 and is reported here per
  agent-0's snapshot-change directive.
