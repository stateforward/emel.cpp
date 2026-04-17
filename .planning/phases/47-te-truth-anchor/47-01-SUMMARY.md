---
phase: 47-te-truth-anchor
plan: 01
subsystem: embeddings
tags: [te75m, fixture, proof-corpus, metadata, truth-anchor]
requires:
  - phase: 46.1-rename-planner-family-wrapper-names-to-remove-ambiguity-and-
    provides: clean planner closeout before v1.11 begins
provides:
  - maintained TE fixture lock at `tests/models/TE-75M-q8_0.gguf`
  - deterministic TE proof corpus manifest with pairwise text-image and text-audio anchors
  - regression coverage for fixture metadata and proof-corpus drift
affects: [v1.11 roadmap, v1.11 requirements, v1.11 state]
tech-stack:
  added: []
  patterns:
    - maintained fixture truth anchor
    - in-memory proof corpus manifest
    - repo-visible metadata regression coverage
key-files:
  created:
    - .planning/phases/47-te-truth-anchor/47-CONTEXT.md
    - .planning/phases/47-te-truth-anchor/47-01-PLAN.md
    - .planning/phases/47-te-truth-anchor/47-01-SUMMARY.md
    - .planning/phases/47-te-truth-anchor/47-VERIFICATION.md
    - tests/embeddings/fixtures/te75m/README.md
    - tests/embeddings/fixtures/te75m/red-square.txt
    - tests/embeddings/fixtures/te75m/pure-tone-440hz.txt
    - tests/model/fixture_manifest_tests.cpp
  modified:
    - .planning/PROJECT.md
    - .planning/REQUIREMENTS.md
    - .planning/ROADMAP.md
    - .planning/STATE.md
    - CMakeLists.txt
    - tests/models/README.md
  local_only:
    - tests/models/TE-75M-q8_0.gguf
key-decisions:
  - "Treat the current Hugging Face GGUF/model metadata as the maintained TE truth source for Phase 47, including `gguf.architecture=omniembed` and the upstream LEAF-IR/MobileNetV4-Medium/EfficientAT component families."
  - "Define the proof corpus as deterministic in-memory payload contracts with pairwise text-image and text-audio anchors instead of widening into generic image/audio file decode."
  - "Keep sibling quantization truth explicit by naming `TE-75M-q5_0.gguf` as an upstream sibling while still locking `TE-75M-q8_0.gguf` as the only maintained TE fixture."
patterns-established:
  - "Embedding milestones can lock proof inputs early by documenting in-memory payload contracts before runtime implementation exists."
  - "Maintained-model README entries should carry exact local checksum and size from the pinned local fixture, while scope notes explicitly reject sibling-file support claims."
requirements-completed:
  - FIX-01
  - FIX-02
completed: 2026-04-13
commit: pending
---

# Phase 47: TE Truth Anchor Summary

**Phase 47 locked the maintained TE truth surface before runtime work starts: the repo now names
one exact local `TE-75M-q8_0.gguf` fixture with upstream provenance and checksum, defines one tiny
deterministic TE proof corpus as in-memory payload contracts, and has a model-shard regression test
that keeps those anchors from drifting.**

## Performance

- **Completed:** 2026-04-13T23:39:06-05:00
- **Tasks:** 3
- **Files modified:** 14 repo-visible files plus the local TE fixture download

## Accomplishments

- Downloaded `tests/models/TE-75M-q8_0.gguf`, computed the local SHA256
  `955b5c847cc95c94ff14a27667d9aca039983448fd8cefe4f2804d3bfae621ae`, and recorded its source,
  size, stable path, and maintained-scope note in `tests/models/README.md`.
- Locked the current upstream truth anchor for the TE slice, including
  `gguf.architecture=omniembed`, 1280-dim shared embeddings with Matryoshka truncation, and the
  current upstream component families `LEAF-IR`, `MobileNetV4-Medium`, and `EfficientAT mn20_as`.
- Defined the canonical proof corpus under `tests/embeddings/fixtures/te75m/` as two narrow
  pairwise anchors:
  `red-square` for text-image smoke and `pure-tone-440hz` for text-audio smoke.
- Added `tests/model/fixture_manifest_tests.cpp` and wired it into `CMakeLists.txt` so the repo
  now has regression coverage for fixture metadata and proof-corpus drift.

## Decisions Made

- Follow current upstream GGUF/model metadata as the maintained Phase 47 truth source rather than
  earlier assumptions about TE internals.
- Keep the proof corpus expressed as in-memory payload contracts because the first TE slice does
  not yet claim generic image/audio file decoding.
- Use pairwise anchors instead of a forced synthetic three-way semantic triplet to keep the proof
  inputs honest and minimal.

## Validation

- `shasum -a 256 tests/models/TE-75M-q8_0.gguf`
- `cmake --build build/zig --target emel_tests_bin -j4`
- `./build/zig/emel_tests_bin --no-breaks --source-file='*tests/model/*'`
- `scripts/quality_gates.sh`

## Residual Notes

- `scripts/quality_gates.sh` reached the paritychecker reference build and failed in the fetched
  `reference_impl` tree on `common/jinja/value.cpp` because
  `common_parse_utf8_codepoint` / `utf8_parse_result` were undeclared. That failure is outside the
  files changed for Phase 47, but it remains a repo-wide gate blocker to resolve separately.
