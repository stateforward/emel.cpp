---
phase: 38
slug: fixture-provenance-and-metadata-truth
created: 2026-04-02
status: ready
---

# Phase 38 Context

## Phase Boundary

Freeze one truthful maintained Bonsai fixture identity before formatter or runtime work starts by
adding the real `tests/models/Bonsai-1.7B.gguf` artifact, recording executable GGUF truth, and
keeping generation-capable tooling honest about Bonsai still being pre-runtime.

## Implementation Decisions

### Fixture Identity
- The maintained Bonsai file lands at `tests/models/Bonsai-1.7B.gguf` with the verified size,
  SHA256, source repo, and direct download URL recorded in repo-visible docs and registries.
- The repo keeps the exact filename `Bonsai-1.7B.gguf`; `Q1_0_g128` is recorded as the internal
  weight format, not promoted into the fixture filename.
- Phase 38 carries the real GGUF artifact in `tests/models/` so later phases can build on the same
  frozen file identity.

### Truth Publication
- Executable Bonsai metadata truth is recorded in the maintained fixture registry instead of being
  inferred later from stale prose.
- The required Phase 38 truth set is `general.architecture=qwen3`,
  `tokenizer.ggml.model=gpt2`, `tokenizer.ggml.pre=qwen2`, `qwen3.context_length=32768`,
  `qwen3.block_count=28`, `qwen3.embedding_length=2048`,
  `qwen3.attention.head_count/head_count_kv=16/8`.
- Operator-facing docs must explicitly call out the stale quickstart filename
  `Bonsai-1.7B-Q1_0_g128.gguf` as incorrect for the maintained repo path.

### Tooling Honesty
- The existing generation tooling keeps a supported-fixture subset separate from the full
  maintained-fixture registry so parity and bench do not overclaim Bonsai support before Phase 39
  and Phase 40.
- No Phase 38 change should require a successful Bonsai generation path yet.
- No state-machine structure or `src/` runtime topology changes are in scope for this phase.

## Existing Code Insights

### Reusable Assets
- `tools/generation_fixture_registry.hpp` is already the central maintained-fixture seam consumed
  by paritychecker and bench tooling.
- `tests/models/README.md` is the repo-visible provenance ledger for maintained GGUF artifacts.
- Existing doctest coverage under `tests/model/` is the lightest place to pin registry truth
  without widening tool behavior prematurely.

### Established Patterns
- Maintained fixture identity is carried by small constexpr tool registries and mirrored in focused
  subprocess tests.
- Tooling publishes honest pre-runtime states instead of pretending a fixture already works.
- Snapshot-backed parity and bench coverage should only iterate generation-supported fixtures.

### Integration Points
- `tools/paritychecker/parity_main.cpp` and `tools/paritychecker/parity_runner.cpp` consume the
  supported generation fixture set for help text and maintained fixture-path validation.
- `tools/paritychecker/paritychecker_tests.cpp` and `tools/bench/bench_runner_tests.cpp` lock the
  supported generation subset and prevent silent drift.
- `tests/model/fixture_registry_tests.cpp` will pin Bonsai provenance, file identity, and the
  filename-versus-format distinction.

## Specific Ideas

- Add the verified `/tmp/Bonsai-1.7B.gguf` artifact to `tests/models/`.
- Extend the fixture registry with provenance and executable-truth fields for Bonsai.
- Add a registry split between all maintained fixtures and generation-supported fixtures.
