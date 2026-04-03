---
phase: 33-fixture-metadata-and-contract-lock
plan: 01
subsystem: planning
tags: [liquid, fixture, metadata, contract]
requires: []
provides:
  - official maintained Liquid fixture provenance under tests/models
  - executable metadata truth anchored on lfm2 and 128000 context
  - explicit maintained structured-chat formatter contract for the Liquid slice
affects: [34, 35, 36, 37, 38]
tech-stack:
  added: []
  patterns:
    - maintained fixture truth anchored in repo evidence
    - maintained chat-template contract published as auditable operator output
key-files:
  created:
    - .planning/phases/33-fixture-metadata-and-contract-lock/33-01-SUMMARY.md
  modified:
    - tests/models/README.md
    - tools/generation_formatter_contract.hpp
    - tools/generation_fixture_registry.hpp
    - tools/paritychecker/parity_runner.cpp
    - tools/bench/generation_bench.cpp
key-decisions:
  - "Keep v1.9 fixed to one maintained Liquid fixture at tests/models/LFM2.5-1.2B-Thinking-Q4_K_M.gguf."
  - "Use GGUF/config metadata truth, including lfm2 architecture and 128000 context, instead of stale prose."
  - "Publish one explicit structured chat-message contract with roles=system,user and tools=none."
patterns-established:
  - "Maintained fixture and formatter truth belongs on auditable repo surfaces before runtime widening."
  - "The maintained Liquid request path rejects unsupported template/tool/history shapes rather than silently falling back."
requirements-completed: [FIX-02, META-01, COND-03]
duration: reconstructed
completed: 2026-04-02
---

# Phase 33 Plan 01: Fixture, Metadata, And Contract Lock Summary

The maintained Liquid fixture, metadata truth, and conditioning contract were landed in the repo
but never summarized. This summary reconstructs that closeout from the delivered evidence.

## Accomplishments

- Documented the official maintained Liquid fixture at
  `tests/models/LFM2.5-1.2B-Thinking-Q4_K_M.gguf` with checksum, stable path, source, and
  download URL.
- Bound maintained metadata truth to executable GGUF/config facts, including
  `general.architecture=lfm2` and context length `128000`.
- Published the maintained Liquid formatter contract as structured chat messages with
  `roles=system,user`, `tools=none`, `add_generation_prompt=true`, and
  `keep_past_thinking=false`.
- Extended maintained generation fixtures additively so Liquid did not replace the prior Qwen
  maintained anchor.

## Evidence

- [README.md](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/tests/models/README.md)
- [generation_formatter_contract.hpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/tools/generation_formatter_contract.hpp)
- [generation_fixture_registry.hpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/tools/generation_fixture_registry.hpp)
- [parity_runner.cpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/tools/paritychecker/parity_runner.cpp)
- [generation_bench.cpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/tools/bench/generation_bench.cpp)

---
*Phase: 33-fixture-metadata-and-contract-lock*
*Completed: 2026-04-02*
