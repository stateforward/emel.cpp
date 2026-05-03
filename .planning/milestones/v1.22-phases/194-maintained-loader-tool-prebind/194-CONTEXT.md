---
phase: 194
slug: maintained-loader-tool-prebind
status: planned
source_audit: .planning/milestones/v1.22-MILESTONE-AUDIT.md
requirements:
  - TENSOR-02
  - LOAD-02
---

# Phase 194 Context

## Audit Gap

The source-backed v1.22 audit found that maintained benchmark, parity, and probe lanes still resize
GGUF KV arena and entry storage while inside `model_loader.process_event(...)`.

Affected lanes:

- `tools/bench/generation_bench.cpp`
- `tools/bench/diarization/sortformer_fixture.hpp`
- `tools/embedded_size/emel_probe/main.cpp`
- `tools/paritychecker/parity_engines.cpp`

## Closure Intent

Prebind or preallocate the required GGUF KV arena and entry storage before loader actor dispatch so
the maintained model-loading path satisfies the no-allocation-during-dispatch rule.

The user explicitly approved updates to snapshots, benchmarks, and models for this gap-closure
work.

## Required Validation

- Focused source-backed check proving the four maintained lanes do not grow KV storage during
  `model_loader.process_event(...)`.
- `ctest --test-dir build/zig --output-on-failure -R emel_tests_model_and_batch`
- `ctest --test-dir build/paritychecker_zig --output-on-failure -R paritychecker_tests`
- Scoped quality gate with changed files and the relevant benchmark suites.
