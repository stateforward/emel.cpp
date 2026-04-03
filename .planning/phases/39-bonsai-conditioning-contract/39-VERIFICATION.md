---
phase: 39-bonsai-conditioning-contract
verified: 2026-04-02T23:40:00Z
status: passed
score: 3/3 phase truths verified
---

# Phase 39 Verification Report

**Phase Goal:** Make the maintained Bonsai slice resolve one explicit embedded-template request
contract and reject unsupported request shapes before runtime/parity claims.
**Verified:** 2026-04-02T23:40:00Z
**Status:** passed

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Loaded maintained Bonsai/Qwen models now resolve one explicit embedded-template formatter contract in `src` instead of relying on tool-only raw fallback logic. | ✓ VERIFIED | [format.hpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/src/emel/text/formatter/format.hpp) now owns embedded-template marker matching, runtime formatter implementations, and `resolve_model_binding(...)`, while [sm.hpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/src/emel/generator/sm.hpp) resolves the formatter binding once at generator construction. |
| 2 | The maintained Bonsai/Qwen request surface is explicitly limited to supported structured message roles with `add_generation_prompt=true` and `enable_thinking=false`, and unsupported shapes fail through explicit guards. | ✓ VERIFIED | [events.hpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/src/emel/text/conditioner/events.hpp), [context.hpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/src/emel/text/conditioner/context.hpp), [actions.hpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/src/emel/text/conditioner/actions.hpp), and [guards.hpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/src/emel/text/conditioner/guards.hpp) propagate explicit contract kind and reject unsupported request shapes before formatting/tokenization. |
| 3 | Maintained bench/parity model loading now publishes embedded chat-template truth into runtime model metadata so the runtime contract follows GGUF truth. | ✓ VERIFIED | [detail.hpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/src/emel/model/detail.hpp) now stores primary and named chat templates in `model_data.meta.tokenizer_data`, and both [generation_bench.cpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/tools/bench/generation_bench.cpp) and [parity_runner.cpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/tools/paritychecker/parity_runner.cpp) populate that metadata from GGUF loader storage. |

**Score:** 3/3 phase truths verified

## Automated Checks

- `./scripts/build_with_zig.sh`
- `./build/zig/emel_tests_bin --test-case='*formatter_*' --no-breaks`
- `./build/zig/emel_tests_bin --test-case='*conditioner_*' --no-breaks`
- `./build/zig/emel_tests_bin --test-case='*generator_qwen3_generator_initializes_and_generates_one_token*' --no-breaks`
- `cmake --build build/paritychecker_zig --parallel --target paritychecker paritychecker_tests`
- `./build/paritychecker_zig/paritychecker_tests --test-case='*formatter contract*' --no-breaks`
- `cmake --build build/bench_zig --parallel --target bench_runner bench_runner_tests`
- `./build/bench_zig/bench_runner_tests --test-case='*maintained Qwen and Liquid fixtures*' --no-breaks`
- `./scripts/paritychecker.sh`
- `./scripts/quality_gates.sh`

## Current Results

- Passed:
  - `./scripts/build_with_zig.sh`
  - `./build/zig/emel_tests_bin --test-case='*formatter_*' --no-breaks`
  - `./build/zig/emel_tests_bin --test-case='*conditioner_*' --no-breaks`
  - `./build/zig/emel_tests_bin --test-case='*generator_qwen3_generator_initializes_and_generates_one_token*' --no-breaks`
  - `cmake --build build/paritychecker_zig --parallel --target paritychecker paritychecker_tests`
  - `./build/paritychecker_zig/paritychecker_tests --test-case='*formatter contract*' --no-breaks`
  - `./scripts/quality_gates.sh`
  - `./scripts/paritychecker.sh`

## Verification Notes

- Focused formatter coverage was raised enough for the repo-wide gate to hit the required
  90.0% line coverage threshold.
- `scripts/paritychecker.sh` now runs green on macOS through a dedicated system-compiler parity
  build path, which avoids a reference-side loader crash seen in the pinned zig-built lane.
- The maintained Qwen generation fixture is present again in `tests/models/`, so maintained
  generation parity no longer fails on a missing baseline fixture path.
