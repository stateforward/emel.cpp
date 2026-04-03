---
phase: 38-fixture-provenance-and-metadata-truth
verified: 2026-04-02T18:14:12Z
status: passed
score: 3/3 phase truths verified
---

# Phase 38 Verification Report

**Phase Goal:** Freeze one truthful maintained Bonsai artifact identity and one executable GGUF
truth set before formatter or runtime work starts.
**Verified:** 2026-04-02T18:14:12Z
**Status:** passed

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Repo-visible docs and registries now freeze the maintained Bonsai artifact as `tests/models/Bonsai-1.7B.gguf` with verified source, download URL, size, and SHA256. | ✓ VERIFIED | [README.md](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/tests/models/README.md) records the maintained ledger entry, and [generation_fixture_registry.hpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/tools/generation_fixture_registry.hpp) carries the same identity in the maintained fixture registry. |
| 2 | Executable GGUF truth for the maintained Bonsai slice is now recorded explicitly as `qwen3` plus the required tokenizer and topology fields. | ✓ VERIFIED | [generation_fixture_registry.hpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/tools/generation_fixture_registry.hpp) records `general.architecture=qwen3`, `tokenizer.ggml.model=gpt2`, `tokenizer.ggml.pre=qwen2`, context length `32768`, block count `28`, embedding length `2048`, and head/KV counts `16/8`, while [fixture_registry_tests.cpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/tests/model/fixture_registry_tests.cpp) proves the registry fields and on-disk file size. |
| 3 | Maintained generation tooling stays honest by keeping Bonsai out of the generation-supported subset until later formatter/runtime phases. | ✓ VERIFIED | [parity_main.cpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/tools/paritychecker/parity_main.cpp), [parity_runner.cpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/tools/paritychecker/parity_runner.cpp), [paritychecker_tests.cpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/tools/paritychecker/paritychecker_tests.cpp), and [bench_runner_tests.cpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/tools/bench/bench_runner_tests.cpp) all consume the supported-generation subset instead of overclaiming Bonsai support. |

**Score:** 3/3 truths verified

## Automated Checks

- `./scripts/build_with_zig.sh`
- `./build/zig/emel_tests_bin --test-case='*fixture registry*' --no-breaks`
- `./build/zig/emel_tests_bin --test-case='*Bonsai model ledger*' --no-breaks`
- `cmake --build build/paritychecker_zig --parallel --target paritychecker paritychecker_tests`
- `./build/paritychecker_zig/paritychecker_tests --test-case='*maintained generation fixture*' --no-breaks`
- `cmake --build build/bench_zig --parallel --target bench_runner bench_runner_tests`
- `./build/bench_zig/bench_runner_tests --test-case='*maintained Qwen and Liquid fixtures*' --no-breaks`

## Verification Notes

- `scripts/quality_gates.sh` did not complete green. After the Phase 38-specific regression of
  Bonsai entering the generic tiny-model parity sweep was fixed in
  [paritychecker_tests.cpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/tools/paritychecker/paritychecker_tests.cpp),
  the remaining gate failure stayed in the broader `paritychecker_tests` suite:
  upstream reference loading rejects Bonsai GGUF type `41`, and unrelated generic tokenizer/kernel
  subprocess cases are aborting on existing non-Bonsai fixtures in this worktree.
- Bench verification required a local non-committed copy of `tests/models/Qwen3-0.6B-Q8_0.gguf`
  because that fixture is not tracked in this worktree. The phase change itself does not depend on
  committing that file.

