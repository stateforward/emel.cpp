---
phase: 12-parity-and-verification-closure
verified: 2026-03-22T03:51:01Z
status: passed
score: 3/3 phase truths verified
---

# Phase 12 Verification Report

**Phase Goal:** The shipped flash-attention path exposed through `tools/paritychecker --generation`
truthfully proves flash execution, validates against a CMake-fetched latest upstream
`llama.cpp` reference, and remains parity-stable on both maintained bounded Llama-68M workloads.
**Verified:** 2026-03-22T03:51:01Z
**Status:** passed

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Paritychecker no longer decides parity against repo-local `tmp/llama.cpp`; it fetches the reference through CMake and publishes the fetched identity. | ✓ VERIFIED | [CMakeLists.txt](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/CMakeLists.txt) now declares `reference_impl` only through `FetchContent`, forces `REF_IMPL_REF` to `master`, and exports `PARITYCHECKER_REFERENCE_SOURCE` plus `PARITYCHECKER_REFERENCE_REF` into the runtime surface. Fresh and default paritychecker builds both configured with `ggml commit: 3306dbaef`, and the normal generation output reported `reference_impl: source=cmake_fetch ref=3306dbaef7553da03971c617e48cd27d00328bb4`. |
| 2 | The normal `paritychecker --generation` surface now proves flash execution and fetched-reference identity directly, and parity failures auto-emit diagnostics without requiring `--dump`. | ✓ VERIFIED | [parity_runner.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/parity_runner.cpp) prints `reference_impl`, `reference_decode_seams`, `kernel_dispatch`, and `flash_dispatch` on success, and routes init failure, flash-proof failure, reference failure, and mismatch paths through `dump_generation_failure_surface(...)`. [paritychecker_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/paritychecker_tests.cpp) asserts the presence of `reference_impl: source=` and `flash_dispatch: calls=` on subprocess-visible normal output. |
| 3 | Both maintained bounded workloads remain parity-stable against the fetched upstream reference on the canonical Llama-68M fixture. | ✓ VERIFIED | [paritychecker_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/paritychecker_tests.cpp) now covers the existing one-token request and a bounded longer decode at `max_tokens=8`. `ctest --test-dir build/paritychecker_zig_latest --output-on-failure -R paritychecker_tests` passed, and a direct `./build/paritychecker_zig_latest/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 8` run passed with `generated_tokens=8` and `flash_dispatch_calls=16`. |

## Requirements Coverage

| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| PAR-01 | ✓ SATISFIED | - |
| PAR-02 | ✓ SATISFIED | - |
| VER-01 | ✓ SATISFIED | - |

## Automated Checks

- `ctest --test-dir build/paritychecker_zig_latest --output-on-failure -R paritychecker_tests`
- `./build/paritychecker_zig_latest/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 8`
- `scripts/paritychecker.sh`
- `scripts/quality_gates.sh`
- `rg 'tmp/llama.cpp|reference_impl: source=|flash_dispatch:' tools/paritychecker/CMakeLists.txt tools/paritychecker/parity_runner.cpp`

## Verification Notes

- The default `build/paritychecker_zig` cache initially retained an older `REF_IMPL_REF`. Phase 12
  corrected that by forcing the cached reference selector to `master`, and the rerun verified the
  default build path now resolves to `ggml commit: 3306dbaef`.
- `scripts/quality_gates.sh` passed with the repo's existing warning-only benchmark policy. The
  warning surface still included
  `logits/validator_raw/vocab_32000`,
  `logits/validator_raw/vocab_256000`,
  `logits/validator_raw/vocab_128000`,
  `text/encoders/fallback_short`,
  and the unbaselined compare row
  `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_8`.
  The final gate output remained
  `warning: benchmark snapshot regression ignored by quality gates`.
- The same gate showed benchmark tooling is still configured separately from paritychecker and
  currently resolves its reference from
  [tmp/llama.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tmp/llama.cpp).
  That does not block Phase 12 because the phase scope is paritychecker-only, but it is a concrete
  benchmark-evidence risk for Phase 13 and should be addressed there.
