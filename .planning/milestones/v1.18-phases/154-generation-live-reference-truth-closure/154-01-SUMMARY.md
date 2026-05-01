# Phase 154-01 Summary: Generation Live Reference Truth Closure

## Outcome

Phase 154 closed the generation reference-truth gap by moving maintained generation comparison from
stored baseline truth to a live reference-lane generation result.

## Changes

- `tools/paritychecker/parity_engines.cpp` now calls
  `run_reference_generate(state.reference, opts, reference_result)` on the normal generation path.
- EMEL generation output is compared against `reference_result` before any baseline file is loaded.
- Stored generation baselines are now secondary publication artifacts:
  - `--write-generation-baseline` writes the live reference result after EMEL/live parity passes.
  - Normal runs compare the loaded baseline against the live reference result to detect stale
    publication artifacts.
- Baseline output is now printed as `generation_baseline:` instead of pretending the snapshot is a
  `reference_impl`.
- Baseline loads now mark trace data unavailable when old append-only baselines do not carry a full
  token trace.
- `tools/paritychecker/paritychecker_tests.cpp` now guards:
  - live reference invocation before baseline load,
  - no `baseline_record.result` alias as reference truth,
  - current maintained LFM2 live-reference success,
  - legacy Qwen live-reference drift reporting,
  - output schema separation between `reference_impl:` and `generation_baseline:`.

## Important Finding

The live reference comparison exposed that the legacy non-current Qwen fixture does not match the
live llama.cpp reference output for `prompt=hello max_tokens=1`. The paritychecker now reports that
as `generation parity mismatch` instead of returning success from the stored EMEL baseline.

The current publication fixture, `LFM2.5-1.2B-Thinking-Q4_K_M.gguf`, passes exact live-reference
generation parity and still validates against its append-only baseline artifact.

## Verification

Commands passed:

```sh
build/paritychecker_zig/paritychecker_tests \
  --test-case="parity engine source keeps EMEL and reference lane objects separate"

build/paritychecker_zig/paritychecker_tests \
  --test-case="paritychecker matches current maintained generation publication against live reference"

build/paritychecker_zig/paritychecker_tests \
  --test-case="paritychecker reports legacy maintained generation live-reference drift"

ctest --test-dir build/paritychecker_zig --output-on-failure
```

Direct live-reference smoke passed for current publication:

```sh
build/paritychecker_zig/paritychecker --generation \
  --model tests/models/LFM2.5-1.2B-Thinking-Q4_K_M.gguf \
  --text hello --max-tokens 1 --dump
```

It reported `generation parity ok`, `generation_baseline:`, `reference_impl: source=cmake_fetch`,
and `reference_decode_calls=2`.

