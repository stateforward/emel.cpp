# Phase 154 Verification

| Check | Result | Evidence |
|-------|--------|----------|
| Live reference is normal truth | Passed | `run_generation_harness_contract(...)` calls `run_reference_generate(state.reference, opts, reference_result)` before loading `baseline_record`. |
| Baseline is not reference truth | Passed | Source guard rejects `generation_result & reference_result = baseline_record.result`; output now uses `generation_baseline:` for snapshot metadata. |
| Current publication live parity | Passed | LFM2 generation CLI returned `generation parity ok` with `reference_decode_calls=2`. |
| Legacy drift is reported | Passed | Non-current Qwen fixture returns `generation parity mismatch` instead of success from stored baseline. |
| Lane isolation | Passed | Source checks still require separate EMEL model data and reference backend ownership. |

## Commands

```sh
cmake --build build/paritychecker_zig --target paritychecker_tests -j2
cmake --build build/paritychecker_zig --target paritychecker -j2
build/paritychecker_zig/paritychecker_tests --test-case="parity engine source keeps EMEL and reference lane objects separate"
build/paritychecker_zig/paritychecker_tests --test-case="paritychecker matches current maintained generation publication against live reference"
build/paritychecker_zig/paritychecker_tests --test-case="paritychecker reports legacy maintained generation live-reference drift"
ctest --test-dir build/paritychecker_zig --output-on-failure
```

## Notes

No snapshot refresh was required for this phase. The existing LFM2 baseline lacks full trace data,
so baseline loading now correctly marks its trace as unavailable for trace comparison while still
checking generated token count and output bytes.

