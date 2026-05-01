---
phase: 118
status: passed
verified: 2026-04-27
requirements:
  - SPEECH-01
  - TOK-02
  - POLICY-01
  - PARITY-01
  - PERF-03
---

# Phase 118 Verification

## Verdict

Passed. The maintained Whisper EMEL parity runner no longer directly reaches Whisper actor detail
headers, the decode policy is narrowed to source-backed timestamp-token behavior, exact `[C]`
parity is preserved, and the single-thread benchmark still reports EMEL faster than the matched
`whisper.cpp` lane.

## Evidence

| Check | Result |
|-------|--------|
| `cmake --build build/whisper_compare_tools --target whisper_emel_parity_runner whisper_benchmark_tests -j 4` | passed |
| `build/whisper_compare_tools/whisper_benchmark_tests` | passed, 10 test cases / 139 assertions |
| `cmake --build build/audit-native --target emel_tests_bin -j 4` | passed |
| `ctest --test-dir build/audit-native -R 'emel_tests_(speech\|whisper)' --output-on-failure` | passed, 2/2 |
| `rg 'model/whisper/detail.hpp\|speech/(encoder\|decoder\|tokenizer)/whisper/detail.hpp\|model::whisper::detail\|speech::(encoder\|decoder\|tokenizer)::whisper::detail\|whisper_detail\|decoder_detail' tools/bench/whisper_emel_parity_runner.cpp` | no matches |
| `scripts/check_domain_boundaries.sh` | passed |
| `scripts/bench_whisper_compare.sh --skip-reference-build --skip-emel-build` | passed, `status=exact_match reason=ok` |
| `EMEL_WHISPER_BENCH_WARMUPS=0 EMEL_WHISPER_BENCH_ITERATIONS=1 scripts/bench_whisper_single_thread.sh --skip-reference-build --skip-emel-build` | passed, `benchmark_status=ok reason=ok` |
| `ctest --test-dir build/audit-native -R 'lint_snapshot' --output-on-failure` | passed after approved snapshot update |
| Changed-file scoped `scripts/quality_gates.sh` with `EMEL_QUALITY_GATES_BENCH_SUITE=whisper_compare` | passed |

## Source-Backed Policy Truth

The earlier no-timestamps policy claim was not source-backed for the maintained `[C]` lane:
forcing a no-timestamps prompt produced transcript drift. Phase 118 narrows the maintained policy
to the actual timestamp-token mode. The EMEL compare record now includes:

- `decode_policy_language: english`
- `decode_policy_task: transcribe`
- `decode_policy_timestamp_mode: timestamp_tokens`
- `decode_policy_suppress_translate: true`
- `decode_policy_prompt_token_count: 3`

## Benchmark Evidence

The single-iteration benchmark summary recorded:

- EMEL transcript: `[C]`
- Reference transcript: `[C]`
- EMEL mean process wall time: `62020084 ns`
- Reference mean process wall time: `66998708 ns`

## Remaining Closeout Work

`CLOSE-01` remains pending for Phase 119, which must rerun the final source-backed milestone audit
and resolve the Phase 113 validation-ledger truth.
