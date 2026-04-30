---
phase: 120
status: passed
verified: 2026-04-27
requirements:
  - TOK-02
  - POLICY-01
---

# Phase 120 Verification

## Verdict

Passed. The maintained Whisper decoder runtime now consumes a speech-owned ASR decode policy
during decoder dispatch, and the decoder no longer exposes or writes a hardcoded `token:<id>`
transcript surface. Maintained transcript publication remains tokenizer-owned.

## Evidence

| Check | Result |
|-------|--------|
| `cmake --build build/audit-native --target emel_tests_bin -j 6` | passed |
| `build/audit-native/emel_tests_bin --no-breaks --test-case='whisper_decoder*'` | passed, 4 test cases / 1419 assertions |
| `build/audit-native/emel_tests_bin --no-breaks --test-case='whisper detail*'` | passed, 9 test cases / 56 assertions |
| `build/audit-native/emel_tests_bin --no-breaks --test-case='speech whisper tokenizer*'` | passed, 4 test cases / 35 assertions |
| `cmake --build build/whisper_compare_tools --target whisper_emel_parity_runner whisper_benchmark_tests -j 6` | passed |
| `build/whisper_compare_tools/whisper_benchmark_tests` | passed, 10 test cases / 139 assertions |
| `scripts/check_domain_boundaries.sh` | passed |
| `rg -n "emel/whisper\|namespace emel::whisper\|kernel/whisper\|kernel::whisper" src tests CMakeLists.txt` | no matches |
| `rg -n "token:" src/emel/speech/encoder/whisper/detail.hpp src/emel/speech/decoder/whisper` | no matches |
| `scripts/bench_whisper_compare.sh --skip-reference-build --skip-emel-build` | passed, `status=exact_match reason=ok` |
| `scripts/bench_whisper_single_thread.sh --skip-reference-build --skip-emel-build` | passed, `benchmark_status=ok reason=ok` |
| Changed-file scoped `scripts/quality_gates.sh` with `EMEL_QUALITY_GATES_BENCH_SUITE=whisper_compare:whisper_single_thread` | passed |

## Source-Backed Policy Truth

The decoder public event now carries
`const emel::speech::tokenizer::whisper::asr_decode_policy &policy`. The decoder SML graph validates
that policy before selected execution, then `effect_run_decoder_variant` builds a small
`decode_policy_runtime` from policy token fields and passes it into `run_decoder_sequence`.

## Transcript Truth

The decoder event no longer contains `std::span<char> transcript`, `transcript_size_out`, or
`events::decode_done::transcript_size`. The maintained runner still creates final transcript text
through `whisper_tokenizer::decode_token_ids(...)` using generated decoder token IDs.
