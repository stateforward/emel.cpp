---
phase: 122
status: passed
verified: 2026-04-27
requirements:
  - CLOSE-01
---

# Phase 122 Verification

## Verdict

Passed for its historical scope. The repaired maintained Whisper source path,
preserved-baseline validation ledger, default warmed benchmark, and full closeout quality gate
passed on 2026-04-27. This verification is superseded for final v1.16 closeout by the
recognizer-backed Phase 127 audit truth and the Phase 128 evidence cleanup.

## Evidence

| Check | Result |
|-------|--------|
| `scripts/check_domain_boundaries.sh` | passed |
| `rg -n "emel/whisper|namespace emel::whisper|kernel/whisper|kernel::whisper" src tests CMakeLists.txt` | no matches |
| `scripts/bench_whisper_compare.sh --skip-reference-build --skip-emel-build` | `status=exact_match reason=ok` |
| `scripts/bench_whisper_single_thread.sh --skip-reference-build --skip-emel-build` | `benchmark_status=ok reason=ok` |
| `EMEL_QUALITY_GATES_SCOPE=full EMEL_QUALITY_GATES_BENCH_SUITE='whisper_compare:whisper_single_thread' scripts/quality_gates.sh` | passed |
| Full-gate coverage | line `90.8%`, branch `55.6%` |
| Full-gate coverage shards | 12/12 passed, 100% tests passed |
| Full-gate paritychecker | `paritychecker_tests` passed |
| Full-gate fuzz lanes | `gguf_parser`, `gbnf_parser`, `jinja_parser`, and `jinja_formatter` completed |
| Full-gate Whisper compare lane | `status=exact_match reason=ok` |
| Full-gate Whisper benchmark lane | `benchmark_status=ok reason=ok` |
| Full-gate docs lane | docsgen configured and built |

## Maintained Compare Summary

`build/whisper_compare/summary.json` records:

- compare group: `whisper/tiny/q8_0/phase99_440hz_16khz_mono`
- comparison status: `exact_match`
- reason: `ok`
- EMEL transcript: `[C]`
- reference transcript: `[C]`
- model SHA: `9ade048c9d3692b411572a9a8ad615766168e62fb1d4c234973825a377c71984`
- audio SHA: `695ac1b2c85a0419b6ee052ef90cd09cd0c5688a1445aea735b66883d199e803`
- runtime surface: `speech/encoder/whisper+speech/decoder/whisper+speech/tokenizer/whisper`
- policy metadata: `english` / `transcribe` / `timestamp_tokens` /
  `suppress_translate=true` with three prompt tokens

## Maintained Benchmark Summary

`build/whisper_benchmark/benchmark_summary.json` records:

- status: `ok`
- reason: `ok`
- benchmark mode: `single_thread_cpu`
- warmups: `1`
- iterations: `3`
- EMEL transcript: `[C]`
- reference transcript: `[C]`
- EMEL mean process wall time: `70709972 ns`
- reference mean process wall time: `81716555 ns`
- matching model SHA:
  `9ade048c9d3692b411572a9a8ad615766168e62fb1d4c234973825a377c71984`

## Closeout Result

The prior audit blockers for Phase 122's historical scope were closed:

- `POLICY-01`: Phase 120 made the decoder public event carry the speech-owned ASR decode policy
  and routed policy validation through decoder SML guards.
- `TOK-02`: Phase 120 removed the public decoder `token:<id>` transcript surface; transcript text
  publication remains tokenizer-owned in the maintained runner.
- Baseline Nyquist: Phase 121 added archived-baseline validation artifacts for Phases 94-102.
- `CLOSE-01`: this phase reran source-backed closeout evidence and the full relevant quality gate.

Phase 122 no longer owns the active `CLOSE-01` proof. Later audits required the public recognizer
route, explicit recognizer route graph, and decoder-owned runtime execution, now closed by Phases
123-127.
