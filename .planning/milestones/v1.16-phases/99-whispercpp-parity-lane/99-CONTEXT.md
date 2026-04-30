# Phase 99: whisper.cpp Parity Lane - Context

**Gathered:** 2026-04-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Add lane-isolated parity proof between the maintained EMEL Whisper runtime and a pinned
`whisper.cpp` reference build for the same fixture/audio pair. This phase owns fetching and
building the reference backend, generating the deterministic audio fixture, running the EMEL lane
through EMEL-owned code, and storing machine-readable comparison records. Single-thread benchmark
timing remains Phase 100 work.

</domain>

<decisions>
## Implementation Decisions

### Reference Pin
- Use `https://github.com/ggml-org/whisper.cpp.git` tag `v1.7.6`.
- Verify the tag resolves to commit `a8d002cfd879315632a579e73f0148d06959de36`.
- Keep the reference checkout and build under `build/whisper_cpp_reference/`.
- Build the CPU reference CLI locally; do not link it into `src/` or the EMEL lane.

### Fixture Pin
- Use the pinned `oxide-lab/whisper-tiny-GGUF` commit
  `94468a6c81edab8c594d9b1d06ea1dfb64292327`.
- Use the `whisper.cpp/whisper-tiny-q8_0.gguf` artifact for the reference lane.
- Verify SHA256
  `9ade048c9d3692b411572a9a8ad615766168e62fb1d4c234973825a377c71984`.
- Generate a deterministic 16 kHz mono one-second 440 Hz WAV fixture.

### Lane Isolation
- EMEL lane uses `tests/models/model-tiny-q80.gguf`, the EMEL GGUF loader, Whisper execution
  contract, encoder actor, and decoder actor.
- Reference lane invokes the pinned `whisper-cli` as a separate process with the pinned reference
  model and deterministic WAV.
- The compare script only exchanges output files and JSON records.

</decisions>

<code_context>
## Integration Points

- `scripts/setup_whisper_cpp_reference.sh` fetches, pins, verifies, and builds `whisper.cpp`.
- `scripts/bench_whisper_compare.sh` builds the EMEL parity runner and invokes the compare driver.
- `tools/bench/whisper_compare.py` writes `whisper_compare/v1` lane records and a
  `whisper_compare_summary/v1` summary.
- `tools/bench/whisper_emel_parity_runner.cpp` drives EMEL through public loader, encoder, and
  decoder events and emits the EMEL lane JSONL record.
- `scripts/quality_gates.sh` treats `EMEL_QUALITY_GATES_BENCH_SUITE=whisper_compare` as a custom
  gate that runs the wrapper instead of the generic benchmark runner.

</code_context>

<specifics>
## Specific Evidence Target

The maintained Phase 99 comparison group is:

`whisper/tiny/q8_0/phase99_440hz_16khz_mono`

The expected Phase 99 verdict is allowed to be `bounded_drift` while the EMEL runtime still
publishes a deterministic token-id transcript and the reference lane publishes text from the full
`whisper.cpp` decoding path. Operational failure is reserved for missing/failed lanes.

</specifics>

<deferred>
## Deferred Ideas

- Single-thread timing records and host/build metadata belong to Phase 100.
- Profiling and performance closure belong to Phase 101.
- Full text-tokenizer parity beyond the deterministic token-id transcript remains future runtime
  work unless required by Phase 100/101 evidence.

</deferred>
