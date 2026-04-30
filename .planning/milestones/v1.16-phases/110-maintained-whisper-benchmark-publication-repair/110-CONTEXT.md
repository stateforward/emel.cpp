---
phase: 110
status: ready
gathered: 2026-04-27
mode: autonomous
---

# Phase 110: Maintained Whisper Benchmark Publication Repair - Context

<domain>
## Phase Boundary

Repair only the maintained single-thread Whisper benchmark publication lane. The phase owns
default model-path truth, deterministic reference policy parity with the maintained compare lane,
and hard failure when EMEL/reference model SHA or transcript differs.
</domain>

<decisions>
## Implementation Decisions

- The EMEL benchmark lane must default to the pinned Phase 99 source model path and use the
  source-owned legacy Whisper conversion path through `whisper_emel_parity_runner`.
- Benchmark summary publication must not report `ok` when the lane model SHA values or transcripts
  differ.
- Reference CLI decode policy must match the maintained compare lane's deterministic settings.
</decisions>

<code_context>
## Existing Code Insights

- `scripts/bench_whisper_single_thread.sh` orchestrates the single-thread benchmark wrapper.
- `tools/bench/whisper_benchmark.py` records benchmark JSONL rows and writes
  `benchmark_summary.json`.
- `tools/bench/whisper_compare.py` already invokes the reference lane with deterministic
  `--audio-ctx 50`, `--beam-size 1`, `--best-of 1`, and `--no-fallback`.
</code_context>

<deferred>
## Deferred Ideas

No scope widening beyond the pinned tiny q8_0 Phase 99 benchmark slice.
</deferred>
