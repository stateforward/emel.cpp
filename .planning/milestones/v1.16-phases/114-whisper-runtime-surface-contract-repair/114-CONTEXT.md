---
phase: 114
status: ready
gathered: 2026-04-27
mode: autonomous
---

# Phase 114: Whisper Runtime Surface Contract Repair - Context

<domain>
## Phase Boundary

Make the maintained v1.16 Whisper ASR runtime surface explicit, source-backed, and
domain-boundary clean.
</domain>

<decisions>
## Implementation Decisions

### Runtime Surface
- The maintained v1.16 ASR surface is the speech-owned
  `speech/encoder/whisper` actor plus `speech/decoder/whisper` actor plus
  `speech/tokenizer/whisper` policy.
- The generic `speech::recognizer::sm` remains unsupported for Whisper in this milestone because
  the domain-boundary rules forbid a generic recognizer route that hardcodes Whisper contracts.
- Compare and benchmark evidence must name the selected surface explicitly.

### Requirement Interpretation
- `SPEECH-01` is satisfied by removing top-level Whisper runtime ownership and keeping Whisper ASR
  runtime under speech encoder/decoder/tokenizer ownership.
- `TOK-01`, `TOK-02`, and `POLICY-01` are satisfied by tokenizer checksum gates, control-token
  validation, and explicit tiny ASR decode policy.
- `PARITY-01` is satisfied by exact `[C]` parity through the selected surface.

### the agent's Discretion
- Keep the repair narrow: no new public recognizer route, no source tree movement, and no fallback
  runtime path.
</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/emel/speech/encoder/whisper/sm.hpp`
- `src/emel/speech/decoder/whisper/sm.hpp`
- `src/emel/speech/tokenizer/whisper/detail.hpp`
- `tools/bench/whisper_emel_parity_runner.cpp`

### Established Patterns
- Variant-specific Whisper runtime belongs under speech encoder/decoder/tokenizer components.
- Model-family binding stays under `src/emel/model/whisper`.
- Domain-boundary enforcement is centralized in `scripts/check_domain_boundaries.sh`.

### Integration Points
- `scripts/bench_whisper_compare.sh` and `scripts/bench_whisper_single_thread.sh` drive the
  maintained compare and benchmark publication lanes.
</code_context>

<specifics>
## Specific Ideas

The runtime contract should be truthful in source-produced JSON: the EMEL backend id and runtime
surface should identify the encoder/decoder/tokenizer lane, not a generic recognizer or a
decoder-only backend.
</specifics>

<deferred>
## Deferred Ideas

A future top-level recognizer route may be added only if it remains generic and variant-clean.
</deferred>
