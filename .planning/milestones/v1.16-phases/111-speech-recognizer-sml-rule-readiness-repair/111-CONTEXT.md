---
phase: 111
status: superseded_by_phase_115
gathered: 2026-04-27
mode: autonomous
---

# Phase 111: Speech Recognizer SML Rule Readiness Repair - Context

<domain>
## Phase Boundary

Phase 111 investigated SML rule-readiness risks for a proposed top-level recognizer route. The
current source-backed contract does not use that route for Whisper.
</domain>

<decisions>
## Implementation Decisions

- The generic recognizer remains unsupported for Whisper in v1.16.
- Active SML proof is carried by the speech encoder and speech decoder actors plus tokenizer
  policy checks.
- Future recognizer work must be variant-clean and approved as a separate phase.
</decisions>

<code_context>
## Existing Code Insights

- `src/emel/speech/recognizer/guards.hpp` intentionally rejects model readiness and backend
  readiness for the generic recognizer.
- `src/emel/speech/encoder/whisper/sm.hpp` and
  `src/emel/speech/decoder/whisper/sm.hpp` are the maintained Whisper actors.
</code_context>

<deferred>
## Deferred Ideas

No top-level recognizer route is added in v1.16.
</deferred>
