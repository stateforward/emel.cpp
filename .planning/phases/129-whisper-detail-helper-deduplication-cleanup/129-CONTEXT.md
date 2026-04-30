---
phase: 129
status: ready
created: 2026-04-28
goal: "Remove duplicate decoder/timestamp helpers from encoder detail while preserving decoder ownership."
requirements: []
---

# Phase 129: Whisper Detail Helper Deduplication Cleanup - Context

**Gathered:** 2026-04-28
**Status:** Ready for implementation

<domain>
## Phase Boundary

Close the remaining non-blocking v1.16 audit debt by removing stale decoder-owned runtime helpers
from `src/emel/speech/encoder/whisper/detail.hpp`. This is cleanup only; it must not reopen active
requirements or change the maintained recognizer-backed runtime path.

</domain>

<decisions>
## Implementation Decisions

### Ownership Cleanup
- Treat `src/emel/speech/decoder/whisper/detail.hpp` as the sole owner for decoder sequence,
  logits, timestamp-policy, decoder workspace, and token-selection helpers.
- Keep encoder detail focused on audio frontend and encoder execution helpers.
- Do not introduce a shared helper surface because the duplicate code is currently used only by
  decoder runtime and tests; moving it to shared detail would widen ownership unnecessarily.

### Validation
- Move timestamp-policy helper tests from encoder detail tests to the existing decoder test file
  so no snapshot-baseline update is required.
- Add a source-level regression proving encoder detail does not contain decoder runtime helper
  names.
- Preserve the existing decoder production ownership regression and maintained compare evidence.

</decisions>

<code_context>
## Existing Code Insights

- `src/emel/speech/encoder/whisper/detail.hpp` duplicated decoder constants and helpers such as
  `decode_policy_runtime`, `required_decoder_workspace_floats`, decoder cross-cache/logit
  helpers, timestamp-aware token selection, and `run_decoder_sequence`.
- `src/emel/speech/decoder/whisper/detail.hpp` owns the live decoder runtime helpers used by
  decoder actions.
- `tests/speech/encoder/whisper/detail_tests.cpp` still covered timestamp helper behavior through
  the encoder namespace, which kept the duplicate code reachable from tests.
- `tests/speech/decoder/whisper/lifecycle_tests.cpp` already has a production-source regression
  that checks decoder files do not include or alias encoder detail and can host the decoder-owned
  timestamp helper coverage.

</code_context>

<deferred>
## Deferred Ideas

None.

</deferred>
