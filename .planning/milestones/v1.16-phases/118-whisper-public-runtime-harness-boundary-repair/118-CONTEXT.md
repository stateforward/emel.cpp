---
phase: 118
status: context
requirements:
  - SPEECH-01
  - TOK-02
  - POLICY-01
  - PARITY-01
  - PERF-03
created: 2026-04-27
---

# Phase 118 Context

## Audit Inputs

The milestone audit reopened the Whisper closeout because the maintained EMEL proof path still
reached implementation internals:

- `tools/bench/whisper_emel_parity_runner.cpp` included speech/model Whisper `detail.hpp` files.
- The runner called actor detail helpers for contract binding, workspace sizing, tokenizer policy,
  and token-id transcript decode.
- `POLICY-01` was only partially source-backed because the policy object exposed fields beyond the
  behavior the maintained compare lane actually used.

## Rule Constraints

- Follow `docs/rules/sml.rules.md`; no queues, no reentrant actor dispatch, no action-owned runtime
  branching changes.
- Keep Whisper runtime under `speech/encoder/whisper`, `speech/decoder/whisper`, and
  `speech/tokenizer/whisper`; model-family binding remains under `model/whisper`.
- Keep EMEL and `whisper.cpp` benchmark/reference lanes isolated.
- Do not make generic recognizer or generator headers expose model-family contracts.

## Important Observation

For the pinned Phase 99 fixture, adding the Whisper no-timestamps prompt token changes both the
maintained EMEL behavior and the matching `whisper.cpp --no-timestamps` reference behavior away
from `[C]`. The source-backed policy must therefore be narrowed to the maintained timestamp-token
mode rather than continuing to claim no-timestamps behavior.
