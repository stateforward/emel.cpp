---
status: investigating
trigger: "Make EMEL PersonaPlex Moshi generate real speech end to end from the same macOS say WAV input as moshi.cpp, with fixed-seed reference comparison showing closely matching audio/token behavior and an output WAV the user can hear. CPU only."
created: 2026-07-09
updated: 2026-07-09
---

# Moshi PersonaPlex Audio Parity

## Symptoms

- expected: EMEL and moshi.cpp consume the same macOS `say` WAV with seed 1234 and produce closely matching PersonaPlex speech audio and token behavior on CPU.
- actual: moshi.cpp produces speech, while completed earlier EMEL attempts produced non-speech; the latest EMEL run reached generation frame 24 but was interrupted before writing a WAV.
- errors: no runtime error was reported; the incomplete run stopped without an output WAV.
- timeline: the current feature branch adds the EMEL Moshi generator and PersonaPlex model binding; parity has not yet been demonstrated.
- reproduction: run `build/emel_moshi_say_e2e_current` with the converted Mimi, Moshi LM, NATF0 voice, and `build/moshi_e2e/say_input_24k.wav` inputs.

## Current Focus

- hypothesis: EMEL diverges from moshi.cpp in depformer token generation after the first matching codebook token, before Mimi decoding.
- test: complete deterministic CPU traces for both implementations and compare the first generated frame codebook by codebook.
- expecting: the first mismatching codebook identifies the earliest model execution or sampling phase that needs correction.
- next_action: gather initial evidence
- reasoning_checkpoint:
- tdd_checkpoint:

## Evidence

- timestamp: 2026-07-09T13:00:00-05:00
  observation: the latest EMEL log completed 51 voice-prefill frames, 12 prompt-prefill frames, and generated frames 0 through 24, but no WAV exists.
  implication: PersonaPlex setup and generation dispatch run on CPU; audible output and parity remain unproven.

## Eliminated

## Resolution

- root_cause:
- fix:
- verification:
- files_changed:
