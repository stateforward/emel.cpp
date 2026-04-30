---
phase: 103
name: Speech Recognizer Domain Cleanup
status: superseded_by_phase_115
---

# Phase 103 Context

Phase 103 originally reopened the v1.16 domain-cleanup work after a bounded transcript drift was
accepted too early. Its historical goal remains valid: remove any top-level Whisper runtime domain
and keep model binding separate from runtime orchestration.

## Corrected Ownership

- `src/emel/speech/encoder/whisper/**` owns the maintained Whisper speech encoder actor.
- `src/emel/speech/decoder/whisper/**` owns the maintained Whisper speech decoder actor.
- `src/emel/speech/tokenizer/whisper/**` owns the maintained Whisper tokenizer and decode policy.
- `src/emel/model/whisper/**` owns Whisper GGUF/model binding validation.
- Shared numeric work remains in existing kernel-owned files; there is no Whisper-specific kernel
  root.

## Supersession

Earlier Phase 103 text described a recognizer-internal Whisper runtime and a Whisper-specific
kernel root. That was not live source truth and is superseded by Phase 114 and Phase 115.
