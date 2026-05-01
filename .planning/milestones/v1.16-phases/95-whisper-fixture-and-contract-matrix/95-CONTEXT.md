# Phase 95: Whisper Fixture And Contract Matrix — Context

**Phase Goal:** Pin the requested model family and validate Whisper tiny GGUF architecture,
tensors, tokenizer, and audio contract.

**Requirements covered:** FIX-01, FIX-02, FIX-03.

## Narrowed Variant-Family Scope (Approved 2026-04-25)

Phase 94's starting-point audit and a Phase 95 upstream check of the
`oxide-lab/whisper-tiny-GGUF` repo at the pinned commit
`94468a6c81edab8c594d9b1d06ea1dfb64292327` confirm that the upstream repo only ships **three**
EMEL-loadable Candle-style GGUFs at the top level:

| Variant | File | Size (bytes) | SHA256 |
|---------|------|--------------|--------|
| q8_0 | `model-tiny-q80.gguf` | `40700160` | `52deb0fdcbb9c36b4d570e35f5a65a5ad4275ccdb85e7a06e81a8b05b3743c9d` |
| q4_0 | `whisper-tiny-q4_0.gguf` | `22087104` | `b2be6457e86d2c917d0c0eecef8e041ed03c60f64fc5744e6720adfb5141c21b` |
| q4_1 | `whisper-tiny-q4_1.gguf` | `24414464` | `7d40a062a67abeb53784edd326610035089164c9c261cbcfa628e017a07e7a3a` |

The remaining sibling files under the `whisper.cpp/` subdirectory
(`whisper-tiny-q2_k.gguf`, `q3_k`, `q4_0`, `q4_1`, `q4_k`, `q5_0`, `q5_1`, `q5_k`, `q6_k`,
`q8_0`) use whisper.cpp's `lmgg` binary format (not GGUF), and are reserved for the
whisper.cpp reference lane. They cannot bootstrap EMEL model state.

agent-0 approved Option A (narrow scope) on 2026-04-25 (`J2bGANuAk61CW6P6`). The v1.16
maintained Whisper tiny GGUF variant family is therefore `{q8_0, q4_0, q4_1}` for the
EMEL-owned runtime/loader/parity lanes. q5/q6/qK variants would require a future approved
EMEL-owned conversion phase before they can be claimed.

## What's Already Landed

- `src/emel/model/whisper/detail.hpp` declares the Whisper execution contract (sample rate,
  mel bin count, vocab size, embed/feed-forward widths, encoder/decoder context, family views).
- `src/emel/model/whisper/detail.cpp` enforces tiny-contract checks
  (`n_mels=80`, `n_vocab=51865`, `n_embd=384`, `n_ff=1536`, `n_head=6`, encoder ctx `1500`,
  decoder ctx `448`, encoder/decoder block count `4`) and tensor-shape requirements for
  `mel_filters`, `model.encoder.conv1.weight`, `model.encoder.embed_positions.weight`,
  `model.decoder.embed_tokens.weight`, `model.decoder.embed_positions.weight`, plus
  per-block `q_proj.weight` presence for encoder self-attention and decoder cross-attention.
- `src/emel/model/architecture/detail.cpp` registers `"whisper"` with `load_hparams` and
  `validate_execution_contract` hooks, so EMEL's loader recognizes Whisper natively without
  whisper.cpp linkage.
- `src/emel/model/data.{hpp,cpp}` exposes `is_whisper_execution_architecture(...)`.
- `tests/model/loader/lifecycle_tests.cpp` covers loader hparam acceptance/rejection,
  contract build, missing-decoder-cross-attention rejection, non-canonical vocab rejection,
  real-fixture parse from `tests/models/model-tiny-q80.gguf`, and tokenizer-metadata
  extraction via the EMEL-owned GGUF parser (no whisper.cpp/ggml objects involved).
- `tests/models/README.md` already documents the q80 root fixture provenance with explicit
  variant-family scope and loader/runtime boundary notes (Phase 94).
- `tests/model/fixture_manifest_tests.cpp` regression-locks the q80 README wording.

## What Phase 95 Must Add Or Tighten

1. **SC1 — Pinned URLs/checksums/sizes for the maintained variant family.** Document the q4_0
   and q4_1 sibling fixtures with the same provenance fields used for q80 (URL, commit,
   size, SHA256, license, variant identity). Replace the broad 11-variant scope note with a
   narrowed `{q8_0, q4_0, q4_1}` scope note that explains why the `whisper.cpp/` subdir
   variants are reference-lane only.
2. **SC2 — Validation accepts maintained contract and rejects unsupported variants.** Add
   focused tests proving that the EMEL loader/contract acceptance is dtype-agnostic across
   the maintained variant set (the existing tensor-shape contract is already dtype-agnostic),
   and add a focused regression that the loader rejects the upstream `whisper.cpp/`
   `lmgg`-magic siblings without leaking partial state.
3. **SC3 — Source-backed metadata.** Make sure tensor, tokenizer, mel-filter, encoder, and
   decoder metadata comes from EMEL's GGUF parsing only. Existing tests already cover
   tokenizer and tensor metadata for the q80 fixture; this phase adds an opportunistic
   q4_0/q4_1 real-fixture check guarded on local fixture presence so CI degrades gracefully
   when the larger variants are not staged.
4. **SC4 — No whisper.cpp object bootstraps EMEL state.** Add a static guardrail asserting
   that no Whisper source/test in `src/emel/model/whisper/**` includes any `whisper.h`,
   `<whisper>`, or `ggml.h` header. The reference lane wrapper in
   `scripts/bench_whisper_reference_whisper_cpp.sh` is allowed to use `whisper.cpp` but is
   excluded from EMEL-owned source/tests.

## Constraints

- AGENTS.md `extern "C"`/error-code rules already apply at the loader boundary — no new C ABI
  in this phase.
- The `whisper/detail.cpp` contract validator has no SML state machine yet (this phase is
  loader/contract truth only). Runtime SML orchestration is out of scope for Phase 95 and
  belongs to ASR runtime phases (97/98).
- Hot-path allocation rules don't apply: this phase is one-time loader/initialization work.
- Reference-policy rule: scripts that drive whisper.cpp must remain isolated and must not
  reach into EMEL-owned headers from the EMEL side.

## Definition Of Done

- `tests/models/README.md` documents q4_0 and q4_1 provenance and narrows the variant-family
  scope to `{q8_0, q4_0, q4_1}` with `lmgg` reference-lane note retained.
- `.planning/REQUIREMENTS.md` `KERN-01` lists only `{q4_0, q4_1, q8_0}`.
- `tests/model/fixture_manifest_tests.cpp` adds regression coverage for the new variant
  metadata, the narrowed scope wording, and the variant ledger.
- `tests/model/loader/lifecycle_tests.cpp` adds tests for q4_0/q4_1 fixture parse (skip when
  fixtures absent) and a static guard that EMEL Whisper source includes no whisper.cpp/ggml
  headers.
- Focused doctest run passes; scoped `scripts/quality_gates.sh` (changed-files mode) passes.
- 95-01-SUMMARY.md and 95-VERIFICATION.md written and referenced from STATE.md/ROADMAP.md.
