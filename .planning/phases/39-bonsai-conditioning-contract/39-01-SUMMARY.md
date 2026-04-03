---
phase: 39-bonsai-conditioning-contract
plan: 01
completed: 2026-04-02
status: implemented
---

# Phase 39 Summary

Phase 39 moved the maintained embedded-template contract into `src` and made it explicit in the
runtime path. `src/emel/text/formatter/format.hpp` now owns supported-template marker matching,
supported ChatML/Qwen formatter implementations, contract-kind resolution from loaded model
metadata, and request-shape validation helpers.

Generator construction now resolves formatter binding from `model::data.meta.tokenizer_data`,
stores the explicit contract kind, and passes it into conditioner bind. Conditioner prepare guards
now reject unsupported request shapes before formatting/tokenization, which makes tool roles,
missing generation prompts, thinking-enabled requests, and unsupported named-template variants fail
as explicit invalid requests instead of hidden raw fallbacks.

The maintained bench/parity GGUF loaders now populate `model_data.meta.tokenizer_data` with the
embedded primary chat template and named template variants, so runtime contract resolution uses the
same loaded metadata truth as the fixture tools. Focused formatter and conditioner doctests were
added to lock this behavior down.
