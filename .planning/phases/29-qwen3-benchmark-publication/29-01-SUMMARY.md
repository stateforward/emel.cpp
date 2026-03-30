---
phase: 29-qwen3-benchmark-publication
plan: 01
completed: 2026-03-28
commit: 43d8220
---

# Phase 29 Plan 01 Summary

The maintained bench compare lane now publishes the canonical Qwen3 benchmark identity explicitly:
`tools/bench` accepts the canonical `qwen3` metadata path, reference tokenization uses the same
GGUF-derived formatter contract already proved in parity, and compare output publishes explicit
formatter, runtime, and native `q8_0` evidence on Qwen-labelled `1/10/100/1000` generation rows.
