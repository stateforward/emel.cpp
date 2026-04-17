---
phase: 58-embedding-generator-benchmark-publication
plan: 01
status: complete
completed: 2026-04-15
requirements-completed: []
---

# Phase 58 Summary

## Outcome

Phase 58 is complete. `tools/bench` now exposes a maintained TE embedding benchmark surface that
drives the real `src/emel/embeddings/generator` actor through initialize plus real text, image,
and audio embedding requests instead of timing helper-local execution.

## Delivered

- Added `tools/bench/embedding_generator_bench.cpp` and
  `tools/bench/embedding_generator_bench_main.cpp`.
- Wired a dedicated `embedding_generator_bench_runner` target in
  `tools/bench/CMakeLists.txt` so the maintained embedding cases live on their own EMEL-owned
  runner rather than distorting the compare-mode harness.
- Reused the maintained TE fixture and payload builders from
  `tests/embeddings/te_fixture_data.hpp` so the runner prints the exact fixture, vocab, prompt,
  and request-contract metadata it exercises.
- Drove `initialize`, `embed_text`, `embed_image`, and `embed_audio` through
  `emel::embeddings::generator::sm`, keeping the benchmark anchored on public event interfaces.
- Published dedicated maintained cases for:
  - `embeddings/generator/steady_request/te_75m_q8_0_text_red_square_full_dim`
  - `embeddings/generator/steady_request/te_75m_q8_0_image_red_square_full_dim`
  - `embeddings/generator/steady_request/te_75m_q8_0_audio_pure_tone_440hz_full_dim`

## Validation

- The dedicated runner rebuilt successfully in `build/bench_tools_ninja`.
- A focused runner pass printed the maintained TE fixture and all three modality request contracts,
  then produced auditable text, image, and audio benchmark rows through the embedding generator.
- The repo-wide gate rerun on the same code state remained clean after the later kernel-tail `nan`
  fix, so the maintained benchmark publication does not reopen the prior embedding/runtime blocker.

## Follow-On

- Phase `58.1` narrows the timed portion of this surface to steady-state throughput and exposes
  `prepare` / `encode` / `publish` attribution on top of the maintained publication lane.
