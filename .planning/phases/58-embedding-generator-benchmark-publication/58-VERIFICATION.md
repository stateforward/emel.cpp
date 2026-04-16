---
phase: 58-embedding-generator-benchmark-publication
status: passed
completed: 2026-04-15
---

# Phase 58 Verification

## Focused Verification

1. `cmake --build build/bench_tools_ninja --target embedding_generator_bench_runner -j4`
   Result: passed. The dedicated maintained embedding benchmark runner rebuilt cleanly after the
   embedding-generator bench sources were wired into `tools/bench`.

2. `./build/bench_tools_ninja/embedding_generator_bench_runner`
   Result: passed. The runner printed the maintained TE fixture metadata plus all three
   embedding-generator cases through the real actor surface:
   - fixture: `tests/models/TE-75M-q8_0.gguf`
   - vocab: `tests/models/mdbr-leaf-ir-vocab.txt`
   - prompt: `tests/embeddings/fixtures/te75m/red-square.txt`
   - text case:
     `embeddings/generator/steady_request/te_75m_q8_0_text_red_square_full_dim ns_per_op=6315666.000 prepare_ns=16375.000 encode_ns=6307541.000 publish_ns=500.000`
   - image case:
     `embeddings/generator/steady_request/te_75m_q8_0_image_red_square_full_dim ns_per_op=731571375.000 prepare_ns=4580708.000 encode_ns=726904583.000 publish_ns=750.000`
   - audio case:
     `embeddings/generator/steady_request/te_75m_q8_0_audio_pure_tone_440hz_full_dim ns_per_op=55392500.000 prepare_ns=30050875.000 encode_ns=25381709.000 publish_ns=4083.000`

3. `scripts/quality_gates.sh`
   Result: passed on the current code state after the later kernel-tail `nan` fix. The repo-wide
   gate kept coverage above threshold (`lines: 90.4%`, `branches: 55.1%`), paritychecker tests
   passed, and fuzz smoke completed. Benchmark snapshot and benchmark-marker issues remain
   warning-only under the current repo policy.

## Evidence

- `tools/bench/embedding_generator_bench.cpp` constructs initialized text, image, and audio
  generator states and drives public `initialize` / `embed_*` events through
  `emel::embeddings::generator::sm`.
- `tools/bench/embedding_generator_bench_main.cpp` prints exact maintained fixture paths and
  modality request contracts so the published numbers are auditable.
- `tools/bench/CMakeLists.txt` now exposes the dedicated `embedding_generator_bench_runner`
  surface without coupling the EMEL lane to any reference-only benchmark path.

## Residual Note

Phase `58.1` intentionally narrows the timed portion of this benchmark to steady-state throughput,
but the maintained Phase `58` publication contract remains satisfied because the runner still owns
real initialization plus real request dispatch through the maintained embedding-generator actor.
