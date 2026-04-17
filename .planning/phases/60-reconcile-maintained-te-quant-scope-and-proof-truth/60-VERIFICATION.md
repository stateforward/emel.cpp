---
phase: 60-reconcile-maintained-te-quant-scope-and-proof-truth
status: complete
verified: 2026-04-16T16:05:00Z
---

# Phase 60 Verification

## Commands

- `cmake --build build/bench_tools_ninja --target embedding_generator_bench_runner -j4`
- `cmake --build build/coverage --target emel_tests_bin -j4`
- `./build/coverage/emel_tests_bin --no-breaks --test-case='maintained TE fixture is documented in tests/models README,maintained TE q5 fixture matches locked local size when present,TE q5 proof compares EMEL outputs against stored upstream goldens,embeddings generator initializes with TE q5 fixture when present,maintained TE fixture selector approves q8 and q5 only'`
- `EMEL_TE_FIXTURE=tests/models/TE-75M-q5_0.gguf EMEL_BENCH_ITERS=10 EMEL_BENCH_RUNS=5 ./build/bench_tools_ninja/embedding_generator_bench_runner`

## Results

- Focused q5 manifest/proof/runtime doctests passed:
  - `5 passed`
  - `0 failed`
  - `725 skipped`
- Maintained q5 steady-state benchmark result:
  - `text ns_per_op=6654545.800`
  - `audio ns_per_op=23228004.200`
  - `image ns_per_op=92363108.300`
- The q5 benchmark also reported stage timing splits:
  - text: `prepare_ns=14066.600`, `encode_ns=6637579.200`, `publish_ns=612.500`
  - audio: `prepare_ns=159566.900`, `encode_ns=23066937.500`, `publish_ns=487.500`
  - image: `prepare_ns=4612158.300`, `encode_ns=87699308.300`, `publish_ns=962.400`
- The maintained README and requirement manifest now contain the q5 fixture path, checksum, URL,
  and explicit approved-scope wording.
