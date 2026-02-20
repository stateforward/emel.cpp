# EMEL

Deterministic, production-grade C++ inference engine built around Boost.SML orchestration.

## Status: WIP

This repository is under active development. APIs, state machines, and formats will change.
If you’re evaluating EMEL, expect fast iteration and breaking changes until the core loader,
allocator, and execution pipelines stabilize.

This inference engine is being implemented by AI under human engineering and architecture direction.

## Why EMEL

EMEL exists to make inference behavior explicit and verifiable. Instead of ad-hoc control flow,
orchestration is modeled as Boost.SML state machines with deterministic, testable transitions.
That enables:

1. Clear operational semantics and failure modes.
2. Deterministic, reproducible inference paths.
3. High-performance, C-compatible boundaries without dynamic dispatch in hot paths.
4. Auditable parity work against reference implementations without copying their control flow.

## The name

“EMEL” is pronounced like “ML”. It’s a short, neutral name that doesn’t carry existing
assumptions or baggage. It’s intentionally low-ceremony while we iterate on the core design.

## Build and test

```bash
scripts/quality_gates.sh
```

Individual gates live in `scripts/build_with_zig.sh`, `scripts/test_with_coverage.sh`,
`scripts/lint_snapshot.sh`, and `scripts/bench.sh`.

### Why Zig for builds

Zig’s C/C++ toolchain gives us consistent, fast, cross-platform builds without forcing a full
dependency on any single system compiler or SDK. It keeps the default dev path reproducible,
while still allowing native toolchains when needed.

### Why CMake for tests and coverage

Coverage and CI tooling are already standardized around CMake + CTest + llvm-cov/gcovr in this
repo. Using CMake for test/coverage builds keeps gates deterministic and portable across CI
environments, while Zig remains the default for day-to-day builds.

## Documentation

- `docs/architecture/` (generated state-machine docs + Mermaid diagrams)
- `docs/benchmarks.md` (generated benchmark snapshot table)
- `docs/sml.md` (Boost.SML conventions and usage)
- `docs/gaps.md` (parity audit status)

## Docs index

- [`docs/benchmarks.md`](docs/benchmarks.md)
- [`docs/architecture/batch_splitter.md`](docs/architecture/batch_splitter.md)
- [`docs/architecture/buffer_allocator.md`](docs/architecture/buffer_allocator.md)
- [`docs/architecture/buffer_chunk_allocator.md`](docs/architecture/buffer_chunk_allocator.md)
- [`docs/architecture/buffer_planner.md`](docs/architecture/buffer_planner.md)
- [`docs/architecture/buffer_realloc_analyzer.md`](docs/architecture/buffer_realloc_analyzer.md)
- [`docs/architecture/decoder_compute_executor.md`](docs/architecture/decoder_compute_executor.md)
- [`docs/architecture/decoder.md`](docs/architecture/decoder.md)
- [`docs/architecture/decoder_ubatch_executor.md`](docs/architecture/decoder_ubatch_executor.md)
- [`docs/architecture/encoder_bpe.md`](docs/architecture/encoder_bpe.md)
- [`docs/architecture/encoder_fallback.md`](docs/architecture/encoder_fallback.md)
- [`docs/architecture/encoder_plamo2.md`](docs/architecture/encoder_plamo2.md)
- [`docs/architecture/encoder_rwkv.md`](docs/architecture/encoder_rwkv.md)
- [`docs/architecture/encoder.md`](docs/architecture/encoder.md)
- [`docs/architecture/encoder_spm.md`](docs/architecture/encoder_spm.md)
- [`docs/architecture/encoder_ugm.md`](docs/architecture/encoder_ugm.md)
- [`docs/architecture/encoder_wpm.md`](docs/architecture/encoder_wpm.md)
- [`docs/architecture/gbnf_parser.md`](docs/architecture/gbnf_parser.md)
- [`docs/architecture/generator.md`](docs/architecture/generator.md)
- [`docs/architecture/kv_cache.md`](docs/architecture/kv_cache.md)
- [`docs/architecture/memory_coordinator.md`](docs/architecture/memory_coordinator.md)
- [`docs/architecture/model_loader.md`](docs/architecture/model_loader.md)
- [`docs/architecture/model_weight_loader.md`](docs/architecture/model_weight_loader.md)
- [`docs/architecture/parser_gguf.md`](docs/architecture/parser_gguf.md)
- [`docs/architecture/parser.md`](docs/architecture/parser.md)
- [`docs/architecture/sampler_candidate_builder.md`](docs/architecture/sampler_candidate_builder.md)
- [`docs/architecture/sampler_pipeline.md`](docs/architecture/sampler_pipeline.md)
- [`docs/architecture/sampler_token_selector.md`](docs/architecture/sampler_token_selector.md)
- [`docs/architecture/telemetry_exporter.md`](docs/architecture/telemetry_exporter.md)
- [`docs/architecture/telemetry_provider.md`](docs/architecture/telemetry_provider.md)
- [`docs/architecture/tensor_allocator.md`](docs/architecture/tensor_allocator.md)
- [`docs/architecture/tensor_lifetime_analyzer.md`](docs/architecture/tensor_lifetime_analyzer.md)
- [`docs/architecture/tokenizer.md`](docs/architecture/tokenizer.md)

## Regenerating docs

```bash
scripts/generate_docs.sh
```

Use `scripts/generate_docs.sh --check` in CI to validate generated artifacts.
