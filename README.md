# EMEL

Deterministic, production-grade C++ inference engine built around Boost.SML orchestration.

## Status: WIP

This repository is under active development. APIs, state machines, and formats will change.
If you’re evaluating EMEL, expect fast iteration and breaking changes until the core loader,
allocator, and execution pipelines stabilize.

This inference engine is being implemented by AI under human engineering and architecture direction.

## Implementation priorities

1. Architect first, then scaffold cleanly.
2. Port math, instructions, and behavior without mirroring reference control flow.
3. Prove parity against llama.cpp.
4. Match model/tokenizer intent as defined by their creators (transformers).
5. Optimize once correctness is locked.

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
`scripts/test_with_sanitizers.sh`, `scripts/fuzz_smoke.sh`, `scripts/lint_snapshot.sh`,
and `scripts/bench.sh`.

### Why Zig for builds

Zig’s C/C++ toolchain gives us consistent, fast, cross-platform builds without forcing a full
dependency on any single system compiler or SDK. It keeps the default dev path reproducible,
while still allowing native toolchains when needed.

### Why CMake for tests and coverage

Coverage and CI tooling are already standardized around CMake + CTest + llvm-cov/gcovr in this
repo. Using CMake for test/coverage builds keeps gates deterministic and portable across CI
environments, while Zig remains the default for day-to-day builds.

## Documentation

- [Architecture](docs/architecture/) (generated state-machine docs + Mermaid diagrams)
- [Benchmarks](docs/benchmarks.md) (generated benchmark snapshot table)
- [SML Conventions](docs/third_party/sml.md) (Boost.SML conventions and usage)
- [Parity Audit](docs/gaps.md) (parity audit status)

## Docs index

- [`docs/benchmarks.md`](docs/benchmarks.md)
- [`docs/architecture/batch_planner.md`](docs/architecture/batch_planner.md)
- [`docs/architecture/buffer_allocator.md`](docs/architecture/buffer_allocator.md)
- [`docs/architecture/buffer_chunk_allocator.md`](docs/architecture/buffer_chunk_allocator.md)
- [`docs/architecture/buffer_planner.md`](docs/architecture/buffer_planner.md)
- [`docs/architecture/buffer_realloc_analyzer.md`](docs/architecture/buffer_realloc_analyzer.md)
- [`docs/architecture/decoder.md`](docs/architecture/decoder.md)
- [`docs/architecture/gbnf_lexer.md`](docs/architecture/gbnf_lexer.md)
- [`docs/architecture/gbnf_parser.md`](docs/architecture/gbnf_parser.md)
- [`docs/architecture/generator.md`](docs/architecture/generator.md)
- [`docs/architecture/graph_allocator.md`](docs/architecture/graph_allocator.md)
- [`docs/architecture/graph_assembler.md`](docs/architecture/graph_assembler.md)
- [`docs/architecture/graph_graph.md`](docs/architecture/graph_graph.md)
- [`docs/architecture/graph_processor.md`](docs/architecture/graph_processor.md)
- [`docs/architecture/kernel_aarch64.md`](docs/architecture/kernel_aarch64.md)
- [`docs/architecture/kernel_any.md`](docs/architecture/kernel_any.md)
- [`docs/architecture/kernel_cuda.md`](docs/architecture/kernel_cuda.md)
- [`docs/architecture/kernel_dispatch.md`](docs/architecture/kernel_dispatch.md)
- [`docs/architecture/kernel_events.md`](docs/architecture/kernel_events.md)
- [`docs/architecture/kernel_metal.md`](docs/architecture/kernel_metal.md)
- [`docs/architecture/kernel_ops.md`](docs/architecture/kernel_ops.md)
- [`docs/architecture/kernel.md`](docs/architecture/kernel.md)
- [`docs/architecture/kernel_vulkan.md`](docs/architecture/kernel_vulkan.md)
- [`docs/architecture/kernel_wasm.md`](docs/architecture/kernel_wasm.md)
- [`docs/architecture/kernel_x86_64.md`](docs/architecture/kernel_x86_64.md)
- [`docs/architecture/logits_sampler.md`](docs/architecture/logits_sampler.md)
- [`docs/architecture/logits_sampler_token_selector.md`](docs/architecture/logits_sampler_token_selector.md)
- [`docs/architecture/logits_validator.md`](docs/architecture/logits_validator.md)
- [`docs/architecture/memory_coordinator_hybrid.md`](docs/architecture/memory_coordinator_hybrid.md)
- [`docs/architecture/memory_coordinator_kv.md`](docs/architecture/memory_coordinator_kv.md)
- [`docs/architecture/memory_coordinator_recurrent.md`](docs/architecture/memory_coordinator_recurrent.md)
- [`docs/architecture/memory_coordinator.md`](docs/architecture/memory_coordinator.md)
- [`docs/architecture/memory_kv.md`](docs/architecture/memory_kv.md)
- [`docs/architecture/model_loader.md`](docs/architecture/model_loader.md)
- [`docs/architecture/model_weight_loader.md`](docs/architecture/model_weight_loader.md)
- [`docs/architecture/parser_gguf.md`](docs/architecture/parser_gguf.md)
- [`docs/architecture/parser.md`](docs/architecture/parser.md)
- [`docs/architecture/telemetry_exporter.md`](docs/architecture/telemetry_exporter.md)
- [`docs/architecture/telemetry_provider.md`](docs/architecture/telemetry_provider.md)
- [`docs/architecture/tensor_allocator.md`](docs/architecture/tensor_allocator.md)
- [`docs/architecture/tensor_lifetime_analyzer.md`](docs/architecture/tensor_lifetime_analyzer.md)
- [`docs/architecture/tensor_tensor.md`](docs/architecture/tensor_tensor.md)
- [`docs/architecture/tensor_view.md`](docs/architecture/tensor_view.md)
- [`docs/architecture/text_conditioner.md`](docs/architecture/text_conditioner.md)
- [`docs/architecture/text_detokenizer.md`](docs/architecture/text_detokenizer.md)
- [`docs/architecture/text_encoders_bpe.md`](docs/architecture/text_encoders_bpe.md)
- [`docs/architecture/text_encoders_fallback.md`](docs/architecture/text_encoders_fallback.md)
- [`docs/architecture/text_encoders_plamo2.md`](docs/architecture/text_encoders_plamo2.md)
- [`docs/architecture/text_encoders_rwkv.md`](docs/architecture/text_encoders_rwkv.md)
- [`docs/architecture/text_encoders.md`](docs/architecture/text_encoders.md)
- [`docs/architecture/text_encoders_spm.md`](docs/architecture/text_encoders_spm.md)
- [`docs/architecture/text_encoders_ugm.md`](docs/architecture/text_encoders_ugm.md)
- [`docs/architecture/text_encoders_wpm.md`](docs/architecture/text_encoders_wpm.md)
- [`docs/architecture/text_formatter.md`](docs/architecture/text_formatter.md)
- [`docs/architecture/text_jinja_formatter.md`](docs/architecture/text_jinja_formatter.md)
- [`docs/architecture/text_jinja_parser.md`](docs/architecture/text_jinja_parser.md)
- [`docs/architecture/text_renderer.md`](docs/architecture/text_renderer.md)
- [`docs/architecture/text_tokenizer_preprocessor_bpe.md`](docs/architecture/text_tokenizer_preprocessor_bpe.md)
- [`docs/architecture/text_tokenizer_preprocessor_fallback.md`](docs/architecture/text_tokenizer_preprocessor_fallback.md)
- [`docs/architecture/text_tokenizer_preprocessor_plamo2.md`](docs/architecture/text_tokenizer_preprocessor_plamo2.md)
- [`docs/architecture/text_tokenizer_preprocessor_rwkv.md`](docs/architecture/text_tokenizer_preprocessor_rwkv.md)
- [`docs/architecture/text_tokenizer_preprocessor.md`](docs/architecture/text_tokenizer_preprocessor.md)
- [`docs/architecture/text_tokenizer_preprocessor_spm.md`](docs/architecture/text_tokenizer_preprocessor_spm.md)
- [`docs/architecture/text_tokenizer_preprocessor_ugm.md`](docs/architecture/text_tokenizer_preprocessor_ugm.md)
- [`docs/architecture/text_tokenizer_preprocessor_wpm.md`](docs/architecture/text_tokenizer_preprocessor_wpm.md)
- [`docs/architecture/text_tokenizer.md`](docs/architecture/text_tokenizer.md)
- [`docs/architecture/token_batcher.md`](docs/architecture/token_batcher.md)


## Regenerating docs

```bash
scripts/generate_docs.sh
```

Use `scripts/generate_docs.sh --check` in CI to validate generated artifacts.