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

### Why state machines everywhere

It might look like over-engineering — "I have a hammer so everything looks like a nail." But a
state machine with two states has virtually zero overhead, and the goal is explicit behavior
modeling, not complexity for its own sake. Stateless functions inevitably accumulate conditional
logic as the code evolves: mode flags, error booleans, retry counters, phase enums. Taming that
accidental complexity before it starts is the whole point of EMEL. Every actor has a visible
state model, every transition is declared, and every unexpected event has a defined handler.
That's the trade I'm making.

End-to-end performance will be inferior to llama.cpp and other engines initially — that's
expected and accepted, even though many individual machines will perform comparably or better in
isolation. Having explicit actions and states makes it straightforward to find hotspots, and if
profiling shows a state machine itself is the bottleneck, it gets removed. Concurrency is
intentionally deferred until single-threaded behavior is verified. That doesn't mean there's no
plan for it — the actor model makes adding concurrency easier than it looks, and it will be
introduced only where measurement says it's necessary.

## The name

“EMEL” is pronounced like “ML”. It’s a short, neutral name that doesn’t carry existing
assumptions or baggage. It’s intentionally low-ceremony while I iterate on the core design.

## Acknowledgements

Huge thanks to the contributors of [llama.cpp](https://github.com/ggml-org/llama.cpp) and
[ggml](https://github.com/ggml-org/ggml). EMEL’s parity work depends on the quality and clarity
of these reference implementations.

Special shout out to [Georgi Gerganov](https://github.com/ggerganov), whose work created the
foundation that made this ecosystem possible.

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
- [`docs/architecture/batch_planner_modes_equal.md`](docs/architecture/batch_planner_modes_equal.md)
- [`docs/architecture/batch_planner_modes_sequential.md`](docs/architecture/batch_planner_modes_sequential.md)
- [`docs/architecture/batch_planner_modes_simple.md`](docs/architecture/batch_planner_modes_simple.md)
- [`docs/architecture/batch_planner.md`](docs/architecture/batch_planner.md)
- [`docs/architecture/gbnf_rule_parser_definition_parser.md`](docs/architecture/gbnf_rule_parser_definition_parser.md)
- [`docs/architecture/gbnf_rule_parser_expression_parser.md`](docs/architecture/gbnf_rule_parser_expression_parser.md)
- [`docs/architecture/gbnf_rule_parser_lexer.md`](docs/architecture/gbnf_rule_parser_lexer.md)
- [`docs/architecture/gbnf_rule_parser_nonterm_parser.md`](docs/architecture/gbnf_rule_parser_nonterm_parser.md)
- [`docs/architecture/gbnf_rule_parser.md`](docs/architecture/gbnf_rule_parser.md)
- [`docs/architecture/gbnf_rule_parser_term_parser.md`](docs/architecture/gbnf_rule_parser_term_parser.md)
- [`docs/architecture/gbnf_sampler_accept_parser.md`](docs/architecture/gbnf_sampler_accept_parser.md)
- [`docs/architecture/gbnf_sampler_candidate_parser.md`](docs/architecture/gbnf_sampler_candidate_parser.md)
- [`docs/architecture/gbnf_sampler_matcher_parser.md`](docs/architecture/gbnf_sampler_matcher_parser.md)
- [`docs/architecture/gbnf_sampler.md`](docs/architecture/gbnf_sampler.md)
- [`docs/architecture/gbnf_sampler_token_parser.md`](docs/architecture/gbnf_sampler_token_parser.md)
- [`docs/architecture/generator_initializer.md`](docs/architecture/generator_initializer.md)
- [`docs/architecture/generator_prefill.md`](docs/architecture/generator_prefill.md)
- [`docs/architecture/generator.md`](docs/architecture/generator.md)
- [`docs/architecture/gguf_loader.md`](docs/architecture/gguf_loader.md)
- [`docs/architecture/graph_allocator_liveness_pass.md`](docs/architecture/graph_allocator_liveness_pass.md)
- [`docs/architecture/graph_allocator_ordering_pass.md`](docs/architecture/graph_allocator_ordering_pass.md)
- [`docs/architecture/graph_allocator_placement_pass.md`](docs/architecture/graph_allocator_placement_pass.md)
- [`docs/architecture/graph_allocator.md`](docs/architecture/graph_allocator.md)
- [`docs/architecture/graph_assembler_assemble_alloc_pass.md`](docs/architecture/graph_assembler_assemble_alloc_pass.md)
- [`docs/architecture/graph_assembler_assemble_build_pass.md`](docs/architecture/graph_assembler_assemble_build_pass.md)
- [`docs/architecture/graph_assembler_assemble_validate_pass.md`](docs/architecture/graph_assembler_assemble_validate_pass.md)
- [`docs/architecture/graph_assembler_reserve_alloc_pass.md`](docs/architecture/graph_assembler_reserve_alloc_pass.md)
- [`docs/architecture/graph_assembler_reserve_build_pass.md`](docs/architecture/graph_assembler_reserve_build_pass.md)
- [`docs/architecture/graph_assembler_reserve_validate_pass.md`](docs/architecture/graph_assembler_reserve_validate_pass.md)
- [`docs/architecture/graph_assembler_reuse_decision_pass.md`](docs/architecture/graph_assembler_reuse_decision_pass.md)
- [`docs/architecture/graph_assembler.md`](docs/architecture/graph_assembler.md)
- [`docs/architecture/graph_processor_alloc_step.md`](docs/architecture/graph_processor_alloc_step.md)
- [`docs/architecture/graph_processor_bind_step.md`](docs/architecture/graph_processor_bind_step.md)
- [`docs/architecture/graph_processor_extract_step.md`](docs/architecture/graph_processor_extract_step.md)
- [`docs/architecture/graph_processor_kernel_step.md`](docs/architecture/graph_processor_kernel_step.md)
- [`docs/architecture/graph_processor_prepare_step.md`](docs/architecture/graph_processor_prepare_step.md)
- [`docs/architecture/graph_processor.md`](docs/architecture/graph_processor.md)
- [`docs/architecture/graph_processor_validate_step.md`](docs/architecture/graph_processor_validate_step.md)
- [`docs/architecture/graph.md`](docs/architecture/graph.md)
- [`docs/architecture/graph_tensor.md`](docs/architecture/graph_tensor.md)
- [`docs/architecture/kernel_aarch64.md`](docs/architecture/kernel_aarch64.md)
- [`docs/architecture/kernel_cuda.md`](docs/architecture/kernel_cuda.md)
- [`docs/architecture/kernel_metal.md`](docs/architecture/kernel_metal.md)
- [`docs/architecture/kernel_vulkan.md`](docs/architecture/kernel_vulkan.md)
- [`docs/architecture/kernel_wasm.md`](docs/architecture/kernel_wasm.md)
- [`docs/architecture/kernel_x86_64.md`](docs/architecture/kernel_x86_64.md)
- [`docs/architecture/logits_sampler.md`](docs/architecture/logits_sampler.md)
- [`docs/architecture/logits_validator.md`](docs/architecture/logits_validator.md)
- [`docs/architecture/memory_hybrid.md`](docs/architecture/memory_hybrid.md)
- [`docs/architecture/memory_kv.md`](docs/architecture/memory_kv.md)
- [`docs/architecture/memory_recurrent.md`](docs/architecture/memory_recurrent.md)
- [`docs/architecture/model_builder.md`](docs/architecture/model_builder.md)
- [`docs/architecture/model_loader.md`](docs/architecture/model_loader.md)
- [`docs/architecture/model_tensor.md`](docs/architecture/model_tensor.md)
- [`docs/architecture/model_weight_loader.md`](docs/architecture/model_weight_loader.md)
- [`docs/architecture/tensor_view.md`](docs/architecture/tensor_view.md)
- [`docs/architecture/text_conditioner.md`](docs/architecture/text_conditioner.md)
- [`docs/architecture/text_detokenizer.md`](docs/architecture/text_detokenizer.md)
- [`docs/architecture/text_encoders_bpe.md`](docs/architecture/text_encoders_bpe.md)
- [`docs/architecture/text_encoders_fallback.md`](docs/architecture/text_encoders_fallback.md)
- [`docs/architecture/text_encoders_plamo2.md`](docs/architecture/text_encoders_plamo2.md)
- [`docs/architecture/text_encoders_rwkv.md`](docs/architecture/text_encoders_rwkv.md)
- [`docs/architecture/text_encoders_spm.md`](docs/architecture/text_encoders_spm.md)
- [`docs/architecture/text_encoders_ugm.md`](docs/architecture/text_encoders_ugm.md)
- [`docs/architecture/text_encoders_wpm.md`](docs/architecture/text_encoders_wpm.md)
- [`docs/architecture/text_formatter.md`](docs/architecture/text_formatter.md)
- [`docs/architecture/text_jinja_formatter.md`](docs/architecture/text_jinja_formatter.md)
- [`docs/architecture/text_jinja_parser_classifier_parser.md`](docs/architecture/text_jinja_parser_classifier_parser.md)
- [`docs/architecture/text_jinja_parser_lexer.md`](docs/architecture/text_jinja_parser_lexer.md)
- [`docs/architecture/text_jinja_parser_program_parser_expression_parser.md`](docs/architecture/text_jinja_parser_program_parser_expression_parser.md)
- [`docs/architecture/text_jinja_parser_program_parser.md`](docs/architecture/text_jinja_parser_program_parser.md)
- [`docs/architecture/text_jinja_parser_program_parser_statement_parser.md`](docs/architecture/text_jinja_parser_program_parser_statement_parser.md)
- [`docs/architecture/text_jinja_parser.md`](docs/architecture/text_jinja_parser.md)
- [`docs/architecture/text_renderer.md`](docs/architecture/text_renderer.md)
- [`docs/architecture/text_tokenizer_preprocessor_bpe.md`](docs/architecture/text_tokenizer_preprocessor_bpe.md)
- [`docs/architecture/text_tokenizer_preprocessor_fallback.md`](docs/architecture/text_tokenizer_preprocessor_fallback.md)
- [`docs/architecture/text_tokenizer_preprocessor_plamo2.md`](docs/architecture/text_tokenizer_preprocessor_plamo2.md)
- [`docs/architecture/text_tokenizer_preprocessor_rwkv.md`](docs/architecture/text_tokenizer_preprocessor_rwkv.md)
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

## Embedded size

Generated from `scripts/embedded_size.sh --snapshot-update` using final dead-stripped Qwen3 E2E runner executables.

- Snapshot: `snapshots/embedded_size/summary.txt`
- Mode: `linked_executable`
- Scope: `e2e_inference`
- Workload: `qwen3_0_6b_prompt_hello_max_tokens_1`
- Backend: `aarch64`
- Reference ref: `8710e5f9b9bd7246608808ccd3626bde8abf6ff9`
- Toolchain: `/opt/homebrew/bin/zig`
- Build type: `MinSizeRel`
- Compile flags: `-ffunction-sections,-fdata-sections`
- Link flags: `-Wl,-dead_strip`
- Model fixture: `tests/models/Qwen3-0.6B-Q8_0.gguf`
- Prompt: `hello`
- Max tokens: `1`
- Runtime smoke: `passed`

| Executable | raw bytes | stripped bytes | section bytes |
| --- | ---: | ---: | ---: |
| `emel` | 4073016 | 4073016 | 1323877 |
| `llama.cpp/ggml reference` | 3334264 | 2795112 | 3094255 |

- Ratio (`emel / reference`) raw: `1.222x`
- Ratio (`emel / reference`) stripped: `1.457x`
- Ratio (`emel / reference`) section: `0.428x`

This is a matched Qwen3-0.6B end-to-end runner size measurement for the maintained `hello` -> first-token path, not a whole-product feature-parity claim. Both binaries still include the platform runtime selected by the toolchain.