# EMEL

CPU-first C++ inference engine for deterministic, high-performance inference built around
Stateforward.SML orchestration.

## Status: WIP

This repository is under active development. APIs, state machines, and formats will change.
If you’re evaluating EMEL, expect fast iteration and breaking changes until the core loader,
allocator, and execution pipelines stabilize.

This inference engine is being implemented by AI under human engineering and architecture direction.

## CPU-first scope

EMEL is a CPU-first inference engine. The goal is deterministic behavior with high-performance
execution on maintained CPU paths. Runtime claims are made only for source-backed CPU slices with
EMEL-owned kernels, lane-isolated parity proof, and benchmark evidence. Backend namespaces may
exist as historical scaffolding, generated architecture inventory, or future device placeholders,
but they are not supported runtime claims unless a milestone explicitly backs them with maintained
tests, parity, and benchmark evidence.

## Implementation priorities

1. Keep the maintained end-to-end path CPU-first, deterministic, and benchmark-backed.
2. Architect first, then scaffold cleanly.
3. Port math, instructions, and behavior without mirroring reference control flow.
4. Prove parity against isolated reference lanes.
5. Match model/tokenizer intent as defined by their creators.
6. Optimize once correctness is locked.

## Why EMEL

EMEL exists to make inference behavior explicit and verifiable. Instead of ad-hoc control flow,
orchestration is modeled as Stateforward.SML state machines with deterministic, testable transitions.
That enables:

1. Clear operational semantics and failure modes.
2. Deterministic, reproducible inference paths.
3. High-performance CPU execution and C-compatible boundaries without dynamic dispatch in hot paths.
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
profiling shows a state machine itself is the bottleneck, it gets removed. Parallelism is introduced
only where deterministic behavior is preserved and measurement says it is necessary.

## I/O and tensor ownership

`model/tensor` is the canonical owner of tensor bind, load, evict, and residency lifecycle
semantics. `emel/io` owns loading strategy, transport, mapping, staging, and
device/resource-specific strategy boundaries without owning residency. `model/loader` stays
orchestration-only: it can dispatch the tensor and I/O actors, but it must not regain low-level
file APIs or a shadow tensor residency lifecycle.

The mmap strategy actor is implemented under `src/emel/io/mmap` and is the maintained
loading path for tensor-backed mmap-resident loads. `model/tensor` requests mmap-backed
loading via public `request_mapped_load` / `release_mapped_load` events; `emel/io/mmap`
owns mapping, slot reservation, and lifetime contracts. The read/copy strategy actor is
implemented under `src/emel/io/read`; `model/tensor` requests read-backed loading through
public `request_read_load` events, retains tensor residency ownership, and commits the
caller-owned target buffer only after the maintained read/copy path succeeds. The staged
constrained-memory strategy actor is implemented under `src/emel/io/staged_read` and is
routed through public `io::loader` strategy selection when staged source-span contracts are
provided; async and device-specific loading strategies remain follow-on work.

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

- [Architecture](.planning/architecture/) (generated state-machine docs + Mermaid diagrams)
- [CPU/backend Inventory](docs/arm-first-inventory.md) (backend surface inventory and removal notes)
- [Benchmarks](docs/benchmarks.md) (generated benchmark snapshot table)
- [SML Conventions](docs/third_party/sml.md) (Stateforward.SML conventions and usage)

## Docs index

- [`docs/benchmarks.md`](docs/benchmarks.md)
- [`.planning/architecture/batch_planner_modes_equal.md`](.planning/architecture/batch_planner_modes_equal.md)
- [`.planning/architecture/batch_planner_modes_sequential.md`](.planning/architecture/batch_planner_modes_sequential.md)
- [`.planning/architecture/batch_planner_modes_simple.md`](.planning/architecture/batch_planner_modes_simple.md)
- [`.planning/architecture/batch_planner.md`](.planning/architecture/batch_planner.md)
- [`.planning/architecture/diarization_sortformer_executor.md`](.planning/architecture/diarization_sortformer_executor.md)
- [`.planning/architecture/diarization_sortformer_pipeline.md`](.planning/architecture/diarization_sortformer_pipeline.md)
- [`.planning/architecture/diarization_sortformer_request.md`](.planning/architecture/diarization_sortformer_request.md)
- [`.planning/architecture/embeddings_generator.md`](.planning/architecture/embeddings_generator.md)
- [`.planning/architecture/gbnf_rule_parser_definition_parser.md`](.planning/architecture/gbnf_rule_parser_definition_parser.md)
- [`.planning/architecture/gbnf_rule_parser_expression_parser.md`](.planning/architecture/gbnf_rule_parser_expression_parser.md)
- [`.planning/architecture/gbnf_rule_parser_lexer.md`](.planning/architecture/gbnf_rule_parser_lexer.md)
- [`.planning/architecture/gbnf_rule_parser_nonterm_parser.md`](.planning/architecture/gbnf_rule_parser_nonterm_parser.md)
- [`.planning/architecture/gbnf_rule_parser.md`](.planning/architecture/gbnf_rule_parser.md)
- [`.planning/architecture/gbnf_rule_parser_term_parser.md`](.planning/architecture/gbnf_rule_parser_term_parser.md)
- [`.planning/architecture/gbnf_sampler_accept_parser.md`](.planning/architecture/gbnf_sampler_accept_parser.md)
- [`.planning/architecture/gbnf_sampler_candidate_parser.md`](.planning/architecture/gbnf_sampler_candidate_parser.md)
- [`.planning/architecture/gbnf_sampler_matcher_parser.md`](.planning/architecture/gbnf_sampler_matcher_parser.md)
- [`.planning/architecture/gbnf_sampler.md`](.planning/architecture/gbnf_sampler.md)
- [`.planning/architecture/gbnf_sampler_token_parser.md`](.planning/architecture/gbnf_sampler_token_parser.md)
- [`.planning/architecture/gguf_loader.md`](.planning/architecture/gguf_loader.md)
- [`.planning/architecture/graph_allocator_liveness_pass.md`](.planning/architecture/graph_allocator_liveness_pass.md)
- [`.planning/architecture/graph_allocator_ordering_pass.md`](.planning/architecture/graph_allocator_ordering_pass.md)
- [`.planning/architecture/graph_allocator_placement_pass.md`](.planning/architecture/graph_allocator_placement_pass.md)
- [`.planning/architecture/graph_allocator.md`](.planning/architecture/graph_allocator.md)
- [`.planning/architecture/graph_assembler_assemble_alloc_pass.md`](.planning/architecture/graph_assembler_assemble_alloc_pass.md)
- [`.planning/architecture/graph_assembler_assemble_build_pass.md`](.planning/architecture/graph_assembler_assemble_build_pass.md)
- [`.planning/architecture/graph_assembler_assemble_validate_pass.md`](.planning/architecture/graph_assembler_assemble_validate_pass.md)
- [`.planning/architecture/graph_assembler_reserve_alloc_pass.md`](.planning/architecture/graph_assembler_reserve_alloc_pass.md)
- [`.planning/architecture/graph_assembler_reserve_build_pass.md`](.planning/architecture/graph_assembler_reserve_build_pass.md)
- [`.planning/architecture/graph_assembler_reserve_validate_pass.md`](.planning/architecture/graph_assembler_reserve_validate_pass.md)
- [`.planning/architecture/graph_assembler_reuse_decision_pass.md`](.planning/architecture/graph_assembler_reuse_decision_pass.md)
- [`.planning/architecture/graph_assembler.md`](.planning/architecture/graph_assembler.md)
- [`.planning/architecture/graph_processor_alloc_step.md`](.planning/architecture/graph_processor_alloc_step.md)
- [`.planning/architecture/graph_processor_bind_step.md`](.planning/architecture/graph_processor_bind_step.md)
- [`.planning/architecture/graph_processor_extract_step.md`](.planning/architecture/graph_processor_extract_step.md)
- [`.planning/architecture/graph_processor_kernel_step.md`](.planning/architecture/graph_processor_kernel_step.md)
- [`.planning/architecture/graph_processor_prepare_step.md`](.planning/architecture/graph_processor_prepare_step.md)
- [`.planning/architecture/graph_processor.md`](.planning/architecture/graph_processor.md)
- [`.planning/architecture/graph_processor_validate_step.md`](.planning/architecture/graph_processor_validate_step.md)
- [`.planning/architecture/graph.md`](.planning/architecture/graph.md)
- [`.planning/architecture/graph_tensor.md`](.planning/architecture/graph_tensor.md)
- [`.planning/architecture/io_loader.md`](.planning/architecture/io_loader.md)
- [`.planning/architecture/io_mmap.md`](.planning/architecture/io_mmap.md)
- [`.planning/architecture/io_read.md`](.planning/architecture/io_read.md)
- [`.planning/architecture/io_staged_read.md`](.planning/architecture/io_staged_read.md)
- [`.planning/architecture/kernel_aarch64.md`](.planning/architecture/kernel_aarch64.md)
- [`.planning/architecture/kernel_x86_64.md`](.planning/architecture/kernel_x86_64.md)
- [`.planning/architecture/logits_sampler.md`](.planning/architecture/logits_sampler.md)
- [`.planning/architecture/logits_validator.md`](.planning/architecture/logits_validator.md)
- [`.planning/architecture/memory_hybrid.md`](.planning/architecture/memory_hybrid.md)
- [`.planning/architecture/memory_kv.md`](.planning/architecture/memory_kv.md)
- [`.planning/architecture/memory_recurrent.md`](.planning/architecture/memory_recurrent.md)
- [`.planning/architecture/memory_streaming.md`](.planning/architecture/memory_streaming.md)
- [`.planning/architecture/model_loader.md`](.planning/architecture/model_loader.md)
- [`.planning/architecture/model_tensor.md`](.planning/architecture/model_tensor.md)
- [`.planning/architecture/model_tensor_window.md`](.planning/architecture/model_tensor_window.md)
- [`.planning/architecture/speech_codec_mimi_decoder.md`](.planning/architecture/speech_codec_mimi_decoder.md)
- [`.planning/architecture/speech_codec_mimi_encoder.md`](.planning/architecture/speech_codec_mimi_encoder.md)
- [`.planning/architecture/speech_codec_mimi_quantizer.md`](.planning/architecture/speech_codec_mimi_quantizer.md)
- [`.planning/architecture/speech_codec_mimi.md`](.planning/architecture/speech_codec_mimi.md)
- [`.planning/architecture/speech_decoder_whisper.md`](.planning/architecture/speech_decoder_whisper.md)
- [`.planning/architecture/speech_encoder_whisper.md`](.planning/architecture/speech_encoder_whisper.md)
- [`.planning/architecture/speech_predictor_moshi_executor.md`](.planning/architecture/speech_predictor_moshi_executor.md)
- [`.planning/architecture/speech_predictor_moshi.md`](.planning/architecture/speech_predictor_moshi.md)
- [`.planning/architecture/speech_tokenizer_moshi.md`](.planning/architecture/speech_tokenizer_moshi.md)
- [`.planning/architecture/tensor_view.md`](.planning/architecture/tensor_view.md)
- [`.planning/architecture/text_conditioner.md`](.planning/architecture/text_conditioner.md)
- [`.planning/architecture/text_detokenizer.md`](.planning/architecture/text_detokenizer.md)
- [`.planning/architecture/text_encoders_bpe.md`](.planning/architecture/text_encoders_bpe.md)
- [`.planning/architecture/text_encoders_fallback.md`](.planning/architecture/text_encoders_fallback.md)
- [`.planning/architecture/text_encoders_plamo2.md`](.planning/architecture/text_encoders_plamo2.md)
- [`.planning/architecture/text_encoders_rwkv.md`](.planning/architecture/text_encoders_rwkv.md)
- [`.planning/architecture/text_encoders_spm.md`](.planning/architecture/text_encoders_spm.md)
- [`.planning/architecture/text_encoders_ugm.md`](.planning/architecture/text_encoders_ugm.md)
- [`.planning/architecture/text_encoders_wpm.md`](.planning/architecture/text_encoders_wpm.md)
- [`.planning/architecture/text_formatter.md`](.planning/architecture/text_formatter.md)
- [`.planning/architecture/text_generator_decode_wavefront.md`](.planning/architecture/text_generator_decode_wavefront.md)
- [`.planning/architecture/text_generator_initializer.md`](.planning/architecture/text_generator_initializer.md)
- [`.planning/architecture/text_generator_matmul.md`](.planning/architecture/text_generator_matmul.md)
- [`.planning/architecture/text_generator_prefill.md`](.planning/architecture/text_generator_prefill.md)
- [`.planning/architecture/text_generator.md`](.planning/architecture/text_generator.md)
- [`.planning/architecture/text_jinja_formatter.md`](.planning/architecture/text_jinja_formatter.md)
- [`.planning/architecture/text_jinja_parser_classifier_parser.md`](.planning/architecture/text_jinja_parser_classifier_parser.md)
- [`.planning/architecture/text_jinja_parser_lexer.md`](.planning/architecture/text_jinja_parser_lexer.md)
- [`.planning/architecture/text_jinja_parser_program_parser_expression_parser.md`](.planning/architecture/text_jinja_parser_program_parser_expression_parser.md)
- [`.planning/architecture/text_jinja_parser_program_parser.md`](.planning/architecture/text_jinja_parser_program_parser.md)
- [`.planning/architecture/text_jinja_parser_program_parser_statement_parser.md`](.planning/architecture/text_jinja_parser_program_parser_statement_parser.md)
- [`.planning/architecture/text_jinja_parser.md`](.planning/architecture/text_jinja_parser.md)
- [`.planning/architecture/text_renderer.md`](.planning/architecture/text_renderer.md)
- [`.planning/architecture/text_tokenizer_preprocessor_bpe.md`](.planning/architecture/text_tokenizer_preprocessor_bpe.md)
- [`.planning/architecture/text_tokenizer_preprocessor_fallback.md`](.planning/architecture/text_tokenizer_preprocessor_fallback.md)
- [`.planning/architecture/text_tokenizer_preprocessor_plamo2.md`](.planning/architecture/text_tokenizer_preprocessor_plamo2.md)
- [`.planning/architecture/text_tokenizer_preprocessor_rwkv.md`](.planning/architecture/text_tokenizer_preprocessor_rwkv.md)
- [`.planning/architecture/text_tokenizer_preprocessor_spm.md`](.planning/architecture/text_tokenizer_preprocessor_spm.md)
- [`.planning/architecture/text_tokenizer_preprocessor_ugm.md`](.planning/architecture/text_tokenizer_preprocessor_ugm.md)
- [`.planning/architecture/text_tokenizer_preprocessor_wpm.md`](.planning/architecture/text_tokenizer_preprocessor_wpm.md)
- [`.planning/architecture/text_tokenizer.md`](.planning/architecture/text_tokenizer.md)
- [`.planning/architecture/token_batcher.md`](.planning/architecture/token_batcher.md)

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
