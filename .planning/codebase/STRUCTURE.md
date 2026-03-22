# Structure Map

## Top-Level Layout

The repository is organized around source code in `src/`, public headers in `include/`, tests in
`tests/`, generated architecture documentation in `docs/architecture/`, automation in `scripts/`,
and snapshots / artifacts in `snapshots/`.

Important top-level files:

- `CMakeLists.txt` and `CMakePresets.json` define the build and test graph
- `README.md` gives the project-level intent and build/test entry points
- `AGENTS.md` and `docs/rules/sml.rules.md` define the local engineering and SML contracts
- `.planning/codebase/` is the mapping output area used by this repository workflow

## Source Tree

The main code lives under `src/emel/`. The directory layout maps closely to namespaces.

### Infrastructure and shared abstractions

- `src/emel/sm.hpp`: common SML wrapper, normalized dispatch, `sm_any`
- `src/emel/machines.hpp`: top-level C++ aliases such as `emel::Generator`, `emel::ModelLoader`,
  `emel::Tokenizer`, `emel::Renderer`, and `emel::ComputeExecutor`
- `src/emel/docs/detail.hpp`: docs-generation support for turning SML types into readable names

### Model and file loading

- `src/emel/model/data.hpp` and `src/emel/model/data.cpp`: model data structures and storage
- `src/emel/model/loader/`: orchestration for loading and validating model state
- `src/emel/model/weight_loader/`: bind / plan / apply flow for weight loading
- `src/emel/gguf/loader/`: GGUF probe, bind, and parse actor

### Text stack

- `src/emel/text/formatter/`: generic prompt-formatting machine plus `format.hpp`
- `src/emel/text/jinja/`: Jinja-oriented parser, formatter, lexer support, and shared `types.hpp`
- `src/emel/text/conditioner/`: formatter + tokenizer orchestration
- `src/emel/text/tokenizer/`: main tokenizer orchestration
- `src/emel/text/tokenizer/preprocessor/`: tokenizer-family-specific preprocessors plus `any.hpp`
- `src/emel/text/encoders/`: tokenizer-family-specific encoders plus `any.hpp`
- `src/emel/text/renderer/`: token-to-text rendering
- `src/emel/text/detokenizer/`: detokenization
- `src/emel/text/unicode.hpp` and `src/emel/text/unicode_data.hpp`: shared Unicode utilities/data

### Grammar and constrained decoding

- `src/emel/gbnf/rule_parser/`: grammar parser and its nested parser actors
- `src/emel/gbnf/sampler/`: grammar-based candidate filtering and logits sampler adaptation
- `src/emel/gbnf/detail.hpp`: shared grammar utilities

### Graph execution stack

- `src/emel/graph/`: top-level graph reserve / compute façade
- `src/emel/graph/assembler/`: graph assembly orchestration
- `src/emel/graph/assembler/*_pass/`: reserve and assemble pass submachines
- `src/emel/graph/allocator/`: graph memory allocation orchestration
- `src/emel/graph/allocator/*_pass/`: allocator pass submachines
- `src/emel/graph/processor/`: compute execution orchestration
- `src/emel/graph/processor/*_step/`: validate, prepare, alloc, bind, kernel, and extract submachines

### Runtime inference orchestration

- `src/emel/generator/`: highest-level initialize / generate actor
- `src/emel/batch/planner/`: batch planning and mode selection
- `src/emel/batch/planner/modes/`: `simple`, `equal`, and `sequential` planning modes
- `src/emel/token/batcher/`: low-level token batch normalization and output-mask handling
- `src/emel/memory/kv/`, `src/emel/memory/recurrent/`, `src/emel/memory/hybrid/`: memory-state actors
- `src/emel/memory/view.hpp`: cross-memory snapshot representation
- `src/emel/logits/validator/` and `src/emel/logits/sampler/`: logits validation and sampling

### Tensor and kernel layers

- `src/emel/tensor/` and `src/emel/tensor/view/`: tensor and tensor-view orchestration layers
- `src/emel/kernel/`: backend-dispatch façade and shared kernel event definitions
- `src/emel/kernel/x86_64/`, `src/emel/kernel/aarch64/`, `src/emel/kernel/wasm/`,
  `src/emel/kernel/cuda/`, `src/emel/kernel/metal/`, `src/emel/kernel/vulkan/`: backend-specific machines

## Public Include Tree

The public headers under `include/emel/` are intentionally small:

- `include/emel/emel.h`: current C ABI root
- `include/emel/error/error.hpp`: shared error bitfield helpers
- `include/emel/callback.hpp`: callback wrapper used across actor boundaries

Most operational APIs are still internal C++ headers under `src/emel/`.

## File Naming Patterns

The repository uses a strong per-component naming convention. Common files in a component directory:

- `sm.hpp`: machine definition and wrapper
- `events.hpp`: public requests, runtime events, done/error payloads
- `context.hpp`: persistent actor-owned state
- `actions.hpp`: action functions
- `guards.hpp`: guard predicates
- `errors.hpp`: local error definitions
- `detail.hpp`: low-level helpers when needed

Common exceptions:

- `any.hpp` for variant selectors, such as `src/emel/kernel/any.hpp`,
  `src/emel/text/encoders/any.hpp`, and `src/emel/text/tokenizer/preprocessor/any.hpp`
- `types.hpp` for shared payload shapes, such as `src/emel/text/jinja/types.hpp`
- `format.hpp` for pure formatter function boundaries in `src/emel/text/formatter/format.hpp`
- `view.hpp` for persistent snapshot / view types such as `src/emel/memory/view.hpp`
- a small number of `.cpp` translation units, currently led by `src/emel/model/data.cpp`

Submachine families use descriptive suffixes:

- `*_pass` in `src/emel/graph/assembler/` and `src/emel/graph/allocator/`
- `*_step` in `src/emel/graph/processor/`
- tokenizer or encoder family names in `src/emel/text/tokenizer/preprocessor/` and
  `src/emel/text/encoders/`

## Namespace and Responsibility Map

Representative namespace-to-directory mapping:

- `emel::generator` -> `src/emel/generator/`
- `emel::graph::processor` -> `src/emel/graph/processor/`
- `emel::model::loader` -> `src/emel/model/loader/`
- `emel::gguf::loader` -> `src/emel/gguf/loader/`
- `emel::text::conditioner` -> `src/emel/text/conditioner/`
- `emel::text::tokenizer::preprocessor` -> `src/emel/text/tokenizer/preprocessor/`
- `emel::text::encoders` -> `src/emel/text/encoders/`
- `emel::memory::hybrid` -> `src/emel/memory/hybrid/`
- `emel::batch::planner` -> `src/emel/batch/planner/`
- `emel::gbnf::rule_parser` -> `src/emel/gbnf/rule_parser/`

`src/emel/machines.hpp` is the main place where internal namespace depth is surfaced as simpler
public C++ aliases.

## Tests and Verification Layout

Tests mirror subsystem structure under `tests/`:

- `tests/model/loader/` and `tests/model/weight_loader/`
- `tests/gguf/loader/`
- `tests/gbnf/`
- `tests/generator/`
- `tests/graph/allocator/`, `tests/graph/assembler/`, `tests/graph/processor/`, `tests/graph/`
- `tests/batch/planner/`
- `tests/memory/kv/`, `tests/memory/recurrent/`, `tests/memory/hybrid/`
- `tests/text/conditioner/`, `tests/text/detokenizer/`, `tests/text/encoders/`,
  `tests/text/formatter/`, `tests/text/jinja/`, `tests/text/renderer/`, `tests/text/tokenizer/`,
  `tests/text/unicode/`
- `tests/kernel/`, `tests/logits/`, `tests/tensor/`, `tests/token/batcher/`, `tests/sm/`

`tests/doctest_main.cpp` is the shared test runner. Fuzz targets live in `tests/fuzz/`. Test
models and fixtures live in `tests/models/` and `tests/gbnf/parity_texts/`.

## Docs, Scripts, and Snapshots

Generated and authored docs are separated:

- `docs/architecture/`: generated state-machine docs
- `docs/architecture/mermaid/`: generated Mermaid diagrams
- `docs/rules/`: coding and SML rules
- `docs/notes/`, `docs/plans/`, and `docs/templates/`: supporting process material

Operational scripts live in `scripts/`:

- `scripts/quality_gates.sh`
- `scripts/build_with_zig.sh`
- `scripts/test_with_coverage.sh`
- `scripts/test_with_sanitizers.sh`
- `scripts/fuzz_smoke.sh`
- `scripts/lint_snapshot.sh`
- `scripts/bench.sh`
- `scripts/generate_docs.sh`

Frozen outputs live in `snapshots/`, notably `snapshots/lint/`, `snapshots/bench/`, and
`snapshots/quality_gates/`.

## Practical Orientation

For architectural onboarding, the most useful starting directories are:

- `src/emel/generator/` for end-to-end runtime orchestration
- `src/emel/model/loader/` and `src/emel/gguf/loader/` for model ingestion
- `src/emel/graph/` plus its `assembler/` and `processor/` children for compute flow
- `src/emel/text/` for prompt-to-token and token-to-text handling
- `tests/` for subsystem-level executable examples that mirror the same directory structure
