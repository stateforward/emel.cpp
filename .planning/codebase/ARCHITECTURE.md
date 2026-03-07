# Architecture Map

## Overview

`emel.cpp` is a deterministic C++ inference engine whose orchestration layer is modeled as Boost.SML
actors rather than ad hoc control flow. The project-level statement is in `README.md`, the core
SML contract is in `docs/rules/sml.rules.md`, and the local wrapper that standardizes machine
construction and dispatch lives in `src/emel/sm.hpp`.

The build graph in `CMakeLists.txt` shows the current packaging shape:

- `emel_core` is an interface target exporting `include/`, `src/`, and the vendored Boost.SML
  headers.
- `emel` is a static library that currently compiles only `src/emel/model/data.cpp`; almost all
  orchestration and behavior lives in headers under `src/emel/`.
- the public C ABI surface is still minimal in `include/emel/emel.h`, while the richer C++
  machine aliases live in `src/emel/machines.hpp`.

## High-Level Patterns

### 1. Actor-style SML orchestration

Every major runtime component is expressed as a `struct model` transition table plus a small wrapper
`struct sm : public emel::sm<model, action::context>` in files such as `src/emel/generator/sm.hpp`,
`src/emel/graph/sm.hpp`, `src/emel/model/loader/sm.hpp`, and `src/emel/text/tokenizer/sm.hpp`.

The shared wrapper in `src/emel/sm.hpp` provides:

- context injection into `boost::sml::sm<...>`
- normalized `process_event(...)` return handling
- state inspection through `is(...)` and `visit_current_states(...)`
- `sm_any`, a zero-virtual-dispatch variant holder used by `src/emel/kernel/any.hpp`,
  `src/emel/text/encoders/any.hpp`, and `src/emel/text/tokenizer/preprocessor/any.hpp`

The rule set in `docs/rules/sml.rules.md` is visible directly in the machines:

- external requests are wrapped into internal runtime events with per-dispatch context objects, for
  example `event::generate_run` in `src/emel/generator/events.hpp` and `event::load_runtime` in
  `src/emel/model/loader/events.hpp`
- internal progress uses `sml::completion<TEvent>` chains instead of self-dispatch
- unexpected external input is handled explicitly with `sml::unexpected_event<sml::_>`

### 2. Separation between orchestration and kernels

Control flow is primarily in `sm.hpp`, `guards.hpp`, and `actions.hpp`. Data-plane work and
helpers are pushed into `detail.hpp` or function-pointer boundaries. Examples:

- `src/emel/text/formatter/format.hpp` exposes `format_fn`
- `src/emel/graph/events.hpp` defines backend hooks such as `validate_fn`, `prepare_graph_fn`,
  `alloc_graph_fn`, `bind_inputs_fn`, `run_kernel_fn`, and `extract_outputs_fn`
- `include/emel/callback.hpp` provides the synchronous callback abstraction used for immediate
  request-reply publication

### 3. Mostly header-defined architecture

The codebase is intentionally source-of-truth-in-headers. Generated architecture documents under
`docs/architecture/` are derived from the state-machine headers via `scripts/generate_docs.sh` and
the docs tooling in `src/emel/docs/detail.hpp`.

## Main Subsystems

### Model ingestion

Model ingestion is split across four layers:

- `src/emel/gguf/loader/sm.hpp` probes, binds, and parses GGUF storage
- `src/emel/model/loader/sm.hpp` orchestrates parse, optional weight loading, layer mapping, and
  structural / architectural validation
- `src/emel/model/weight_loader/sm.hpp` manages bind, plan, and apply phases for weight materialization
- `src/emel/model/data.hpp` and `src/emel/model/data.cpp` hold the large fixed-capacity in-memory
  model representation, including vocabulary, tensor records, and hyperparameters

This path is callback-driven at the edges. `src/emel/model/loader/events.hpp` shows the abstract
operations that the loader expects: `parse_model`, `load_weights`, `map_layers`,
`validate_structure`, and `validate_architecture_impl`.

### Text processing

The text stack is layered rather than monolithic:

- `src/emel/text/formatter/sm.hpp` and `src/emel/text/formatter/format.hpp` handle prompt formatting
- `src/emel/text/jinja/parser/sm.hpp` and `src/emel/text/jinja/formatter/sm.hpp` implement template parsing
  and rendering support
- `src/emel/text/conditioner/sm.hpp` binds formatting plus tokenization into a prompt-preparation actor
- `src/emel/text/tokenizer/sm.hpp` orchestrates preprocessing, BOS/EOS handling, raw-fragment encoding,
  and final token publication
- `src/emel/text/tokenizer/preprocessor/*/sm.hpp` provides tokenizer-family-specific preprocessing
- `src/emel/text/encoders/*/sm.hpp` provides tokenizer-family-specific encoding
- `src/emel/text/renderer/sm.hpp` and `src/emel/text/detokenizer/sm.hpp` cover the reverse path from
  selected tokens back to output text

Variant selection is explicit:

- preprocessors are selected through `src/emel/text/tokenizer/preprocessor/any.hpp`
- encoders are selected through `src/emel/text/encoders/any.hpp`
- tokenizer and conditioner contexts carry function pointers to dispatch into those selected actors,
  visible in `src/emel/text/tokenizer/events.hpp` and `src/emel/text/conditioner/context.hpp`

### Grammar-constrained sampling

The `gbnf` family provides grammar parsing and candidate filtering:

- `src/emel/gbnf/rule_parser/sm.hpp` is a higher-level rule parser that composes
  `definition_parser`, `expression_parser`, `nonterm_parser`, `term_parser`, and `lexer`
- `src/emel/gbnf/sampler/sm.hpp` turns a parsed grammar into a sampler compatible with the logits
  pipeline via `as_logits_sampler_fn()`

This architecture makes constrained decoding a plug-in to the regular sampling pipeline rather than
a separate generation stack.

### Graph construction and execution

The compute path is split into orchestration stages:

- `src/emel/graph/sm.hpp` is the façade that accepts reserve / compute requests
- `src/emel/graph/assembler/sm.hpp` handles graph reservation and assembly
- `src/emel/graph/allocator/sm.hpp` and its pass subdirectories manage graph memory planning
- `src/emel/graph/processor/sm.hpp` executes validated graph steps through submachines
  `validate_step`, `prepare_step`, `alloc_step`, `bind_step`, `kernel_step`, and `extract_step`
- `src/emel/kernel/any.hpp` selects the backend implementation among
  `src/emel/kernel/x86_64/sm.hpp`, `src/emel/kernel/aarch64/sm.hpp`,
  `src/emel/kernel/wasm/sm.hpp`, `src/emel/kernel/cuda/sm.hpp`,
  `src/emel/kernel/metal/sm.hpp`, and `src/emel/kernel/vulkan/sm.hpp`

The façade context in `src/emel/graph/context.hpp` owns both `assembler::sm` and `processor::sm`,
so the public graph actor is a composed orchestration shell around two specialized child actors.

### Runtime generation

The highest-level inference session actor is `src/emel/generator/sm.hpp`. Its context in
`src/emel/generator/context.hpp` owns or references:

- `emel::text::conditioner::sm`
- `emel::text::renderer::sm`
- `emel::batch::planner::sm`
- `emel::memory::hybrid::sm`
- `emel::graph::sm`
- `emel::logits::sampler::sm`
- model, formatter, tokenizer, and backend binding data

Its two major flows are:

- initialize flow: bind conditioner, initialize renderer, reserve memory, reserve graph, configure sampler
- generate flow: optionally reset sequence, condition prompt, plan batches, allocate memory, snapshot state,
  run prefill compute, sample, render, loop decode, then flush

Supporting runtime subsystems include:

- `src/emel/batch/planner/sm.hpp` for batch-shape planning
- `src/emel/token/batcher/sm.hpp` for low-level token / position / mask batch formation
- `src/emel/memory/kv/sm.hpp`, `src/emel/memory/recurrent/sm.hpp`, and
  `src/emel/memory/hybrid/sm.hpp` for sequence state management
- `src/emel/logits/validator/sm.hpp` and `src/emel/logits/sampler/sm.hpp` for output validation
  and token selection

## State-Machine Organization

The standard component layout is visible in directories like `src/emel/graph/processor/`,
`src/emel/text/conditioner/`, and `src/emel/model/loader/`:

- `sm.hpp` contains the transition table and wrapper
- `events.hpp` defines external requests, internal runtime wrappers, and done/error payloads
- `context.hpp` holds persistent actor-owned state
- `guards.hpp` contains pure branch predicates
- `actions.hpp` contains phase kernels and publication actions
- `errors.hpp` provides local error enums / conversions
- `detail.hpp` is used when a machine needs branch-free helpers or storage utilities

Nested orchestration is done with submachine states rather than direct action calls. Examples:

- `src/emel/graph/processor/sm.hpp` enters `validate_step::model`, `prepare_step::model`,
  `alloc_step::model`, `bind_step::model`, `kernel_step::model`, and `extract_step::model`
- `src/emel/graph/assembler/sm.hpp` enters `reserve_validate_pass::model`,
  `reserve_build_pass::model`, `reserve_alloc_pass::model`, `assemble_validate_pass::model`,
  `reuse_decision_pass::model`, `assemble_build_pass::model`, and `assemble_alloc_pass::model`
- `src/emel/batch/planner/sm.hpp` enters `modes::simple::model`, `modes::equal::model`, and
  `modes::sequential::model`

## Data and Control Flow

### Model load path

The dominant model-load path is:

`src/emel/gguf/loader/sm.hpp` -> `src/emel/model/loader/sm.hpp` ->
`src/emel/model/weight_loader/sm.hpp` -> `src/emel/model/data.hpp`

The data object is passed by reference through events, not mirrored across multiple orchestration
layers. `src/emel/model/loader/events.hpp` and `src/emel/gguf/loader/events.hpp` show this request
shape clearly.

### Generation path

The dominant generation path is:

`src/emel/generator/sm.hpp` -> `src/emel/text/conditioner/sm.hpp` ->
`src/emel/text/tokenizer/sm.hpp` / `src/emel/text/formatter/sm.hpp` ->
`src/emel/batch/planner/sm.hpp` -> `src/emel/memory/hybrid/sm.hpp` ->
`src/emel/graph/sm.hpp` -> `src/emel/graph/assembler/sm.hpp` / `src/emel/graph/processor/sm.hpp` ->
`src/emel/kernel/any.hpp` -> `src/emel/logits/sampler/sm.hpp` ->
`src/emel/text/renderer/sm.hpp`

The important architectural property is that these are synchronous same-RTC calls coordinated by
typed events and callbacks, not queued background jobs.

## Entry Points and Important Abstractions

- Build entry point: `CMakeLists.txt`
- Public C headers: `include/emel/emel.h`, `include/emel/error/error.hpp`, `include/emel/callback.hpp`
- C++ machine aliases: `src/emel/machines.hpp`
- Core machine wrapper and type-erased variant holder: `src/emel/sm.hpp`
- Docs generation entry point: `scripts/generate_docs.sh`
- Generated machine reference set: `docs/architecture/*.md` and `docs/architecture/mermaid/*.mmd`
- Test harness entry point: `tests/doctest_main.cpp`

## Architectural Takeaways

`emel.cpp` is organized as a graph of small, explicit actors. The project favors compile-time
composition, fixed-capacity data structures, and callback/function-pointer boundaries over virtual
interfaces and opaque runtime orchestration. The practical result is that `src/emel/generator/sm.hpp`,
`src/emel/model/loader/sm.hpp`, `src/emel/graph/sm.hpp`, and `src/emel/text/tokenizer/sm.hpp` are
the best top-level files for understanding end-to-end system behavior.
