# Phase 07.1 Deferred Items

## Wave 2 Blocker

Execution stopped after `07.1-01`.

The current repo does not contain the native typed compute substrate that `07.1-02` assumed could
be wired into the existing generator callbacks:

- no native forward-pass / decode implementation for transformer or Llama-like models in `src/`
- no concrete `model_topology`, `prefill_plan`, or `decode_plan` objects or builders in `src/`
- no loader-side mapping from `model::data::tensor_record` names to stable named layer views for
  forward compute

Relevant evidence:
- [events.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/generator/events.hpp)
- [actions.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/generator/actions.hpp)
- [events.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/graph/processor/events.hpp)
- [data.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/data.hpp)
- [parity_runner.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/parity_runner.cpp)
- [generation_bench.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/bench/generation_bench.cpp)

Smallest missing substrate:

1. a loader-side typed mapping from flat tensor records to stable named Llama weight views
2. concrete canonical `model_topology`, `prefill_plan`, and `decode_plan` objects built from those
   views
3. native validate/prepare/bind/run/extract implementations that target those typed objects through
   the existing generator/graph orchestration

Until that substrate exists, replacing `llama_decode` in paritychecker or bench is not a simple
callback swap.
