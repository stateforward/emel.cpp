# rearchitecture plan (hard cutover, design-final)

status: draft
owner: emel
branch: `rearchitecture`

## execution rules

- [ ] treat `src/emel/**/sm.hpp` design docstrings as the finalized architecture contract.
- [ ] hard cutover only: delete legacy/deprecated machines and tests; no compatibility shims,
      no dual-path dispatch, no legacy adapter layers.
- [ ] clean slate first: if a domain is not yet implemented, scaffold its machines to keep
      build + tests + gates green, but do not retain legacy files for reference.
- [ ] keep scaffolds SML-compliant (unexpected-event handling, bounded RTC, no allocations).
- [ ] update snapshots only with explicit approval.
- [ ] run `scripts/quality_gates.sh` at the end of every phase and record pass/fail notes in PR.
- [ ] if a phase gate fails, stop and fix regressions before starting the next phase.
- [ ] coordinate domain work in parallel: one agent per domain (text, token/batch, memory,
      graph, kernel, generator), each working only within their domain scope.
- [ ] explicitly mark cross-domain dependencies; if a dependency is missing, add a wait point
      instead of coupling implementation across domains.

---

## phase 0: hard cutover baseline (clean slate)

goal: remove legacy artifacts and establish a scaffolded, buildable baseline aligned to designs.

- [ ] delete all legacy/deprecated machines, tests, benches, and dead adapters not present in
      the design docstrings embedded in `src/emel/**/sm.hpp` (no legacy files kept for reference).
- [ ] remove prepare-era coordinator surfaces and legacy public contracts superseded by lifecycle APIs.
- [ ] align directory layout, namespaces, and includes to the design domain structure.
- [ ] introduce scaffolds for any missing machines to keep compilation green.
- [ ] ensure scaffolds include: events + outcomes, context, guards/actions stubs,
      and `sml::unexpected_event` handling per `docs/sml.rules.md`.
- [ ] update CMake/test/bench manifests to match the new layout (no legacy paths).
- [ ] reduce tests to design-aligned contracts only; remove legacy-only expectations.
- [ ] verify no tracked artifacts (`.DS_Store`, `Testing/*`, tmp logs) remain.
- [ ] phase gate: run `scripts/quality_gates.sh`.

exit criteria:
- [ ] no legacy/deprecated machine files, tests, or benches remain in tree.
- [ ] every design-specified machine exists (implemented or scaffolded).
- [ ] build + tests + gates are green on the clean baseline.
- [ ] phase 0 wait point: all domains can compile against each other's headers without
      requiring cross-domain implementation details.

---

## scaffolding rules (applies to all phases)

- [ ] scaffolds must compile and pass tests while enforcing SML rules.
- [ ] scaffold SML machines must:
      - [ ] define explicit states and outcomes (`events::*_done`, `events::*_error`).
      - [ ] handle `sml::unexpected_event<sml::_>` explicitly.
      - [ ] keep actions/guards bounded and allocation-free during dispatch.
- [ ] scaffold tests should validate:
      - [ ] `unexpected_event` routing.
      - [ ] basic success/error transitions.
      - [ ] no legacy API surface usage.

---

## parallelization boundaries + wait points

- [ ] each domain may proceed independently after phase 0 as long as it only consumes
      design-specified event/view types from other domains.
- [ ] if a domain needs a cross-domain type or event that is not yet finalized, add a wait
      point here and in the phase before proceeding.
- [ ] paritychecker is allowed to be disabled while GGUF parsing remains scaffolded.
      wait point: re-enable paritychecker once `parser/gguf` produces vocab and the tokenizer
      parity path is revalidated.

---

## phase 1: text domain (conditioner/tokenizer/encoders/detokenizer/renderer)

goal: implement the text pipeline per design boundaries with clean contracts.

- [ ] finalize `text/conditioner` contract to emit raw token arrays only.
- [ ] keep formatting concerns in `text/formatter` and `text/jinja/*` components.
- [ ] align `text/tokenizer` ownership of preprocessor + encoder dispatch.
- [ ] align `text/encoders::any` variant routing and kind mapping.
- [ ] implement `text/detokenizer` + `text/renderer` contracts for byte buffering and stop matching.
- [ ] add focused tests for conditioner/tokenizer/renderer boundaries.
- [ ] phase gate: run `scripts/quality_gates.sh`.

exit criteria:
- [ ] text pipeline composes cleanly with no ownership ambiguity.
- [ ] tokenizer and renderer boundaries are deterministic and test-covered.
- [ ] quality gates green.
- [ ] wait point before phase 6: `text/conditioner`, `text/tokenizer`, `text/detokenizer`,
      and `text/renderer` public events are frozen and documented.

---

## phase 2: token batching + planning

goal: deterministic token intake and executable `batch::plan` steps.

- [ ] finalize `token/batcher` sanitation and autopopulation rules.
- [ ] align `token/batcher` output contract with `batch/planner` input contract.
- [ ] finalize `batch/planner` step slicing and mapping structure.
- [ ] keep planner stateless and allocation-free during dispatch.
- [ ] add tests for malformed input rejection, continuity rules, and split policy behavior.
- [ ] add bench coverage for hot-path planner scenarios.
- [ ] phase gate: run `scripts/quality_gates.sh`.

exit criteria:
- [ ] `token/batcher -> batch/planner` contract stable and test-verified.
- [ ] planned steps directly consumable by graph processing.
- [ ] quality gates green.
- [ ] wait point before phase 4: `batch::plan` payload and invariants are frozen.

---

## phase 3: memory domain (kv/recurrent/hybrid + coordinator any)

goal: implement memory lifecycle and view semantics per design.

- [ ] align `memory/kv`, `memory/recurrent`, and `memory/hybrid` state/event contracts with designs.
- [ ] finalize `memory/coordinator::any` role and variant selection semantics.
- [ ] verify sequence lifecycle operations (`reserve`, `allocate`, `branch`, `free`, `rollback`).
- [ ] ensure rollback semantics for partial failures in hybrid/coordinator flows.
- [ ] add tests for capacity errors, branching, and lifecycle consistency.
- [ ] add/refresh bench coverage for memory actors and coordinator hot paths.
- [ ] phase gate: run `scripts/quality_gates.sh`.

exit criteria:
- [ ] memory lifecycle semantics are stable under mixed workloads.
- [ ] coordinator and memory actors communicate only via explicit events.
- [ ] quality gates green.
- [ ] wait point before phase 4/6: `memory::view::any` and lifecycle events are frozen.

---

## phase 4: graph domain (graph/assembler/allocator/processor)

goal: explicit graph assembly/execution contracts with device-agnostic orchestration.

- [ ] finalize `graph::sm` orchestration flow (`reserve`, `compute`, done/error).
- [ ] align `graph/assembler` inputs to consume `batch::plan` + memory view.
- [ ] align `graph/allocator` interval/liveness rules and offset outputs.
- [ ] align `graph/processor` runtime opcode routing to kernel events.
- [ ] validate tensor lifecycle/ref-count handling in graph execution loop.
- [ ] add tests for reserve/compute reuse and topology-change rebuild paths.
- [ ] phase gate: run `scripts/quality_gates.sh`.

exit criteria:
- [ ] graph contracts stable across reserve + per-step compute.
- [ ] execution path deterministic and test-covered.
- [ ] quality gates green.
- [ ] wait point before phase 6: graph compute events and output buffer contracts are frozen.

---

## phase 5: kernel domain (any + hardware variants + ops)

goal: unify kernel contract on per-op event execution.

- [ ] normalize `kernel::any` and backend docs/code to per-op `op::*` dispatch.
- [ ] remove graph-level scheduling semantics from kernel backends.
- [ ] validate fallback path behavior via `sml::unexpected_event` and acyclic routing.
- [ ] align op payload structures with trivially-copyable event constraints where required.
- [ ] add tests for supported op dispatch and unsupported-op fallback/error behavior.
- [ ] phase gate: run `scripts/quality_gates.sh`.

exit criteria:
- [ ] one kernel contract only: per-op dispatch.
- [ ] fallback behavior deterministic and bounded.
- [ ] quality gates green.
- [ ] wait point before phase 6: kernel op event payloads are frozen.

---

## phase 6: generator orchestration end-to-end

goal: wire all machines into the final step loop and lifecycle API.

- [ ] finalize generator-owned components and injected dependencies.
- [ ] implement end-to-end pipeline:
      `conditioner -> token/batcher -> batch/planner -> graph -> logits/validator -> logits/sampler -> renderer`.
- [ ] verify sequence lifecycle API synchronization with memory + graph.
- [ ] add integration tests for text generation loop behavior and failure routing.
- [ ] phase gate: run `scripts/quality_gates.sh`.

exit criteria:
- [ ] full end-to-end single-modality flow working with new machine boundaries.
- [ ] integration tests pass.
- [ ] quality gates green.
- [ ] wait point before phase 7: generator integrates text, token/batch, memory, graph, kernel
      without temporary adapters.

---

## phase 7: parity + performance hardening

goal: recover output parity and protect performance after architecture changes.

- [ ] run paritychecker suites and close token/output deltas.
- [ ] baseline and resolve benchmark regressions introduced by rearchitecture.
- [ ] verify quality gate runtime and benchmark snapshot behavior.
- [ ] tighten hot-path allocations and copy behavior where regressions are found.
- [ ] phase gate: run `scripts/quality_gates.sh`.

exit criteria:
- [ ] paritychecker stable for supported fixtures.
- [ ] benchmark regressions resolved or explicitly approved.
- [ ] quality gates green.

---

## phase 8: cleanup + finalization

goal: remove migration residue and leave a clean maintainable tree.

- [ ] remove temporary scaffolds that are fully implemented.
- [ ] delete dead docs/design fragments superseded by finalized architecture.
- [ ] ensure AGENTS/rules references stay accurate with final paths.
- [ ] verify no transient artifacts are tracked.
- [ ] final full quality pass.
- [ ] phase gate: run `scripts/quality_gates.sh`.

exit criteria:
- [ ] no migration leftovers.
- [ ] docs and code reflect the same architecture.
- [ ] quality gates green.
