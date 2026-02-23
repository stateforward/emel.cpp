# rearchitecture plan

status: draft
owner: emel
branch: `rearchitecture`

## execution rules

- [ ] treat `docs/designs/**` as the target architecture contract for this plan.
- [ ] keep changes incremental and phase-scoped (no cross-phase refactors in one PR).
- [ ] do not update snapshots without explicit approval.
- [ ] run `scripts/quality_gates.sh` at the end of every phase and record pass/fail notes in PR.
- [ ] if a phase gate fails, stop and fix regressions before starting the next phase.

---

## phase 1: organization + renaming (first)

goal: align source tree, namespaces, include paths, and test/bench layout to the new domain structure.

- [ ] create and validate a rename map from old paths to new paths based on `docs/designs/**`.
- [ ] move source directories to match domain/component/type conventions.
- [ ] update namespace declarations to match moved directories.
- [ ] update include paths across `src/`, `tests/`, `tools/bench/`, and `tools/paritychecker/`.
- [ ] remove stale paths and dead includes from the pre-rearchitecture layout.
- [ ] keep compatibility shims only if required for staged migration, and track each shim with removal TODO.
- [ ] move test files to mirror domain paths (`tests/<domain>/<component>/*`).
- [ ] move benchmark files to mirror machine/domain names.
- [ ] fix build scripts and CMake targets for moved files.
- [ ] verify no new unintended tracked artifacts (for example `.DS_Store`, `Testing/` logs).
- [ ] phase gate: run `scripts/quality_gates.sh`.

exit criteria:
- [ ] `src/` and `tests/` path layout match the new architecture naming scheme.
- [ ] build graph compiles without old-path includes.
- [ ] quality gates green.

---

## phase 2: public machine contracts and events

goal: normalize machine APIs, event names, and callback/outcome semantics to match SML rules and designs.

- [ ] audit all machine interfaces for event naming (`event::*`, `events::*_done`, `events::*_error`).
- [ ] remove `cmd_*` naming where present and align to noun-like intent events.
- [ ] standardize callback usage for immediate synchronous replies only.
- [ ] ensure callback fields are not stored in context.
- [ ] align bind/init event contracts across related machines (`conditioner`, `tokenizer`, `renderer`, `detokenizer`, `kernel`, `graph`).
- [ ] ensure unexpected external events route via `sml::unexpected_event`.
- [ ] remove stale alias exposure not intended for public API surface.
- [ ] update docs/comments for event I/O contracts where implementation changed.
- [ ] add/adjust unit tests for event contract behavior and error routing.
- [ ] phase gate: run `scripts/quality_gates.sh`.

exit criteria:
- [ ] event contracts match docs and SML policy.
- [ ] callback and unexpected-event behavior verified by tests.
- [ ] quality gates green.

---

## phase 3: text domain pipeline (conditioner/tokenizer/encoders/detokenizer/renderer)

goal: enforce text pipeline boundaries and ownership per design.

- [ ] finalize `text/conditioner` contract to emit raw token arrays only.
- [ ] keep formatting concerns in `text/formatter` and `text/jinja/*` components.
- [ ] align `text/tokenizer` ownership of preprocessor + encoder dispatch.
- [ ] align `text/encoders::any` variant routing and kind mapping.
- [ ] implement/normalize `text/detokenizer` + `text/renderer` contracts for byte buffering and stop matching.
- [ ] ensure no generator-specific sequencing logic leaks into text codec components.
- [ ] add focused tests for conditioner/tokenizer/renderer boundaries.
- [ ] phase gate: run `scripts/quality_gates.sh`.

exit criteria:
- [ ] text pipeline composes cleanly with no ownership ambiguity.
- [ ] tokenizer and renderer boundaries are deterministic and test-covered.
- [ ] quality gates green.

---

## phase 4: token batching and planning

goal: make token intake deterministic and produce executable `batch::plan` steps.

- [ ] finalize `token/batcher` sanitation and autopopulation rules.
- [ ] align `token/batcher` output contract with `batch/planner` input contract.
- [ ] finalize `batch/planner` step slicing and mapping structure.
- [ ] ensure planner remains stateless and allocation-free during dispatch.
- [ ] add tests for malformed input rejection, continuity rules, and split policy behavior.
- [ ] add bench coverage for hot-path planner scenarios.
- [ ] phase gate: run `scripts/quality_gates.sh`.

exit criteria:
- [ ] `token/batcher -> batch/planner` contract stable and test-verified.
- [ ] planned steps are directly consumable by graph processing.
- [ ] quality gates green.

---

## phase 5: memory domain integration (kv/recurrent/hybrid + coordinator any)

goal: align memory ownership, lifecycle events, and runtime status interfaces.

- [ ] align `memory/kv`, `memory/recurrent`, and `memory/hybrid` state/event contracts with design docs.
- [ ] finalize `memory/coordinator::any` role and variant selection semantics.
- [ ] verify sequence lifecycle operations (`allocate`, `branch`, `free`) are deterministic.
- [ ] ensure rollback semantics for partial failure paths in hybrid/coordinator flows.
- [ ] add tests for capacity errors, branching, and lifecycle consistency.
- [ ] add/refresh bench coverage for coordinator hot paths.
- [ ] phase gate: run `scripts/quality_gates.sh`.

exit criteria:
- [ ] memory lifecycle semantics are stable under mixed workloads.
- [ ] coordinator and memory actors communicate only via explicit events.
- [ ] quality gates green.

---

## phase 6: graph domain (graph/assembler/allocator/processor)

goal: make graph assembly and execution contracts explicit and device-agnostic.

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

---

## phase 7: kernel domain (any + hardware variants + ops)

goal: unify kernel contract on per-op event execution.

- [ ] normalize `kernel::any` and backend docs/code to per-op `op::*` dispatch model.
- [ ] remove remaining graph-level scheduling semantics from kernel backends.
- [ ] validate fallback path behavior via `sml::unexpected_event` and acyclic routing.
- [ ] align op payload structures and ensure trivially-copyable event constraints where required.
- [ ] add tests for supported op dispatch and unsupported-op fallback/error behavior.
- [ ] phase gate: run `scripts/quality_gates.sh`.

exit criteria:
- [ ] one kernel contract only: per-op dispatch.
- [ ] fallback behavior deterministic and bounded.
- [ ] quality gates green.

---

## phase 8: generator orchestration end-to-end

goal: wire all machines into the final step loop and lifecycle API.

- [ ] finalize generator-owned components and injected dependencies.
- [ ] implement end-to-end step pipeline:
      `conditioner -> token/batcher -> batch/planner -> graph -> logits/validator -> logits/sampler -> renderer`.
- [ ] verify sequence lifecycle API synchronization with memory + graph.
- [ ] add integration tests for text generation loop behavior and failure routing.
- [ ] phase gate: run `scripts/quality_gates.sh`.

exit criteria:
- [ ] full end-to-end single-modality flow working with new machine boundaries.
- [ ] integration tests pass.
- [ ] quality gates green.

---

## phase 9: parity + performance hardening

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

## phase 10: cleanup + finalization

goal: remove migration residue and leave a clean maintainable tree.

- [ ] remove temporary shims and stale compatibility aliases.
- [ ] delete dead docs/design fragments superseded by finalized architecture.
- [ ] ensure AGENTS/rules references stay accurate with final paths.
- [ ] verify no transient artifacts are tracked.
- [ ] final full quality pass.
- [ ] phase gate: run `scripts/quality_gates.sh`.

exit criteria:
- [ ] no migration leftovers.
- [ ] docs and code reflect same architecture.
- [ ] quality gates green.
