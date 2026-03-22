# Phase 3: Generator Initialization Wiring - Research

**Researched:** 2026-03-08
**Domain:** EMEL generator initialize integration for the paritychecker Llama-68M slice
**Confidence:** MEDIUM-HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

No phase-specific `CONTEXT.md` exists, so this research uses locked project decisions from
`.planning/PROJECT.md`, `.planning/ROADMAP.md`, `.planning/STATE.md`, and
`docs/rules/sml.rules.md`.

Locked constraints already established:
- The first end-to-end slice stays pinned to `tests/models/Llama-68M-Chat-v1-Q2_K.gguf`.
- Acceptance remains in `tools/paritychecker/`, not a new public C API example.
- Phase 3 must preserve the RTC actor model, no-queue invariant, and explicit error publication.
- Phase 3 stops at generator initialization; bounded generation execution belongs to Phase 4.
</user_constraints>

<research_summary>
## Summary

Phase 2 already loads the pinned GGUF fixture into a caller-owned `emel::model::data` through the
real EMEL load path, but paritychecker still stops immediately after `model::loader::events::load_done`.
The next seam is clear: `src/emel/generator/sm.hpp` already exposes a real
`event::initialize` contract and owns the child-actor graph the roadmap wants to exercise
(`conditioner`, `memory`, `graph`, `sampler`, and `renderer`).

The important brownfield constraint is that generator initialization needs more than a loaded model.
`event::initialize` also requires a tokenizer actor, tokenizer dispatch callbacks, backend callback
functions for graph execution, a sampler function span, and opaque `model_topology`,
`prefill_plan`, and `decode_plan` handles. Repo searches did not find shared production objects for
those topology/plan handles outside generator and graph tests. The practical Phase 3 move is
therefore not to invent a new orchestration path, and not to widen scope into real decode compute.
It is to build a paritychecker-local initialization harness that:

1. Owns the missing actors and callback capture around the already loaded `emel::model::data`.
2. Dispatches the real `emel::generator::event::initialize`.
3. Uses bounded opaque topology/plan placeholders and deterministic backend callbacks sufficient
   for initialization only.
4. Publishes explicit `initialize_done` / `initialize_error` evidence at the tool boundary.

**Primary recommendation:** Bridge paritychecker into the existing generator actor graph using a
tool-local harness state, but keep the compute inputs phase-limited to initialization scaffolding
until Phase 4 introduces real prefill/decode execution.
</research_summary>

<standard_stack>
## Standard Stack

Phase 3 should extend the current runtime path rather than create a parallel init path.

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| C++20 | repo standard | Runtime and tool implementation | Already enforced across the repo |
| Boost.SML | pinned in repo | Generator and child-actor orchestration | Required by project rules and existing machines |
| doctest | vendored in repo | Existing generator lifecycle coverage | Already used by `tests/generator/*` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `scripts/paritychecker.sh` | repo standard | Fast build and paritychecker regression check | Run after every task commit |
| `scripts/quality_gates.sh` | repo standard | Full repo-level gate | Run at the end of the phase |
| `tests/models/Llama-68M-Chat-v1-Q2_K.gguf` | local fixture | Pinned real GGUF input | Use for the paritychecker initialize success path |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Reusing `src/emel/generator/sm.hpp` | Tool-local init booleans plus direct child calls | Violates the actor-boundary requirement for Phase 3 |
| Opaque init-only handles for topology/plans | Trying to introduce real decode plans now | Pulls Phase 4 compute scope into Phase 3 |
| Focused generator + CLI smoke verification | Broad subprocess matrix in paritychecker | Increases surface area before the first real generate path exists |
</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Recommended Project Structure
```text
tools/paritychecker/
└── parity_runner.cpp          # owns the load->initialize bridge and outcome publication

src/emel/generator/
├── events.hpp                 # initialize contract stays the single source of truth
├── sm.hpp                     # existing actor orchestration remains unchanged
└── actions.hpp                # child-actor sequencing remains unchanged

tests/generator/
└── lifecycle_tests.cpp        # paritychecker-style initialize assumptions and lifecycle checks
```

### Pattern 1: Paritychecker-Owned Initialization Harness
**What:** Add a bounded state object in paritychecker that owns the tokenizer actor, conditioner
actor, generator actor, sampler function span, backend placeholder context, and callback capture.
**When to use:** When the loaded model already exists, but the tool still lacks the actor graph
needed to reach `generator::ready`.
**Example:** After `load_done`, paritychecker constructs `generator::sm` with the loaded model and
then dispatches `event::initialize`.

### Pattern 2: Init-Only Opaque Compute Inputs
**What:** Keep `model_topology`, `prefill_plan`, `decode_plan`, and graph backend callbacks
tool-local and opaque for this phase, matching the generator tests' contract shape.
**When to use:** When initialization needs non-null compute handles, but the roadmap has not yet
reached real prefill/decode execution.
**Example:** Use bounded placeholder handles and deterministic backend callbacks that satisfy
reserve/bind/init requirements without pretending Phase 4 compute already exists.

### Pattern 3: Explicit Initialize Outcome Capture
**What:** Capture `events::initialize_done` and `events::initialize_error` directly at the
paritychecker boundary and map them to stable stdout/stderr.
**When to use:** When the roadmap explicitly requires error publication and SML-visible lifecycle
proof, not just a boolean success path.
**Example:** `generation initialize ok (...)` only after `initialize_done`; non-zero exit with a
stable error name on rejection or `initialize_error`.

### Anti-Patterns to Avoid
- **Tool-local orchestration shortcut:** do not reserve memory, graph, or sampler state directly in
  paritychecker and then claim generator initialization succeeded.
- **Phase 4 creep:** do not start dispatching `event::generate` or reference decode work here.
- **Generator redesign:** do not reshape `src/emel/generator/*` unless a concrete bug requires it;
  Phase 3 should primarily be a bridge, not a machine rewrite.
- **Synthetic test flags:** do not add tool-only CLI knobs just to trigger init failures.
</architecture_patterns>

<dont_hand_roll>
## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Generator lifecycle orchestration | New paritychecker-only state machine | `src/emel/generator/sm.hpp` | The machine already models init success/error and child-actor sequencing |
| Actor ownership split | Global singletons or static helpers | Paritychecker-local harness state | Keeps ownership bounded to one generation request path |
| Init success/failure reporting | Ad hoc booleans | `initialize_done` / `initialize_error` callbacks plus deterministic CLI output | Matches project event/outcome rules |
| Compute-plan completeness | Real decode execution now | Opaque init-only handles and callbacks | Keeps the phase boundary aligned with the roadmap |

**Key insight:** Phase 3 should prove that paritychecker can enter the generator's real
initialization lifecycle, not that the full generation stack already works.
</dont_hand_roll>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: Stopping at load success again
**What goes wrong:** The implementation still reports only Phase 2 load evidence and never reaches
`generator::ready`.
**Why it happens:** The load path is already working and easy to leave in place.
**How to avoid:** Make `generator::event::initialize` the required next step after `load_done`.
**Warning signs:** The success output still contains "generator initialization reserved for later
phases."

### Pitfall 2: Treating the opaque compute inputs as full compute support
**What goes wrong:** The phase claims real prefill/decode readiness because the initialize request
contains non-null plan pointers.
**Why it happens:** The generator initialize contract accepts opaque handles by pointer.
**How to avoid:** Keep the docs and output explicit that these handles are initialization scaffolds
only, with real generation deferred to Phase 4.
**Warning signs:** The plan starts discussing prompt execution, logits comparison, or token output.

### Pitfall 3: Bypassing the child actors
**What goes wrong:** The tool mutates equivalent context directly instead of letting generator
actions dispatch into conditioner, memory, graph, sampler, and renderer.
**Why it happens:** Direct context mutation looks shorter than satisfying the full init request.
**How to avoid:** Use the existing actor graph and verify ready-state transitions through the real
machine.
**Warning signs:** The implementation does not construct `emel::text::conditioner::sm` or
`emel::generator::sm`.

### Pitfall 4: Hiding rejected initialize requests behind generic load errors
**What goes wrong:** Missing init fields or backend rejection collapse into vague tool output.
**Why it happens:** Multiple actor boundaries exist between paritychecker and generator.
**How to avoid:** Normalize rejected dispatch, `initialize_error`, and callback absence into
deterministic init-specific messages.
**Warning signs:** All failures still print `generation load failed`.
</common_pitfalls>

## Validation Architecture

Phase 3 should keep the same repo-level gates, but the phase-specific checks need to prove that the
tool now reaches generator initialization rather than stopping after model load.

- Quick verification: `scripts/paritychecker.sh`
- Focused unit verification: `build/zig/emel_tests_bin --dt-test-case="*generator*"`
- Full verification: `scripts/quality_gates.sh`
- Additional spot check:
  - `build/paritychecker_zig/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text "hello"`
  - Confirm the success path reports initialize success and does not yet claim prompt generation

## Open Questions

1. **Where should the init-only backend scaffolding live?**
   - What we know: paritychecker currently owns the load-path adapter in `parity_runner.cpp`.
   - Recommendation: keep the first slice there unless the helper count makes extraction necessary.

2. **How should Phase 3 choose initialization limits?**
   - What we know: `generator::event::initialize` requires explicit capacities and bounds, and the
     generator tests already use small fixed values that satisfy initialization.
   - Recommendation: start from the generator lifecycle fixture values and only widen if the pinned
     model proves they are insufficient.

3. **Do we need paritychecker subprocess matrices now?**
   - What we know: the roadmap's current goal is initialization wiring, not full generation parity.
   - Recommendation: no; keep validation to one CLI smoke path plus focused generator coverage.

## Metadata

**Research scope:**
- Core technology: EMEL generator initialize contract and paritychecker load->initialize handoff
- Patterns: actor-owned harness state, init-only opaque compute handles, explicit init outcome
  capture
- Pitfalls: load-only success, actor bypass, hidden init failures, premature decode scope

**Confidence breakdown:**
- Generator initialize seam: HIGH - the event surface and lifecycle tests are explicit
- Paritychecker bridge scope: HIGH - the current load success seam is narrow and well defined
- Concrete compute-plan inputs: MEDIUM - init can use opaque placeholders, but real decode objects
  still belong to later work
