# Phase 2: EMEL Model Loading Path - Research

**Researched:** 2026-03-08
**Domain:** EMEL GGUF/model-loader integration for the paritychecker generation slice
**Confidence:** MEDIUM-HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

No phase-specific `CONTEXT.md` exists, so this research uses locked project decisions from
`.planning/PROJECT.md`, `.planning/ROADMAP.md`, `.planning/STATE.md`, and `docs/rules/sml.rules.md`.

Locked constraints already established:
- The first slice stays pinned to `tests/models/Llama-68M-Chat-v1-Q2_K.gguf`.
- Acceptance remains in `tools/paritychecker/`, not a new public C API example.
- The slice must preserve the RTC actor model, no-queue invariant, and explicit error publication.
- Phase 2 must stop at real EMEL model loading; generator initialization belongs to Phase 3.
</user_constraints>

<research_summary>
## Summary

Phase 2 is the first point where generation mode needs to stop being a harness stub and start
touching real EMEL runtime state. The existing orchestration seam already exists:
`src/emel/model/loader/sm.hpp` drives a `parse -> load_weights -> map_layers ->
validate_structure -> validate_architecture` lifecycle through injected callbacks, and its request
surface hands ownership of `emel::model::data` to the caller. That is the right seam for
paritychecker.

The critical brownfield fact is that the lower GGUF backend is not complete yet.
`src/emel/gguf/loader/sm.hpp` is a real actor boundary, but the concrete helpers in
`src/emel/gguf/loader/detail.hpp` are still stubs. The current paritychecker tokenizer path also
bypasses EMEL loading entirely by loading the model with llama.cpp and copying only vocab data into
an EMEL layout. Phase 2 therefore cannot be planned as "just wire paritychecker into existing load
helpers." It needs two linked outcomes:

1. Make the EMEL GGUF/model-loading path real enough for the pinned Llama-68M slice.
2. Route paritychecker generation mode through that path and publish deterministic load success or
   failure evidence without widening scope into generator init.

**Primary recommendation:** Build a paritychecker-local adapter around `model::loader::event::load`
using the existing `gguf::loader` and `weight_loader` actors as the concrete backend, then make the
generation branch report explicit `load_done` / `load_error` outcomes.
</research_summary>

<standard_stack>
## Standard Stack

This phase should extend the repo's current runtime stack rather than introduce a parallel loader.

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| C++20 | repo standard | Runtime and tool implementation | Already enforced across the repo |
| Boost.SML | pinned in repo | Loader and GGUF actor orchestration | Required by project rules and existing loader machines |
| doctest | vendored in repo | Existing loader and paritychecker verification surface | Already used by `tests/model/loader/*`, `tests/gguf/loader/*`, and `tools/paritychecker/*` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `scripts/paritychecker.sh` | repo standard | Fast build and paritychecker regression check | Run after every task commit |
| `scripts/quality_gates.sh` | repo standard | Full repo-level gate | Run at the end of the phase |
| `tests/models/Llama-68M-Chat-v1-Q2_K.gguf` | local fixture | Pinned real GGUF input | Use for success-path paritychecker load verification |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Reusing `model::loader::sm` | New paritychecker-only load orchestration | Violates the existing EMEL architecture boundary |
| Implementing GGUF parsing behind `gguf::loader` | Parsing GGUF directly in paritychecker | Bypasses the actor contract Phase 2 is supposed to exercise |
| Stopping at vocab-only load | Full model-load slice now | Vocab-only would not satisfy the roadmap's "loaded `model::data` is available for later generator initialization" requirement |
</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Recommended Project Structure
```text
src/emel/gguf/loader/
├── detail.hpp            # concrete probe/parse backend for the GGUF actor
├── actions.hpp           # actor actions remain bounded and delegate into detail
└── sm.hpp                # existing orchestration contract

src/emel/model/loader/
├── events.hpp            # caller-owned model_data + callback contract
└── sm.hpp                # existing orchestration contract

tools/paritychecker/
└── parity_runner.cpp     # paritychecker-local adapter around the EMEL loaders
```

### Pattern 1: Adapter Around Existing Loader Actors
**What:** Use `model::loader::event::load` as the single paritychecker entry surface and satisfy it
with paritychecker-local callbacks that drive `gguf::loader` and `weight_loader`.
**When to use:** When the orchestration contract exists but the concrete backend still has to be
wired for the current slice.
**Example:** `parity_runner.cpp` owns the file image, scratch storage, callback captures, and the
caller-owned `emel::model::data`.

### Pattern 2: Caller-Owned Model Data
**What:** Keep `emel::model::data` owned by the paritychecker generation request and let
`events::load_done` confirm that the object is now populated enough for the next phase.
**When to use:** When later phases need to inject the same loaded model into `src/emel/generator`.
**Example:** On success, paritychecker can report stable evidence from `model_data`, such as
`n_tensors`, `n_layers`, `bytes_total`, and `used_mmap`.

### Pattern 3: Phase-Limited Runtime Output
**What:** Emit deterministic load success or failure signals and stop before generator
initialization.
**When to use:** When the roadmap wants real load-path proof now, but generator init and decode
belong to later phases.
**Example:** Replace the Phase 1 reserved harness message with explicit load evidence, but do not
construct or dispatch the generator actor yet.

### Anti-Patterns to Avoid
- **Parody load path:** do not call llama.cpp load functions from generation mode and label the
  result "EMEL loading."
- **Stub success propagation:** do not keep `gguf::loader/detail.hpp` returning success without
  parsing and then claim the load path is real.
- **Generator creep:** do not let Phase 2 absorb generator initialization or decode responsibilities
  from Phase 3 and Phase 4.
- **`mock_main.cpp` revival:** it reflects an older API shape and should not drive Phase 2 design.
</architecture_patterns>

<dont_hand_roll>
## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Top-level model load orchestration | New paritychecker state machine | `src/emel/model/loader/sm.hpp` | Existing actor already models the lifecycle and outcomes |
| GGUF event sequencing | Manual probe/bind/parse flow without the actor | `src/emel/gguf/loader/sm.hpp` | Phase 2 specifically needs to prove paritychecker reaches that actor boundary |
| Success/failure reporting | Ad hoc booleans in context | `events::load_done` / `events::load_error` and deterministic CLI output | Matches project event/outcome rules |
| Fixture identity | Directory scans or implicit defaults | Existing pinned Phase 1 fixture contract | Prevents drift to a different `.gguf` file |

**Key insight:** The right brownfield move is not "teach paritychecker how to parse GGUF." The
right move is "make paritychecker drive the EMEL loaders that own GGUF parsing and model assembly."
</dont_hand_roll>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: Treating the current GGUF loader as already complete
**What goes wrong:** Planning assumes `gguf::loader` is real end-to-end because the state machine
exists.
**Why it happens:** The actor boundary is present, but `detail.hpp` still returns stub results.
**How to avoid:** Plan explicit backend work in `src/emel/gguf/loader/detail.hpp`.
**Warning signs:** The phase only touches `tools/paritychecker/` and never updates GGUF backend code.

### Pitfall 2: Proving only vocab load instead of model load
**What goes wrong:** The implementation reproduces the tokenizer parity bypass with a slightly
different wrapper.
**Why it happens:** The existing tokenizer path already copies vocab out of llama.cpp, so it is an
easy local shortcut.
**How to avoid:** Make `events::load_done` evidence include whole-model signals, not vocab-only
signals.
**Warning signs:** Success output never mentions tensor count, layer count, or weight-load bytes.

### Pitfall 3: Losing deterministic error publication
**What goes wrong:** Invalid path or loader rejection becomes a generic failure with unstable text.
**Why it happens:** Multiple callback boundaries can collapse distinct failures into one message.
**How to avoid:** Normalize missing-file, invalid-request, parse-failed, backend-error, and
model-invalid outcomes at the paritychecker boundary.
**Warning signs:** The same stderr text appears for missing file, malformed file, and loader error.

### Pitfall 4: Pulling generator init into the same phase
**What goes wrong:** Phase 2 sprawls into generator wiring, making it harder to validate where
loading ends and initialization begins.
**Why it happens:** A loaded `model::data` naturally invites immediate generator construction.
**How to avoid:** Stop the phase at a verified loaded model plus deterministic CLI evidence.
**Warning signs:** The plan starts modifying `src/emel/generator/*` or generator lifecycle tests.
</common_pitfalls>

## Validation Architecture

Phase 2 can still use the existing paritychecker gate path, but the quick checks need to assert
that generation mode now performs a real EMEL load rather than printing the Phase 1 reserved
harness message.

- Quick verification: `scripts/paritychecker.sh`
- Full verification: `scripts/quality_gates.sh`
- Additional spot checks:
  - `build/paritychecker_zig/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text "hello"`
  - `build/paritychecker_zig/paritychecker --generation --model tests/models/does-not-exist.gguf --text "hello"`
  - Confirm the success path prints load evidence and the failure path prints a deterministic reason

## Open Questions

1. **How much GGUF parsing is actually required for Phase 2's first slice?**
   - What we know: the current backend stubs are insufficient for a real load path.
   - What's unclear: whether Phase 2 should implement only the exact metadata/tensor path Llama-68M
     needs, or a broader reusable GGUF subset.
   - Recommendation: implement the minimum real subset needed by the pinned fixture, but do it
     behind `src/emel/gguf/loader/detail.hpp` so later phases can widen it without changing the
     actor boundary.

2. **Should architecture validation be fully real in Phase 2 or just explicitly wired?**
   - What we know: the model loader already has a `validate_architecture_impl` callback slot.
   - What's unclear: whether the current codebase already has enough architecture rules to make that
     callback meaningful for the pinned fixture.
   - Recommendation: wire the callback path now and make unsupported validation fail
     deterministically rather than silently skipping it.

3. **Can the phase rely on `tools/mock_main.cpp` as an integration reference?**
   - What we know: it describes an older intended composition, but its request shape no longer
     matches the current loader event API.
   - Recommendation: no; treat the current `src/emel/model/loader/*`, `src/emel/gguf/loader/*`,
     and `src/emel/model/weight_loader/*` APIs as source of truth.

## Metadata

**Research scope:**
- Core technology: EMEL GGUF/model loader and paritychecker generation harness
- Patterns: callback-driven loader orchestration, caller-owned model data, deterministic load
  outcomes
- Pitfalls: stubbed GGUF backend, vocab-only shortcuts, unstable error publication

**Confidence breakdown:**
- Loader orchestration seam: HIGH - the public event surfaces and lifecycle tests are already clear
- Concrete GGUF backend readiness: MEDIUM - the shape is clear, but the current detail helpers are
  still stubs
- Phase boundary confidence: HIGH - the roadmap and existing generator API make the load/init split
  explicit

**Research date:** 2026-03-08
**Valid until:** 2026-04-07

---
*Phase: 02-emel-model-loading-path*
*Research completed: 2026-03-08*
*Ready for planning: yes*
