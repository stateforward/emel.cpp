# Phase 4: Deterministic Generation Parity - Research

**Researched:** 2026-03-08
**Domain:** EMEL prompt-to-output generation parity in `tools/paritychecker/`
**Confidence:** MEDIUM

<user_constraints>
## User Constraints

No phase-specific `CONTEXT.md` exists, so this research uses locked project decisions from
`.planning/PROJECT.md`, `.planning/ROADMAP.md`, `.planning/STATE.md`, and
`docs/rules/sml.rules.md`.

Locked constraints already established:
- The first end-to-end generation target stays pinned to
  `tests/models/Llama-68M-Chat-v1-Q2_K.gguf`.
- Acceptance remains inside `tools/paritychecker/`, not a new public C API example.
- The slice must preserve the RTC actor model, no-queue invariant, and explicit outcome channels.
- Phase 4 must deliver deterministic generation behavior and structured parity evidence, not just a
  manual smoke string.
</user_constraints>

<research_summary>
## Summary

The important Phase 4 discovery is that `src/emel/generator/sm.hpp` is already further along than
the CLI harness. The generator machine already accepts `event::generate`, owns a bounded
prefill/decode/render/flush loop, and focused tests in `tests/generator/lifecycle_tests.cpp`
already prove that the machine can emit `generation_done` and `generation_error` with deterministic
`tokens_generated` and `output_length`.

The actual gap is in `tools/paritychecker/parity_runner.cpp`:

1. The tool still stops after `run_emel_initialize_generator(...)`.
2. The paritychecker backend callbacks are still initialization-grade scaffolding.
3. `backend_extract_outputs(...)` reports one output but does not fill real logits from model
   weights, so calling the current tool path "generation parity" would be false.
4. The tool has no reference-side generation invocation and no structured parity report yet.

That means Phase 4 should not start by redesigning `src/emel/generator/*`. It should add a
tool-local runtime harness that:

1. Executes the real EMEL `event::generate` path after Phase 3 initialization succeeds.
2. Replaces the fake paritychecker compute callbacks with a deterministic tool-local decode bridge
   that can populate logits for the pinned Llama-68M slice.
3. Runs the same prompt and deterministic settings through a reference `llama.cpp` path in
   paritychecker.
4. Compares more than final text: at minimum rendered text, `tokens_generated`, and
   `output_length`; preferably first mismatch metadata and an explicit stop classification when it
   can be derived without widening `src/` contracts unnecessarily.

**Primary recommendation:** Keep all reference coupling and decode bridging inside
`tools/paritychecker/`, let `src/emel/generator/*` remain the single source of orchestration
truth, and stage the work into three plans: EMEL generate dispatch, reference generation, then
structured mismatch diagnostics.
</research_summary>

<standard_stack>
## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| C++20 | repo standard | Tool and runtime implementation | Already enforced across the repo |
| Boost.SML | pinned in repo | Generator orchestration | Required by project rules and existing machines |
| doctest | vendored in repo | Focused generator behavior coverage | Already used across `tests/generator/*` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `llama.cpp` tool linkage | repo-local | Reference generation and tool-local decode bridge | Allowed in `tools/paritychecker/` only |
| `scripts/paritychecker.sh` | repo standard | Fast paritychecker build/smoke feedback | Run after each plan wave |
| `scripts/quality_gates.sh` | repo standard | Full repo-level verification | Run at phase end |

### Existing Local Patterns Worth Reusing
| File | Pattern | Why Relevant |
|------|---------|--------------|
| `tools/bench/memory/bench_common.hpp` | `llama_model`, `llama_batch_allocr`, reference memory setup | Shows existing in-repo reference-side batch and memory primitives already used under the tool exception |
| `tools/paritychecker/parity_runner.cpp` | EMEL load + initialize harness and deterministic CLI output | Phase 4 should extend this path, not create a parallel entrypoint |
| `tests/generator/lifecycle_tests.cpp` | Real `event::generate` execution with deterministic sampler | The machine contract is already proven; the CLI harness needs to catch up |
</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Pattern 1: Paritychecker-Owned Generation Session
**What:** Extend `generation_load_state` into a bounded generation session that owns the loaded
model, tokenizer, conditioner, generator, reference-side objects, fixed output buffers, and
callback capture for both initialize and generate.
**When to use:** When Phase 4 needs end-to-end execution in paritychecker without widening public
API surface.
**Example:** After Phase 3 initialize succeeds, paritychecker constructs one
`emel::generator::event::generate` request and captures `generation_done` / `generation_error`.

### Pattern 2: Tool-Local Decode Bridge for EMEL Compute
**What:** Replace the fake backend callbacks with a tool-local bridge that uses paritychecker-only
reference facilities to produce real logits for the EMEL generator path.
**When to use:** When EMEL orchestration is the target under test but the repo still allows
reference linkage in paritychecker.
**Example:** `validate`, `prepare_graph`, `alloc_graph`, `bind_inputs`, `run_kernel`, and
`extract_outputs` remain the generator contract, but their implementation becomes a deterministic
decode adapter backed by `llama.cpp` state inside the tool.

### Pattern 3: Dual-Path Deterministic Generation Harness
**What:** Run one prompt with the same model, prompt text, max-token bound, and greedy sampler
settings through both the EMEL generator path and a direct `llama.cpp` reference path.
**When to use:** When the requirement is parity evidence rather than only successful EMEL output.
**Example:** `--generation --model ... --text hello --max-tokens 1` yields one EMEL parity record
and one reference parity record, then compares them before printing success or mismatch.

### Pattern 4: Structured Parity Evidence
**What:** Publish a small parity record with deterministic fields instead of a yes/no string.
**When to use:** When requirements call for evidence beyond the final rendered bytes.
**Example:** Compare rendered text, generated token count, output byte length, and first mismatch
location; add stop classification only if it can be derived cleanly without widening machine
structure.

### Anti-Patterns to Avoid
- **Fake decode success:** do not reuse the Phase 3 placeholder backend and call the result
  generation parity.
- **Reference leak into `src/`:** keep all `llama_*` / `ggml_*` identifiers inside
  `tools/paritychecker/` only.
- **Generator rewrite:** do not restructure `src/emel/generator/sm.hpp` unless execution uncovers a
  concrete bug.
- **String-only parity:** do not declare parity success on final text alone when token count or
  termination behavior can already diverge.
</architecture_patterns>

<dont_hand_roll>
## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| EMEL generation orchestration | New paritychecker-only state machine | `src/emel/generator/sm.hpp` | The existing machine already owns the bounded generate flow |
| Reference batch/memory semantics | Ad hoc token arrays and hidden global state | Existing `llama.cpp` primitives already used in `tools/bench/*` | Reduces accidental drift from the pinned reference |
| Parity evidence | One final string compare | Structured parity record with deterministic fields | Matches `PARI-02` and existing planning guidance |
| Tool verification | Manual shell notes only | CLI smoke commands plus normal repo gates | Keeps the slice executable and reviewable |

**Key insight:** The real work is not inventing generation inside `src/`; it is making the
paritychecker harness truthful about the compute path and deterministic about the comparison.
</dont_hand_roll>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: Calling the current backend "real generation"
**What goes wrong:** `event::generate` is dispatched, but the paritychecker backend never writes
real logits, so output is undefined or only accidentally stable.
**How to avoid:** Treat the current backend callbacks as Phase 3 scaffolding and replace them
before claiming Phase 4 parity.

### Pitfall 2: Comparing only final text
**What goes wrong:** One prompt matches by accident while token count or termination behavior has
already drifted.
**How to avoid:** Compare at least output text, `tokens_generated`, and `output_length`, then dump
the first mismatching field.

### Pitfall 3: Widening scope into full public generation API
**What goes wrong:** The work drifts into new API design or examples outside paritychecker.
**How to avoid:** Keep all acceptance in `tools/paritychecker/` and postpone public API expansion.

### Pitfall 4: Breaking SML rules to expose diagnostics
**What goes wrong:** Per-dispatch state gets copied into generator context or runtime branching gets
moved into actions just to publish more information.
**How to avoid:** Keep diagnostics tool-local where possible and only widen event payloads if the
current contract is insufficient.
</common_pitfalls>

## Validation Architecture

Phase 4 should stay bounded but must prove both generation execution and parity comparison:

- Quick verification: `scripts/paritychecker.sh`
- Focused generator verification: `build/zig/emel_tests_bin --dt-test-case="*generator*"`
- Manual parity smoke:
  - `build/paritychecker_zig/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text "hello" --max-tokens 1`
  - confirm the output now reports generation parity rather than initialize-only readiness
- Full verification: `scripts/test_with_coverage.sh`
- Full gate: `scripts/quality_gates.sh`

## Open Questions

1. **How much structured evidence can stay tool-local?**
   - What we know: `generation_done` already exposes `tokens_generated` and `output_length`.
   - Planning assumption: Phase 4 can satisfy `PARI-02` without changing generator topology by
     starting with those fields plus output text and a first-mismatch report.

2. **Where should the reference decode bridge live?**
   - What we know: the repo already keeps reference-only code inside `tools/bench/*` and
     `tools/paritychecker/*`.
   - Recommendation: keep it in `tools/paritychecker/parity_runner.cpp` unless helper volume forces
     an extraction inside that tool directory.

3. **Do we need subprocess regression tests in this phase?**
   - What we know: the roadmap reserves full gate hardening for Phase 5.
   - Recommendation: no; keep Phase 4 focused on executable parity behavior and leave default test
     surface expansion to Phase 5.

## Metadata

**Research scope:**
- Core technology: generator `event::generate` plus tool-local reference decode integration
- Patterns: paritychecker-owned session state, deterministic dual-path execution, structured parity
  record
- Pitfalls: fake backend parity, string-only comparison, reference leakage into `src/`

**Confidence breakdown:**
- Generator generate seam: HIGH - current machine and tests already prove it exists
- Tool-local reference bridge: MEDIUM - repo patterns exist, but the exact paritychecker adapter is
  still implementation work
- Structured parity evidence: MEDIUM-HIGH - enough deterministic fields already exist to ship a
  useful first slice without a machine rewrite
