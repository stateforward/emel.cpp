# Phase 228: Span, Target-Window, and Platform Gating — Context

**Gathered:** 2026-05-07  
**Status:** Ready for execution (manager approved start; stale Phase 227 hold messages superseded)

<domain>

## Phase boundary

Implement **guard-modeled precondition validation only** on `emel::io::staged_read::sm`: invalid **source span /
staging chunk contracts** (STG-02), invalid **caller-owned target-window / layout** preconditions (STG-03), and
**unsupported platform / resource shapes** (PLAT-02) must be rejected **via explicit transitions and guards before
any staged file-I/O-related acceptance** lands in Phase 229+.

This phase does **not** add syscall-backed copy loops, mmap, device paths, loader detail, or coroutine staging.
Orchestration choice stays in `guards.hpp` + `sm.hpp`; actions remain bounded bookkeeping / callbacks aligned with
`io/read` validation chains.

Canonical authority: `.planning/ROADMAP.md` **Phase 228** success criteria; `.planning/REQUIREMENTS.md` rows **STG-02**,
**STG-03**, **PLAT-02**.

</domain>

<decisions>

## Implementation decisions

### Discretion

- Event field layout and exact numeric invariants inside guards (chunk vs total span arithmetic, stride/alignment knobs)
  are engineer discretion **as long as** every rejection path is modeled with **complementary guard pairs** + explicit
  destination states/events and **no** behavior selection buried in `actions.hpp`/`detail`.

### Locked constraints (`AGENTS.md` + `docs/rules/sml.rules.md`)

- **RTC / no-queue**; no allocation during dispatch; guards are pure predicates; actions do not choose runtime variants.
- **No dispatch-local mirrored state** in `action::context` (Phase 227 empty context persists; payloads stay on typed
  events / internal carriers only across completion phases—pattern after `detail::read_tensor_runtime`).
- Mirror **`io/read`** structural lessons: INTERNAL completion carriers for phased validation chains; **`unexpected_event`**
  boundary rows remain defined.
- **No synthetic production test-only knobs** in `event` payloads; doctests drive machines through **public dispatch**
  only (`process_event`).
- **`read` / `mmap` strategies untouched** aside from unavoidable shared headers already included.

### Out of scope (later phases)

- Per-stage byte copy, monotone span coverage, terminal success semantics → **Phase 229** (STG-04..06).
- Context lifetime / transient handle rules → **Phase 230**.

</decisions>

<code_context>

## Existing code insights

### Reusable assets

- `src/emel/io/read/sm.hpp` — canonical **multi-step guard chain** pattern (`state_*_decision` + `completion<detail::*_runtime>`).
- `src/emel/io/read/guards.hpp` — predicates on span, layout, platform (`EMEL_IO_READ_PLATFORM_SUPPORTED`).
- `src/emel/io/read/detail.hpp` — INTERNAL carriers bridging public `event::read_*` → runtime completion events.
- `src/emel/io/read/actions.hpp`, `events.hpp`, `errors.hpp` — bookkeeping / callback symmetry for accepted vs terminal
  error shapes.
- `src/emel/io/events.hpp::tensor_load_span` — precedent for numeric span fields shared across I/O vocabulary.

### Integration points

- `src/emel/io/staged_read/**` Phase 227 scaffold owns the machine surface; Phase 228 extends **events**, **guards**, **errors**,
  **actions** (narrow), **`detail` INTERNAL carriers**, **`sm.hpp`** transitions.
- Future loader/tensor wiring consumes the same strategy actor; Phase 228 only proves **validators** ahead of staged copy
  semantics.

</code_context>

<specifics>

## Specific modelling notes

1. **STG-02 (source span + chunk contract):** Guards must enforce non-zero logical span, sane `file_offset` /
   `logical_byte_length` arithmetic (no overflow), positive `stage_chunk_bytes`, and coherent coverage (e.g. final partial
   chunk allowed; zero-length stages rejected)—exact predicates documented beside guard definitions.

2. **STG-03 (target window / layout):** Guards must enforce non-null writable target identity, contiguous window sizing
   against the staging contract baseline (minimum bytes cover the validated stage slab—exact rule matches STG-04 preview in
   plan), optional layout/stride predicates if spelled in events (defaults to contiguous byte window if unspecified).

3. **PLAT-02 (platform gate):** Introduce **`EMEL_IO_STAGED_READ_PLATFORM_SUPPORTED`** (or reuse read macro only if genuinely
   identical semantics—prefer staged-specific macro for clarity). Unsupported hosts route to **`state_unsupported_platform_error_decision`**
   analogue via guards + transitions, mirroring **`platform_read_*`**.

</specifics>

<canonical_refs>

## Canonical references

- `.planning/ROADMAP.md` — Phase **228** goal + success bullets.
- `.planning/REQUIREMENTS.md` — **STG-02**, **STG-03**, **PLAT-02**.
- `.planning/research/SUMMARY.md` — milestone tone + pitfalls for staged-read.
- `docs/rules/sml.rules.md`, `AGENTS.md` — guard/action split, forbidden context fields, naming.

</canonical_refs>

<deferred>

## Deferred ideas

- Actual deterministic staged copy proofs, monotone span tests, taxonomy polish, tensor graphs, tooling guardrails —
  ROADMAP **229–236**.

</deferred>

---

*Phase directory: `.planning/phases/228-span-target-window-platform-gating`*
