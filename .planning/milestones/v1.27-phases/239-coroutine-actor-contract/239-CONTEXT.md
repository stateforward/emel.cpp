# Phase 239: Coroutine Actor Contract - Context

**Gathered:** 2026-05-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Codify the `co_sm` coroutine actor contract in project rules before any runtime behavior depends on
coroutine dispatch. This phase updates rules/docs only; it does not add `emel::co_sm`, async I/O
runtime code, schedulers, or tests beyond source/text checks needed to validate the contract.

</domain>

<decisions>
## Implementation Decisions

### Scope
- Treat this phase as pure infrastructure/documentation for `CO-01`.
- Keep the contract aligned between `docs/rules/sml.rules.md`, `AGENTS.md`, and `CLAUDE.md`.
- Preserve the existing synchronous RTC/no-queue rules and add a narrow opt-in coroutine actor
  contract instead of weakening default actor rules.

### Contract Shape
- Coroutine continuations are internal actor/scheduler progress, not mailbox messages.
- Scheduler guarantees must be FIFO, single-consumer per actor, bounded, and run-to-completion for
  the selected cooperative tick.
- `co_await` may appear only at explicit SML phase boundaries and must not hide behavior selection.
- Suspension-surviving state must be owned by stable actor/scheduler/caller storage; stack-backed
  event payload, spans, mutable references, and callbacks must not be retained across suspension.

### Claude's Discretion
Use concise rule text in the existing style and avoid broad async-inference or platform-specific
implementation details.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `docs/third_party/sml.md` documents an intended `emel::co_sm` scheduler policy surface.
- `src/emel/sm.hpp` currently wraps synchronous `stateforward::sml::sm`; no project-owned
  `emel::co_sm` wrapper is implemented yet.

### Established Patterns
- `docs/rules/sml.rules.md` is the source of truth; `AGENTS.md` and `CLAUDE.md` mirror project
  engineering guidance.
- Prior I/O strategy milestones kept concrete strategy behavior under `src/emel/io` and tensor
  residency ownership under `model/tensor`.

### Integration Points
- `docs/rules/sml.rules.md`
- `AGENTS.md`
- `CLAUDE.md`

</code_context>

<specifics>
## Specific Ideas

Issue #64 explicitly forbids retaining borrowed stack-backed request spans/references across
suspension, storing callbacks for later invocation, and hiding control flow inside blocking
actions. The rule contract should make those constraints hard requirements.

</specifics>

<deferred>
## Deferred Ideas

Runtime `emel::co_sm` wrapper, scheduler policies, async I/O component code, and behavior tests are
deferred to Phase 240 and later.

</deferred>
