# Research: Architecture for v1.27 co_sm Cooperative Async I/O

**Date:** 2026-05-09
**Milestone:** v1.27 co_sm Cooperative Async I/O Strategy

## Existing Integration Points

- `src/emel/sm.hpp` is the project-owned wrapper around `stateforward::sml::sm` and is the
  natural place to add an opt-in `emel::co_sm` wrapper.
- `docs/rules/sml.rules.md` and `AGENTS.md` are the source of truth for actor-model rules and
  must describe coroutine actors before any production actor relies on them.
- `src/emel/io` owns loading strategy boundaries. Synchronous strategies already exist for
  mmap, read/copy, and staged source-span loading.
- `src/emel/model/tensor` owns tensor load, bind, evict, and residency transitions. Async I/O
  must remain below that boundary.
- `model/loader`, benchmarks, paritychecker, and probes must route through public loader/tensor/I/O
  surfaces and must not include actor internals.

## Recommended Build Order

1. **Coroutine Rules and Wrapper:** Codify coroutine actor invariants, add `emel::co_sm`, add
   scheduler concepts/policies, and prove synchronous behavior is unchanged.
2. **Async I/O Component Boundary:** Add a dedicated cooperative async I/O strategy component with
   fail-closed behavior and canonical layout.
3. **Owned Progress State:** Define request/progress ownership that can survive suspension without
   retaining stack-backed spans, mutable internal references, or callbacks.
4. **Suspend/Resume Semantics:** Model bounded progress ticks, suspend/resume, completion, and
   errors through explicit SML states/events.
5. **Tensor Integration:** Wire `model/tensor` and `io/loader` through public events so tensor-owned
   residency consumes async progress and terminal outcomes.
6. **Maintained Surfaces and Guardrails:** Add tests, docs, reporting, and source checks that prove
   async usage is truthful and does not regress synchronous strategies.

## Data and Control Flow

```text
caller/model_loader
  -> model/tensor request_async_load
  -> io/loader selects cooperative async strategy
  -> io/async strategy co_sm processes bounded progress/resume events
  -> io/async publishes progress/done/error
  -> model/tensor commits residency only after explicit terminal success
```

## Architectural Constraints

- Coroutine continuations are internal progress, not mailbox messages.
- Scheduler ticks must be bounded and single-consumer per actor.
- `co_await` is allowed only at explicit phase boundaries modeled in SML.
- Runtime behavior selection remains in `guards.hpp` and `sm.hpp`; awaitables and actions do not
  choose backend, fallback, error channel, callback path, or next behavior.
- Any suspension-surviving data must be owned by stable actor/scheduler/request storage.
- No public C ABI or generic public runtime header should expose coroutine implementation types.
