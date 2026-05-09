# Research: Pitfalls for v1.27 co_sm Cooperative Async I/O

**Date:** 2026-05-09
**Milestone:** v1.27 co_sm Cooperative Async I/O Strategy

## Risks and Prevention

| Pitfall | Warning Sign | Prevention |
|---------|--------------|------------|
| Treating continuations as a mailbox | Scheduler accepts unbounded work or retains arbitrary events for later. | Require FIFO, single-consumer, run-to-completion scheduler contracts and bounded tick tests. |
| Retaining borrowed request data across suspension | Async context stores pointers/references to event payload, source spans, target windows, or callbacks. | Require owned request/progress storage and tests that prove no stack-backed data survives suspension. |
| Hidden behavior selection in awaitables/actions | Awaitable chooses backend, error path, fallback, callback, or next phase. | Keep runtime choice in guards/transitions; add source checks and code review gates. |
| Heap allocation during async dispatch | Coroutine frames allocate from heap or pool exhaustion silently falls back. | Fixed-capacity pools or setup-time allocation only; allocation counters and hard-fail tests. |
| Regressing synchronous strategies | Existing mmap/read/staged-read behavior changes while adding async path. | Add non-regression doctests and guardrails over shipped strategies. |
| Actor-internal reach-through from tools | Bench/parity/probe code includes async actor `detail.hpp` or coroutine internals. | Public-surface source scans and maintained lane tests. |
| Over-broad ARM inference scope | Roadmap drifts into decode batching, tokenizer overlap, NPU, or kernel work before I/O async is safe. | Keep issue #64 requirements limited to cooperative async I/O and defer broader scheduler experiments. |
| Ambiguous RTC boundary | `process_event_async(...)` appears to allow recursive dispatch or callback-later behavior. | Document coroutine RTC semantics and prove ordering with deterministic state traces. |

## Review Hot Spots

- `src/emel/sm.hpp`: wrapper API, scheduler policy, allocation model, sync behavior preservation.
- `docs/rules/sml.rules.md` / `AGENTS.md`: coroutine actor contract must be explicit and consistent.
- `src/emel/io/**`: async strategy must be separate from synchronous strategies.
- `src/emel/model/tensor/**`: residency ownership must remain tensor-owned.
- `tools/bench`, `tools/paritychecker`, probes: no actor-internal or coroutine-type reach-through.

## Stop Conditions

- If the repo cannot expose or pin the needed `co_sm` utility surface, stop after documenting the
  blocker instead of inventing a parallel coroutine framework.
- If no-heap dispatch cannot be proven, do not land production async loading behavior.
- If async behavior requires retained callbacks or stack-backed request data, redesign before
  continuing.
