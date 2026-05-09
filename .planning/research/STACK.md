# Research: Stack Additions for v1.27 co_sm Cooperative Async I/O

**Date:** 2026-05-09
**Milestone:** v1.27 co_sm Cooperative Async I/O Strategy
**Source:** GitHub issue #64 plus `docs/*-async.plan.md`, `docs/third_party/sml.md`, and current `src/emel/sm.hpp`.

## Existing Stack

- EMEL state machines use `stateforward::sml::sm` through the project-owned `emel::sm`
  wrapper in `src/emel/sm.hpp`.
- `docs/third_party/sml.md` documents an intended `emel::co_sm` coroutine extension surface:
  `emel::policy::coroutine_scheduler<emel::policy::fifo_scheduler<>>`, optional
  `inline_scheduler`, and scheduler requirements for `schedule(fn)`, `guarantees_fifo`,
  `single_consumer`, and `run_to_completion`.
- The tracked source currently has no `emel::co_sm` wrapper, no
  `process_event_async(...)` wrapper surface, and no project-owned coroutine scheduler or
  pooled coroutine allocator code in `src/emel/sm.hpp`.
- Shipped I/O strategies are synchronous actors under `src/emel/io`: `mmap`, `read`, and
  `staged_read`, with tensor residency still owned by `model/tensor`.

## Required Additions

- Add a project-owned `emel::co_sm` wrapper parallel to `emel::sm`, rather than replacing
  existing synchronous machines.
- Add or bind the upstream `stateforward::sml` coroutine utility API that provides
  `co_sm` / async dispatch in the repo's dependency surface.
- Add scheduler policy types under an EMEL-owned namespace, with compile-time checks for FIFO,
  single-consumer, and run-to-completion guarantees.
- Add fixed-capacity coroutine frame ownership or a proof that coroutine frames are allocated
  only during setup and never during dispatch/hot loading progress.
- Add deterministic coroutine trace/test utilities before relying on coroutine behavior for
  production I/O strategy semantics.

## What Not To Add

- No generic thread-pool offload as a substitute for an explicit cooperative async I/O strategy.
- No public C ABI exposure of coroutine tasks, scheduler types, awaitables, or coroutine frame
  ownership details.
- No `co_await` in kernel loops, quantized kernels, flash attention, packing, or other numeric
  hot loops.
- No model-family, benchmark-family, or device-specific strategy widening in this milestone.

## Integration Notes

- `docs/rules/sml.rules.md` and `AGENTS.md` must be updated before runtime code depends on
  coroutine behavior.
- The first async strategy should live under `src/emel/io/**`, separate from shipped
  synchronous strategies, and integrate through existing public `emel/io` and `model/tensor`
  events.
- Existing quality gates must gain source checks for coroutine boundary leaks and hidden
  mailbox-like behavior.
