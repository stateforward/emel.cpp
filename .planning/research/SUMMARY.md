# Research Summary: v1.27 co_sm Cooperative Async I/O

**Date:** 2026-05-09
**Milestone:** v1.27 co_sm Cooperative Async I/O Strategy
**Source:** GitHub issue #64

## Stack Additions

- Add an opt-in project-owned `emel::co_sm` wrapper parallel to `emel::sm`.
- Add scheduler policy types and compile-time contracts for FIFO, single-consumer,
  run-to-completion scheduling.
- Add fixed-capacity coroutine frame ownership or setup-time-only allocation proof.
- Add deterministic coroutine trace tests and no-heap dispatch checks.

## Feature Table Stakes

- Codify coroutine actor rules before implementation.
- Add a dedicated cooperative async I/O strategy under `src/emel/io`.
- Model suspend/resume, bounded progress, terminal success, and errors as explicit
  SML states/events.
- Keep suspension-surviving request/progress state owned and stable.
- Integrate only through the public `model/tensor -> emel/io` boundary.
- Preserve shipped synchronous `io/mmap`, `io/read`, and `io/staged_read` behavior.

## Watch Out For

- Do not treat coroutine continuations as a hidden mailbox or unbounded queue.
- Do not retain borrowed event payload, stack-backed spans, mutable references, or callbacks across
  suspension.
- Do not hide runtime behavior selection in awaitables, actions, or detail helpers.
- Do not broaden the milestone into decode batching, tokenizer overlap, platform-specific
  `io_uring`/DMA/NPU work, or kernel coroutine integration.

## Recommended Scope

The v1.27 milestone should be an I/O-focused first adoption of `co_sm`: rules, wrapper,
dedicated async loading strategy, explicit progress semantics, tensor integration, maintained
surfaces, and guardrails. Broader ARM inference scheduler work belongs in future milestones after
this actor contract is source-backed.
