# Research: Feature Scope for v1.27 co_sm Cooperative Async I/O

**Date:** 2026-05-09
**Milestone:** v1.27 co_sm Cooperative Async I/O Strategy

## Table Stakes

| Feature | Expected Behavior | Notes |
|---------|-------------------|-------|
| Coroutine actor contract | Maintainers can inspect project rules and understand exactly how `co_sm` preserves RTC, no-queue, allocation, callback, and lifetime invariants. | Must land before async runtime behavior. |
| Opt-in `co_sm` wrapper | EMEL can construct a coroutine-capable machine through a project-owned alias/wrapper without migrating existing actors by default. | Keep synchronous `emel::sm` unchanged. |
| Dedicated async I/O strategy | A separate `src/emel/io` strategy owns cooperative async/resumable loading behavior. | Do not mutate shipped `io/mmap`, `io/read`, or `io/staged_read` semantics. |
| Explicit suspend/resume progress | Callers can advance bounded loading progress and observe partial progress, resume, success, and errors through explicit states/events. | No hidden mailbox or unbounded queue. |
| Suspension-safe ownership | Any state that survives suspension is owned by the async strategy or fixed storage, not borrowed from stack-backed request spans or stored callbacks. | Issue #64 hard constraint. |
| Tensor-owned integration | `model/tensor` can drive resumable loading through the public I/O boundary while retaining load/bind/evict/residency ownership. | Builds on #59/#60. |
| Verification | Tests cover suspend/resume ordering, error propagation, partial progress, no-heap dispatch, and non-regression of synchronous strategies. | Must use public dispatch surfaces. |

## Differentiators Worth Including

| Feature | Value | Scope Decision |
|---------|-------|----------------|
| Deterministic coroutine trace tests | Makes first `co_sm` adoption auditable and prevents scheduler behavior from becoming folklore. | In scope. |
| Inline/no-op coroutine baseline | Proves `co_sm` overhead before real suspension code becomes load-bearing. | In scope if small and source-backed. |
| Async strategy reporting | Maintained tool lanes can truthfully report async strategy usage only when the async runtime path executed. | In scope. |

## Deferred

| Feature | Reason |
|---------|--------|
| Continuous decode batching | Valuable broader `co_sm` experiment, but issue #64 is about async I/O / resumable tensor loading. |
| Tokenizer/detokenizer overlap | Broader inference scheduler work, not this I/O strategy milestone. |
| `io_uring`, `madvise`, vendor DMA, or NPU implementation | Platform-specific source ownership belongs after the project has a safe coroutine actor contract and generic async strategy shape. |
| Multi-threaded scheduler | Higher-risk target-specific work; first milestone should prove strict single-consumer semantics. |
| Kernel coroutine integration | Explicitly out of scope; kernels stay synchronous and kernel-owned. |
