# Leveraging `co_sm` for High-Efficiency ARM Inference in EMEL

This document outlines a strategy to utilize `emel::co_sm` (C++20 Coroutine State Machines) to maximize inference efficiency on ARM (AArch64) architectures, specifically targeting Apple Silicon and ARMv8.2+ with dotprod/i8mm extensions.

The primary goal is to use asynchronous suspension to pipeline memory and computation, while strictly adhering to the EMEL engineering constraints defined in `GEMINI.md`.

## 1. Architectural Alignment

Before detailing performance strategies, we must ensure `co_sm` usage aligns with the `stateforward.SML` rules:

*   **No Dynamic Allocation on Hot Paths:** `co_sm` supports a `pooled_coroutine_allocator`. We must use this policy, sizing the pool strictly to `max_batch_size` or the statically bounded number of concurrent requests, ensuring zero heap allocations during the dispatch loop.
*   **Run-to-Completion (RTC):** Coroutine resumption via the scheduler must act as a distinct RTC boundary. `co_sm` maintains this by ensuring the coroutine suspension point yields control back to the caller, and resumption executes synchronously until the next yield or completion.
*   **No Branching in Actions:** Coroutines allow linearizing asynchronous flow. We must be careful not to introduce control-flow branching based on runtime values inside the coroutine actions; behavior selection must remain in the SML transitions and guards.

## 2. Key ARM Optimization Strategies

### A. Memory Pipelining and Prefetching (Memory Hiding)

ARM performance, especially for LLMs (which are memory-bandwidth bound during decode and compute-bound during prefill), is highly dependent on keeping the ALUs fed.

*   **Strategy:** Use `co_sm` to interleave tensor loading with AArch64 kernel execution.
*   **Execution:** A state machine action initiates a memory prefetch (e.g., triggering a background thread or DMA engine to pull the next layer's weights into L2/L3 cache, or issuing explicit `prfm` instructions) and `co_await`s a signal that the cache line is warm.
*   **Benefit:** The main orchestrator thread is freed to handle other requests (batching) or execute dotprod instructions on already-resident data, drastically reducing pipeline stalls.

### B. Heterogeneous Compute Orchestration (big.LITTLE)

Modern ARM chips utilize big.LITTLE or Performance/Efficiency (P/E) cores.

*   **Strategy:** The central `co_sm` orchestrator runs on an Efficiency (LITTLE) core. When the graph reaches a heavy dense computation node (e.g., Matrix-Vector multiplication utilizing `i8mm`), the machine dispatches the kernel to a thread pool pinned to Performance (big) cores.
*   **Execution:** The action returns a `bool_task` and the `co_sm` suspends. The Performance core executes the tight, allocation-free assembly loop. Upon completion, the custom `co_sm` scheduler resumes the orchestrator on the Efficiency core.
*   **Benefit:** Keeps the P-cores strictly dedicated to math without being interrupted by orchestration, state transitions, or metadata parsing, maximizing instruction throughput and minimizing thermal throttling.

### C. Asynchronous Graph Execution and Batching

*   **Strategy:** When executing a compute graph, instead of a blocking `wait()` on a barrier, the `graph::processor` can be wrapped in a `co_sm`.
*   **Execution:**
    1.  The machine transitions to a `Dispatching` state.
    2.  The action queues the ARM kernel onto the execution context and issues a `co_await`.
    3.  If multiple sequences are in flight, the scheduler immediately picks up the next `co_sm` (next sequence) and queues its kernels.
    4.  The kernels execute concurrently on the hardware.
*   **Benefit:** Naturally achieves dynamic batching without violating the "no-queue invariant" inside the individual actors. Each actor is strictly linear and synchronous in its own perception, but the CPU execution units are kept saturated across multiple independent state machines.

### D. I/O and Paging Overlap

For models loaded via `mmap` that exceed physical RAM, page faults are disastrous for latency.

*   **Strategy:** Integrate a `madvise(MADV_WILLNEED)` or asynchronous read system into a custom coroutine scheduler.
*   **Execution:** The machine state transitions to `LoadingLayer`. It issues an async read for the tensor data and suspends. It is only resumed when the data is fully paged into RAM.
*   **Benefit:** Prevents the thread executing the tight ARM NEON loops from hard page-faulting and blocking the entire execution lane.

## 3. Proposed Implementation Phases

1.  **Phase 1: Scheduler and Allocator Binding**
    *   Implement an `emel::policy::coroutine_scheduler` that integrates with the existing EMEL `thread_pool` or AArch64 work-stealing queues.
    *   Configure `emel::policy::pooled_coroutine_allocator` in `src/emel/sm.hpp` for `co_sm` to guarantee zero-allocation async dispatches.

2.  **Phase 2: Graph Processor Refactoring**
    *   Migrate the `graph::processor` (or equivalent execution orchestrator) from `sml::sm` to `emel::co_sm`.
    *   Replace blocking waits on kernel execution with `co_await` on the kernel completion signals.

3.  **Phase 3: Kernel Integration**
    *   Update `src/emel/kernel/aarch64_*.cpp` routines. Ensure that the low-level kernels do not own orchestration logic, but simply provide completion callbacks that plug into the coroutine scheduler to resume the `co_sm`.

## Conclusion

By treating `co_sm` as a structured, allocation-free way to yield control, we can hide ARM memory latency and explicitly target big/LITTLE core clusters without breaking the clean, pure-predicate model of `stateforward.SML`. It transforms I/O bounds and compute bounds into pipelined, non-blocking state transitions.