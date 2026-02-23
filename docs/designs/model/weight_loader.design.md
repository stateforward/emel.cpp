---
title: model/weight_loader architecture design
status: draft
---

# model/weight_loader architecture design

this document defines the `model/weight_loader` actor. it acts as the bridge between parsed file metadata and the physical memory backing the model's weight tensors, supporting both zero-copy memory mapping (`mmap`) and streaming loads.

## role
- act as a pure SML actor that consumes a list of parsed tensor offsets and binds them to physical storage.
- abstract the complexities of OS-level memory mapping (e.g., `mmap` on Linux/macOS, `MapViewOfFile` on Windows) away from the mathematical execution pipeline.
- support dynamic loading strategies, including chunked streaming over a network or split-file architectures, without exposing those details to the `generator`.
- handle graceful fallback and cleanup if memory boundaries are exceeded.

## architecture shift: decoupled hardware binding
in `llama.cpp`, the logic to open file descriptors, map memory, and construct tensors was tightly coupled to the parser.

in `emel`, the `model/weight_loader` operates solely on the structural metadata provided by the `model/loader`. it does not know what a GGUF header is. it simply receives a directive: "Tensor X is a `Q4_K` array of size `[4096, 4096]` located at byte offset `0x10000`." The weight loader then orchestrates the OS calls to make that data accessible to the `kernel` backends.

## events
- `event::load_weights`
  - inputs: a parsed structural map of the model (tensor offsets and sizes) and hardware policy preferences (e.g., `use_mmap = true`).
  - outputs: executes the loading strategy, verifying the physical data is accessible and aligned, and invokes a completion callback.

## state model

```text
uninitialized ──► initialized
                      │
initialized ──► selecting ──► strategy_decision
                                  ├──► (mmap) ──► initializing ──► loading_mmap ──► load_decision
                                  │
                                  └──► (stream) ──► loading_streamed ──► load_decision
                                                                               │
                 (done | errored) ◄── cleaning_up ◄── validating ◄─────────────┘
```

- `selecting` — determines the optimal backend loading strategy based on the hardware context and user flags.
- `initializing` — (mmap only) ensures the file size and alignment support zero-copy mapping.
- `loading_mmap` — delegates to the OS to map the file into virtual memory.
- `loading_streamed` — (alternative) allocates host/device memory and reads the file progressively (e.g., for split models or systems without `mmap` support).
- `validating` — confirms that all required tensors are successfully backed by physical addresses or valid file descriptors.
- `cleaning_up` — ensures temporary file handles or intermediate buffers are closed before returning control.
- `done` — weights are bound and ready.

## responsibilities & constraints

1. **strategy abstraction**:
   - the weight loader must seamlessly hide whether a tensor's data lives in RAM, VRAM, or on-disk via `mmap`. the downstream `graph/graph` simply sees a `buffer` pointer.
   
2. **zero-allocation hot path**:
   - this actor is invoked exactly once during model initialization. while it may allocate buffers during `loading_streamed`, it is never invoked during the `compute` phase, preserving `emel`'s strict inference rules.

3. **deterministic cleanup**:
   - if a memory map fails (e.g., `ENOMEM`), the actor immediately routes to `errored`, executing cleanup actions to release partial mappings before signaling failure to the `model/loader`.
