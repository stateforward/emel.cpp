---
title: model/loader architecture design
status: draft
---

# model/loader architecture design

this document defines the `model/loader` actor. it is the top-level orchestrator responsible for safely transforming an on-disk model format (like GGUF) into an executable memory topology ready for inference.

## role
- coordinate the entire model initialization process without blocking the inference hot path.
- maintain strict format agnosticism (it doesn't know what a GGUF is, only that a parser satisfies the interface).
- delegate format-specific metadata extraction to a parser (e.g., `parser/gguf::sm`).
- delegate hardware mapping to a dedicated loader (e.g., `model/weight_loader::sm`).
- validate that the parsed tensors structurally match the mathematical requirements of the model's reported architecture.

## architecture shift: format agnosticism
in legacy `llama.cpp`, the logic to parse a GGUF file, allocate VRAM, memory map the file, and build the tensor computation graph was completely entangled inside `llama_model_load`.

in `emel`, the `model/loader` orchestrates these distinct steps using dependency injection:
1. **Metadata Parsing:** delegates to a `parser` actor to extract hyper-parameters, vocabulary, and tensor offsets.
2. **Weight Mapping:** delegates to the `weight_loader` actor to map those tensor offsets to physical memory or disk blocks.
3. **Graph Assembly:** verifies that the mapped tensors perfectly align with the `graph/assembler`'s mathematical requirements before transitioning to `done`.

## events
- `event::load`
  - inputs: a URI or file handle, model options, and optional synchronous callbacks.
  - outputs: drives the multi-stage initialization pipeline. emits `loading_done` upon success, ensuring the model is ready to be bound to a `generator`.

## state model (linear orchestration)

```text
uninitialized ──► initialized
                      │
initialized ──► mapping_parser ──► parsing ──► loading_weights ──► mapping_layers ──► validating_structure ──► validating_architecture ──► (done | errored)
  ▲                                                                                                                                            │
  └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

- `initialized` — waiting for a load command.
- `mapping_parser` — selects the appropriate parser backend (e.g., `parser/gguf`) based on the file signature.
- `parsing` — delegates to the parser to extract structural metadata (dimensions, tensor names/offsets) and vocabulary.
- `loading_weights` — delegates to the `model/weight_loader` to bind the tensor metadata to actual hardware memory (e.g., `mmap` or streaming buffers).
- `mapping_layers` — organizes the flat list of parsed tensors into hierarchical layer blocks (e.g., matching `blk.0.attn_q` to Layer 0).
- `validating_structure` / `validating_architecture` — mathematically verifies that the loaded tensors perfectly satisfy the model's forward pass requirements (preventing runtime kernel crashes).
- `done` — the model is safely loaded and ready for inference.
- unexpected events route to `errored`.

## responsibilities & constraints

1. **strict pipeline enforcement**:
   - the loader enforces that parsing happens before mapping, and mapping happens before execution. if any stage fails (e.g., `parser/gguf` encounters a corrupted header), the loader immediately transitions to `errored` and cleans up.
   
2. **delegated operations**:
   - the `model/loader` itself must never call `mmap` or read binary headers directly. its actions purely exist to orchestrate sub-machines and handle the business logic of state transitions.
