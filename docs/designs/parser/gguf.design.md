---
title: parser/gguf architecture design
status: draft
---

# parser/gguf architecture design

this document defines the parser/gguf actor. it is responsible for safely decoding the binary GGUF file format, extracting metadata, and providing a structured view of the model's architecture and tensors to the higher-level `model/loader`.

## role
- act as a one-time initialization actor that consumes a binary buffer (or file handle) of a GGUF file.
- parse the GGUF header, key-value pairs (hyper-parameters, architecture, vocabulary), and tensor metadata (name, shape, offset, quantization type).
- validate the integrity and alignment of the binary data before it is mapped to hardware.
- facilitate the zero-copy `mmap` of weight data by yielding tensor offsets without copying the actual weights into RAM.

## architecture shift: decoupled metadata extraction
in legacy `llama.cpp`, GGUF parsing, memory mapping, and backend buffer allocation were entangled within `llama_model_load`. 

in `emel`, the `parser/gguf` is **strictly a metadata parser**. it does not allocate hardware buffers, it does not execute `mmap` itself, and it does not know about GPU or CPU backends. it simply validates the binary structure and yields a clean, parsed metadata struct to the orchestrating `model/loader`.

## events
- `event::parse_model`
  - inputs: a pointer to the raw file buffer or an I/O interface, file size, and optional synchronous callbacks (`dispatch_done`, `dispatch_error`).
  - outputs: kicks off the sequential parsing pipeline. populates the caller-provided context with parsed architecture, hyperparameters, and tensor maps.

## state model (linear pipeline)

because model parsing is a one-time initialization operation (explicitly permitted by `AGENTS.md`), the parser utilizes a linear chain of internal transitions to segment the work logically:

```text
uninitialized ──► initialized
                      │
initialized ──► parsing_architecture ──► mapping_architecture ──► parsing_hparams ──► parsing_vocab ──► mapping_tensors ──► (done | errored)
  ▲                                                                                                                            │
  └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

- `initialized` — idle state awaiting parse intent.
- `parsing_architecture` — reads the GGUF magic, version, and the core `general.architecture` key.
- `mapping_architecture` — translates the string-based architecture (e.g., "llama", "rwkv") into an internal enum.
- `parsing_hparams` — extracts the mathematical dimensions (e.g., `n_embd`, `n_head`, `n_layer`).
- `parsing_vocab` — reads the token arrays, scores, and merges, feeding them into the vocabulary struct.
- `mapping_tensors` — iterates over the tensor infos, accumulating their names, shapes, and binary offsets.
- `done` — parsing complete; the `model/loader` can now use the metadata to `mmap` the file and build the DAG.
- `errored` — hit an invalid GGUF magic, version mismatch, or corrupted data offset.

## responsibilities & constraints

1. **zero-allocation metadata traversal**:
   - the parser navigates the binary structure using standard pointers and offsets.
   - it does not heap-allocate intermediate strings for keys. it uses `std::string_view` mapped directly over the binary buffer.

2. **mmap delegation**:
   - the parser only computes `file_offset` for each tensor.
   - the higher-level `model/loader` takes this metadata, invokes the OS `mmap`, and uses the `graph/assembler` to create the `leaf` (weight) tensors pointing to those memory-mapped regions.

3. **strict binary validation**:
   - all reads must be strictly bounds-checked against the file size.
   - string lengths and array counts in the GGUF format must not overflow buffer capacities, transitioning to `errored` with `EMEL_ERR_INVALID_ARGUMENT` upon violation.
