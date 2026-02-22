# kernel/device/gpu/cuda architecture design (draft)

this document defines kernel/device/gpu/cuda. it schedules graph instructions on
NVIDIA GPUs via CUDA.

## role
- schedule and compute graph instructions via CUDA kernels.
- available on platforms with NVIDIA CUDA support.

## events (draft)
- `event::schedule` inputs: bound `graph`, gpu execution policy.
- `events::schedule_done` outputs: status, outputs written in-place to bound buffers.
- `events::schedule_error` outputs: error_out.

## state model (draft)
- `uninitialized` -> `binding` -> `idle`.
- `idle` -> `scheduling` -> (`done` | `errored`).
- unexpected events route to `unexpected`.

## responsibilities
- map opcodes to CUDA kernel launches.
- manage CUDA stream and device buffer bindings for graph tensors.
- synchronize stream completion before returning done.
