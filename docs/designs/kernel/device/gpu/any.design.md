# kernel/device/gpu/any architecture design (draft)

this document defines kernel/device/gpu/any. it is the gpu-level `sm_any` dispatcher
that selects a concrete gpu backend variant for kernel scheduling.

## role
- dispatch graph scheduling to the gpu backend variant matching the target hardware.

## variants
- `kernel/device/gpu/metal` — Apple Metal (macOS/iOS).
- `kernel/device/gpu/vulkan` — Vulkan (cross-platform).
- `kernel/device/gpu/cuda` — NVIDIA CUDA.

## events (draft)
- `event::schedule` inputs: bound `graph`, gpu execution policy.
- `events::schedule_done` outputs: status, outputs written in-place to bound buffers.
- `events::schedule_error` outputs: error_out.

## state model (draft)
- mirrors the active variant's state model via `sm_any` dispatch.

## responsibilities
- detect available gpu backend at construction and select variant.
- forward `event::schedule` to the active variant.
- own gpu context and command queue lifecycle; inject into variants by reference.
