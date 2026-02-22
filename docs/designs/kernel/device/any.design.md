# kernel/device/any architecture design (draft)

this document defines kernel/device/any. it is the device-level `sm_any` dispatcher
that selects a concrete device variant for graph computation.

## role
- dispatch graph scheduling to the device variant matching the target hardware.
- forward `event::schedule` to the active device variant.

## variants
- `kernel/device/cpu` — cpu execution (arch variants selected internally).
- `kernel/device/gpu` — gpu execution (backend variants: metal, vulkan, cuda).

## events (draft)
- `event::schedule` inputs: bound `graph`, device execution policy.
- `events::schedule_done` outputs: status, outputs written in-place to bound buffers.
- `events::schedule_error` outputs: error_out.

## state model (draft)
- mirrors the active variant's state model via `sm_any` dispatch.

## responsibilities
- select device variant at construction from configuration or hardware detection.
- forward `event::schedule` to the active variant.
