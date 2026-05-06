---
phase: 215-tensor-owned-read-integration
status: complete
type: context
created: 2026-05-05T18:05:00Z
requirements:
  - TIO-01
  - TIO-02
---

# Phase 215 Context

Phase 215 closes the tensor-owned read integration gap from the milestone audit. The
read actor is already RTC-safe and source-buffer based; this phase gives
`model/tensor` a public read/copy request surface that dispatches through `emel/io`
while keeping tensor residency ownership in `model/tensor`.

## Source Truth

- `src/emel/model/tensor/events.hpp` already exposes public mmap request/release events.
  Read integration should follow that public event pattern instead of adding
  loader-private hooks.
- `src/emel/io/read/sm.hpp` owns read/copy validation and copied-byte publication.
- `model/tensor` owns resident/evicted lifecycle state and must commit only the
  caller-owned target buffer after read success.

## Non-Goals

- Do not wire maintained benchmark, paritychecker, or embedded probe read reporting here.
- Do not move low-level read/copy behavior into `model/loader`.
- Do not add staged/chunked, async, device, or mmap runtime changes.
