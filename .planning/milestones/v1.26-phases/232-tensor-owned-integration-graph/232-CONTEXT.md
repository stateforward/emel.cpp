# Phase 232 Context - Tensor-Owned Integration Graph

## Scope

Implement TNX-01/TNX-02/TNX-03/TNX-04 by integrating staged reads from
`model/tensor` through public `emel::io::staged_read` events and state machine
dispatch only.

## Source-backed constraints

- `model/tensor` remains lifecycle owner (bind/load/evict/residency).
- `io/staged_read` remains copy-only; it does not claim tensor residency.
- No actor reach-through into `io/staged_read` internals (`actions/detail/guards`).
- Per-dispatch status stays stack-scoped runtime event payload, not context mirrors.
- No synthetic fault knobs or test-only production fields.

## Existing integration patterns reused

- `request_mapped_load` path in `model/tensor` (public event + injected child actor).
- `request_read_load` path in `model/tensor` (explicit done/error terminal handling).

## Implementation focus

1. Add public tensor event surface for staged loading.
2. Inject `io::staged_read::sm` pointer via tensor context.
3. Add tensor runtime status/event wrappers for staged loading.
4. Route success/failure through explicit tensor states and `_done/_error` events.
5. Add focused tensor lifecycle tests proving:
   - unsupported child actor failure,
   - successful staged copy + tensor residency capture,
   - staged validation error propagation.
