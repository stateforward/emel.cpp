# Feature Research

**Domain:** Bounded tensor byte loading under memory caps
**Researched:** 2026-05-07
**Confidence:** HIGH for scope boundaries; MEDIUM for user-visible chunk contract details

## Feature Landscape

### Table stakes

| Feature | Why expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Chunked read into fixed windows | Host cannot map or allocate full tensor span at once | MEDIUM | Must preserve deterministic ordering and byte offsets |
| Explicit completion / failure per stage | Operators need to reason about partial progress | MEDIUM | Model with `_done` / `_error`, not hidden retries |
| Same residency rules as read/mmap | Architecture consistency | LOW | Tensor owns target; strategy copies bytes only |

### Differentiators (when done well)

| Feature | Value | Complexity | Notes |
|---------|-------|------------|-------|
| Pure SML-visible stage graph | Auditable behavior; reviewable guards | MEDIUM | Avoids “smart” loops that encode routing |
| Bounded transient resources | Predictable FD/handle use per stage | MEDIUM | Align with v1.24/v1.25 lifetime discipline |

### Anti-features (reject for v1.26)

| Feature | Why tempting | Why problematic |
|---------|--------------|-----------------|
| Cooperative yield / coroutines | Natural for “streaming” | Explicitly out of scope unless separately approved; conflicts with stated RTC constraints |
| Implicit internal buffering pools | Simpler API | Hides ownership; risks allocation during dispatch |

## MVP (v1.26)

- Deterministic multi-stage copy into caller-owned tensor-visible buffer regions.
- Validation and unsupported paths fail closed like other I/O actors.
- Integration only through public tensor↔I/O boundary established in #60.

## Sources

- Archived requirement STAGED-01 precursor in v1.25 requirements archive
- Issue #63 wording (constrained-memory staged loading)

---
*Feature research for v1.26*
