# Phase 246 Validation

| ID | Plan | Requirement | Claim | Method | Evidence | Result |
|----|------|-------------|-------|--------|----------|--------|
| 246-01-01 | 01 | GRD-01 | Coroutine task/scheduler/awaitable types do not leak into public ABI or generic model/generator contracts. | source guard test | `phase 246 coroutine types stay out of public abi and generic contracts` | pass |
| 246-01-02 | 01 | GRD-02 | Maintained tools and model-loader code avoid async actor internals. | source guard test | `phase 246 maintained entrypoints avoid async actor internals` | pass |
| 246-01-03 | 01 | GRD-03 | Async actions/detail do not choose runtime behavior; guards/transitions retain those choices. | source guard test | `phase 246 async actions and detail do not choose behavior` | pass |
| 246-01-04 | 01 | GRD-04 | Shipped mmap, read/copy, staged-read, loader, tensor, and model-loader regression tests remain public dispatch tests. | source guard test, focused doctest runs | `phase 246 shipped io strategy regression tests remain public`, focused strategy test runs | pass |
