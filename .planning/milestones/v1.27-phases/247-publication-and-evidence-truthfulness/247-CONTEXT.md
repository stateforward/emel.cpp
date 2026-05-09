# Phase 247 Context

Phase 247 closes the v1.27 milestone by aligning docs, snapshots, benchmark evidence, and planning
state with the source-backed async I/O implementation.

The direct async runtime path exists at the tensor boundary (`model/tensor` dispatching
`emel::io::async::event::load_window`). The maintained model-loader/tool path exposes
`cooperative_async` as a public strategy token but intentionally reports it as unsupported until a
published maintained async execution path can run end to end. Performance evidence must distinguish
those two facts: direct tensor async exists, but maintained benchmark/tool paths must not label an
unsupported public strategy as an async performance result.
