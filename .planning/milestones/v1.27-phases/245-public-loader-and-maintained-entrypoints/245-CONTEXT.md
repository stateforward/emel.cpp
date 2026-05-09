# Phase 245 Context

Phase 245 connects the tensor-owned async work to maintained public loader reporting without
claiming broader async runtime execution prematurely.

The existing maintained path routes model loading through `model/loader` and `io/loader` strategy
policies. `io/loader` already dispatches read/copy and staged-read via public actor events, and it
reports unsupported strategies explicitly. The cooperative async tensor runtime now exists, but
batch/public maintained loader execution still needs a separate performance and evidence phase
before it can be claimed as an executed async path.

Therefore this phase adds a public strategy token and reporting path that is explicit and
fail-closed. Tools may request/report `cooperative_async` through public strategy policy helpers,
but maintained tools must not include async actor internals or reach into async events/actions.
