# Phase 246 Context

The async I/O milestone is now wired through direct tensor async dispatch and public maintained
loader reporting. Phase 246 freezes the boundary before publication: coroutine implementation types
must stay out of public ABI and generic contracts, maintained tools must not reach into async actor
internals, behavior choice must remain in guards/transitions, and shipped synchronous I/O strategy
tests must remain in place.
