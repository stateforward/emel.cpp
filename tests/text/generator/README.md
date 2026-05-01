# Text Generator Test Surface

Phase 141 classifies text-generator tests by maintenance role:

- `lifecycle_tests.cpp`, initializer lifecycle tests, prefill lifecycle tests, paritychecker lanes,
  and benchmark lanes are maintained behavior proof. They must drive public events or public tool
  entrypoints and must not reach through actor internals.
- `action_guard_tests.cpp` is a component-private SML rule regression surface. It intentionally
  includes private action/guard headers to exercise transition predicates and action side effects.
- `detail_tests.cpp` is a component-private numeric and binding regression surface. It intentionally
  includes private generator detail helpers until those helpers are extracted to a kernel-owned
  surface.

Milestone closeout and maintained runtime claims must cite the public lifecycle/parity/benchmark
proof, not the component-private regression files.
