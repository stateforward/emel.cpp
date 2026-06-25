## Phase 234 Context

- **Parent workstream:** Milestone 63, issue `#63`, public dispatch tests follow-up after Phase 233.
- **Scope owned by Phase 234:** `TST-01` and `TST-02` only.
- **Primary target:** Add doctest evidence that staged-load public dispatch behavior is exercised through `process_event(...)` wrappers with explicit SML state inspection.
- **Note on strategy evidence:** This phase uses the maintained model-loader staged-load public surface where staged reads are selected via `io_strategy` policy and exercised through loader/tensor/io actors. Existing read-copy lifecycle paths remain relevant because they are still public staged-load dispatch paths in the same maintained machinery.
- **Out of scope:** Runtime behavior changes, actor action/guard rewrites, and unrelated milestone file churn.
