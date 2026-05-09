# Phase 245 Validation

| ID | Plan | Requirement | Claim | Method | Evidence | Result |
|----|------|-------------|-------|--------|----------|--------|
| 245-01-01 | 01 | TNX-04 | Maintained public strategy contracts name `cooperative_async` explicitly. | source review, unit test | `strategy_kind::cooperative_async`, strategy helper source checks | pass |
| 245-01-02 | 01 | TNX-04 | `io/loader` reports cooperative async through a public unsupported-strategy route, not by including async actor internals. | source review, unit test | `strategy_cooperative_async`, `io loader fails closed for absent and explicit strategies` | pass |
| 245-01-03 | 01 | TNX-04 | `model/loader` can request cooperative async through public `io_loader` policy and receives deterministic unavailable evidence. | unit test | `model loader reports cooperative async strategy through public io loader` | pass |
| 245-01-04 | 01 | TNX-04 | Maintained tools parse/report cooperative async via the shared public helper and do not include async actor internals. | source guard test | `maintained tool cooperative async surface stays public-contract only` | pass |
