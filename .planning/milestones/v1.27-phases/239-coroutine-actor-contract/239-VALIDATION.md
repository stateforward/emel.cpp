# Phase 239 Validation

| ID | Plan | Requirement | Claim | Method | Evidence | Result |
|----|------|-------------|-------|--------|----------|--------|
| 239-01-01 | 01 | CO-01 | `co_sm` coroutine actor contract is explicit in source-of-truth rules. | source review | `docs/rules/sml.rules.md` section `10.1` | pass |
| 239-01-02 | 01 | CO-01 | Operational guidance is aligned with workspace agent rules. | source review | `AGENTS.md`, `CLAUDE.md` coroutine actor paragraphs | pass |
| 239-01-03 | 01 | CO-01 | Contract forbids hidden mailboxes, retained stack data, stored callbacks, hidden behavior selection, dispatch allocation, and public ABI coroutine leaks. | source review | rule text + focused `rg` check | pass |
| 239-01-04 | 01 | CO-01 | Scoped project quality gate remains green for changed files. | command | changed-file `scripts/quality_gates.sh` exit `0` | pass |
