# Phase 247 Validation

**Date:** 2026-07-04

Committed-state scoped gates: `test_with_coverage` PASS (changed-line 100%,
changed-branch 87.5%), `paritychecker` PASS (LFM2.5 generation through the
block-mapped addressing path), `emel_tests` 14/14; `bench_snapshot` FAIL 138 =
pre-existing reference-lane ggml SIGBUS (standing disposition, Phase 245).

Gate-usage lesson recorded: run coverage gates on COMMITTED state. With
uncommitted src edits the changed-line mapper unions the committed diff (HEAD
frame) with the worktree diff (current frame); insertions above committed hunks
shift attribution onto innocent lines (here: construct_at's ~80 phantom
exception-cleanup branches), making the branch threshold unattainable.
