# Phase 132 Context: Generator Tooling And Closeout Proof

## Goal

Cut over maintained tooling and prove the move is a pure ownership refactor.

## Scope

- CMake test source and shard filters.
- Coverage and quality-gate shard inference.
- Generation benchmark includes and namespaces.
- Paritychecker includes and namespaces.
- Embedded-size probe includes and namespaces.
- Compliance documentation references.
- Domain-boundary stale-root check.

## Risk

Changed-file coverage treats the moved generator headers as changed files even though the runtime
change is a namespace/path refactor. Existing generator coverage is below the current changed-file
threshold for those headers.
