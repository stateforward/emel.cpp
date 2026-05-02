---
phase: 173-sml-migration-evidence-reconstruction
verified: 2026-05-01T20:24:00Z
status: passed
score: 5/5 requirements verified
---

# Phase 173 Verification Report

**Phase Goal:** Reconstruct source-backed evidence for DEP-01, DEP-02, DEP-03, SRC-01, and
SRC-02.  
**Verified:** 2026-05-01T20:24:00Z  
**Status:** passed

## Requirement Evidence

| Requirement | Status | Evidence |
|-------------|--------|----------|
| DEP-01 | passed | `git log -p -n 2 -- cmake/sml_version.cmake` shows the prior `EMEL_BOOST_SML_GIT_TAG` value `02cbea023f035185cfb400e6015c981f9b946bae` and current `EMEL_SML_GIT_TAG` value `4a7109b5dd4aae40e78304e3ac03440ccc35031e`. |
| DEP-02 | passed | `cmake/sml_version.cmake` defines `EMEL_SML_GIT_REPOSITORY` as `https://github.com/stateforward/sml.cpp` and `EMEL_SML_GIT_TAG` as `4a7109b5dd4aae40e78304e3ac03440ccc35031e`; root CMake and docsgen CMake consume that shared pin. |
| DEP-03 | passed | Active legacy include/namespace scans returned no matches; there are no active compatibility exceptions to document for project-owned code. |
| SRC-01 | passed | Active include scan reports `<stateforward/sml.hpp>` and `stateforward/sml/utility/*` usage in `src`, `tests`, `tools/docsgen`, and docs/rules. |
| SRC-02 | passed | Active namespace scan reports `stateforward::sml` usage in source, tests, tools, and rules; legacy `boost::sml` scan returned no matches. |

## Automated Checks

- `git log -p -n 2 -- cmake/sml_version.cmake`
- `rg -n '#\s*include\s*[<"](boost/sml|sml\.hpp|boost/sml\.hpp)' src include tests tools docs scripts .codex/get-shit-done cmake CMakeLists.txt || true`
- `rg -n '\bboost::sml\b|using\s+namespace\s+boost::sml|#\s*include\s*[<"]boost/sml' src include tests tools docs scripts .codex/get-shit-done cmake CMakeLists.txt || true`
- `rg -n '#\s*include\s*[<"]stateforward/sml|stateforward::sml' src include tests tools docs/rules tools/docsgen CMakeLists.txt cmake`

## Notes

This phase is evidence reconstruction only. It does not claim SRC-03 logger/dispatch-table behavior
or VAL guardrail completion.

