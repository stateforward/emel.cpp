---
phase: 151
title: "Parity Dependency Manifest Emission"
requirements:
  - MANIFEST-01
  - MANIFEST-02
  - MANIFEST-03
status: complete
created: 2026-05-01
---

# Phase 151 Context

Phase 150 left paritychecker with explicit runner, engine registration, engine implementation,
tokenizer-engine, reference-support, and shared common source groups. Phase 151 adds the manifest
surface that later quality-gate work can consume without changing parity behavior yet.

## Inputs

- `tools/paritychecker/CMakeLists.txt`
- `tools/paritychecker/parity_engine.cpp`
- `tools/paritychecker/parity_runner.cpp`
- `tools/paritychecker/paritychecker_tests.cpp`
- `scripts/quality_gates.sh`
- `.planning/ROADMAP.md`
- `.planning/REQUIREMENTS.md`

## Constraints

- Manifest data must be conservative. Missing, stale, or uncertain data must trigger the relevant
  full parity gate.
- The manifest must remain deterministic and source-backed.
- This phase must not change tokenizer, GBNF, kernel, Jinja, or generation parity output behavior.
