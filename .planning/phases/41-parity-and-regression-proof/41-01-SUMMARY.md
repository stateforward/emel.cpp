---
phase: 41-parity-and-regression-proof
plan: 01
completed: 2026-04-03
status: implemented
---

# Phase 41 Summary

Phase 41 closed the maintained Bonsai correctness contract on the parity surface without collapsing
every maintained model onto one global reference lane. The maintained fixture registry now carries
explicit `(fixture, engine, repository, ref, contract)` truth, and the parity harness resolves
fixtures against the currently built reference lane instead of pretending one repo/ref pair can
prove Qwen, Liquid, and Bonsai simultaneously.

The maintained generation parity surface now splits cleanly by contract. Qwen and Bonsai use live
reference generation on their own truthful lanes, while Liquid keeps its append-only baseline lane.
That keeps the older maintained anchors protected without forcing them onto the Prism fork, and it
lets Bonsai publish the exact pinned Prism identity on the same formatter contract the runtime
actually uses.

The full scripted parity gate now executes isolated lanes instead of one shared monolithic build.
That is the key phase outcome: EMEL and the reference stay separated per lane, reviewers can audit
the exact fixture/repo/ref tuple, and maintained regression coverage now includes Bonsai without
breaking the prior anchors.
