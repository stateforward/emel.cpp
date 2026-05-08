# Pitfalls Research

**Domain:** Adding staged loading to strict SML I/O actors
**Researched:** 2026-05-07
**Confidence:** HIGH

## Critical pitfalls

1. **Coroutine-shaped control flow** — Implementing “resume later” chunk I/O as cooperative scheduling inside the actor. **Avoid:** Keep each dispatch RTC-complete; stage progress via explicit states/events, not async suspension (unless separately approved as a different milestone).

2. **Context as scratch for stage position** — Storing file offset, remaining length, or request pointers in `context` during a single top-level dispatch chain in ways that violate EMEL context rules. **Avoid:** Use typed internal events for phase handoff; keep context to durable actor configuration only.

3. **Behavior selection in `detail` or actions** — Choosing chunk size, fallback path, or “try again” inside helpers. **Avoid:** Guards + explicit transitions only.

4. **Regressing read/mmap** — Touching shipped strategies when adding staged paths. **Avoid:** Additive strategy component; narrow edits in tensor/loader only through public integration points.

5. **False evidence in tools** — Reporting staged strategy in benchmarks when maintainers still use bulk read. **Avoid:** Same discipline as VAL-04 in v1.25.

6. **Per-logit style completion chains** — Using one completion transition per byte or per tiny slice. **Avoid:** Bounded number of internal phases per top-level dispatch; bulk memcpy in data-plane helpers.

## Prevention strategy

- Phase 1 establishes guardrails and empty/fail-closed actor before real I/O.
- Early domain-boundary checks extend existing `emel/io` patterns.
- Closeout requires source-backed tests through `process_event`, not direct detail calls.

## Sources

- `docs/rules/sml.rules.md` RTC and completion rules
- v1.25 audit lessons on read actor RTC repair (Phase 214.1 narrative)

---
*Pitfalls research for v1.26*
