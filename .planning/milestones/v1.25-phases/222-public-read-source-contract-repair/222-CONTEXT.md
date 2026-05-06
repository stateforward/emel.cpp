# Phase 222: Public Read Source Contract Repair - Context

**Gathered:** 2026-05-06T04:46:52Z
**Status:** Ready for planning

<domain>
## Phase Boundary

Repair the v1.25 maintained read/copy source-byte contract so benchmark,
paritychecker, and embedded probe lanes no longer include or call actor-internal
`io/read/detail.hpp` helpers while preserving the public
`model/loader -> model/tensor -> io/loader -> io/read` runtime evidence path.

</domain>

<decisions>
## Implementation Decisions

### Source Contract Placement
- Use an EMEL-owned setup-time source-loading API under the `emel/io` ownership
  boundary, outside actor internals.
- Keep `io/read` as the Stateforward.SML actor copy boundary only.
- Preserve existing read error taxonomy so current `source_error` plumbing
  remains compatible.

### Guardrails
- Maintained lanes must not include `emel/io/read/detail.hpp`.
- Maintained lanes must not recreate tool-local `read_file_bytes` helpers.
- Parity and benchmark harnesses must avoid actor `actions.hpp`,
  `guards.hpp`, and `detail.hpp` reach-through.

### the agent's Discretion
Implementation details may follow existing header-only public facade patterns
when no new state machine is needed.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/emel/io/read/events.hpp` already models source spans and source errors
  as event-provided data.
- Maintained tools already pass loaded file bytes into model-loader requests.
- Existing guardrails in model-loader and paritychecker tests scan maintained
  tool sources for forbidden reach-through.

### Established Patterns
- Public facade headers such as `src/emel/model/any.hpp` expose allowed helper
  surfaces outside actor internals.
- Tool lanes use public model-loader done/error evidence for strategy reporting.

### Integration Points
- Generation benchmark, Sortformer fixture, embedded probe, and paritychecker
  load model/source bytes before dispatching into model-loader.
- Phase 222 must keep those lanes reporting `read_copy` only after the public
  runtime path executes.

</code_context>

<specifics>
## Specific Ideas

Use `emel::io::source::load_file_bytes` from `src/emel/io/source/any.hpp` as
the maintained setup-time source-byte API.

</specifics>

<deferred>
## Deferred Ideas

Final docs, snapshots, benchmark output publication, and milestone audit
closeout belong to Phase 223.

</deferred>
