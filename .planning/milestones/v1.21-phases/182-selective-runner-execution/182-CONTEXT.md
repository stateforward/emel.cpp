# Phase 182: Selective Runner Execution - Context

**Gathered:** 2026-05-02
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase executes only impacted maintained parity and benchmark runners when impact resolution is
trustworthy. It does not add new parity or benchmark domains; it wires selection into existing
maintained entrypoints.

</domain>

<decisions>
## Implementation Decisions

### Parity Execution
- Extend `scripts/paritychecker.sh` with `--runner=<name>` and `--mode=<name>` filters.
- Keep no-filter behavior as the full paritychecker test target.
- Normalize `gbnf` to the maintained `gbnf_parser` runner name.
- Run selected doctest cases through the maintained `paritychecker_tests` binary.

### Benchmark Execution
- Preserve existing benchmark selective execution through `scripts/bench.sh --suite=<runner>`.
- Continue using full benchmark fallback when benchmark manifest impact is broad or uncertain.
- Keep selected benchmark runner names visible in quality-gate output.

### the agent's Discretion
Use doctest test-case filters for parity runner execution because the maintained paritychecker test
binary already owns the runner-specific validation cases.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/paritychecker.sh` already configures and builds the maintained paritychecker tool with
  zig.
- `paritychecker_tests` contains runner-specific doctest case names for tokenizer, GBNF parser,
  kernel, Jinja, and generation parity.
- `scripts/bench.sh --suite=<name>` already supports suite-filtered maintained benchmark runs.

### Established Patterns
- Shell entrypoints validate required tools before execution.
- The maintained test and benchmark binaries remain the execution boundary.
- Quality-gate output is written to stderr for decision traceability.

### Integration Points
- `scripts/paritychecker.sh`
- `scripts/quality_gates.sh`
- `scripts/bench.sh`
- `tools/bench/quality_gates_tests.cpp`

</code_context>

<specifics>
## Specific Ideas

Selective parity execution must remain an entrypoint feature, not a tool-only test scaffold.

</specifics>

<deferred>
## Deferred Ideas

No new parity runner families were added in this phase.

</deferred>
