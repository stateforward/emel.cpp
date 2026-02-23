---
title: gbnf/parser architecture design
status: draft
---

# gbnf/parser architecture design

this document defines the gbnf parser actor. it replaces the legacy procedural `recursive_descent_parser` (from `llama.cpp`) with a hierarchical `boost::sml` state machine. it consumes a stream of `gbnf::token` objects to safely populate a `gbnf::grammar` without using the C++ call stack for recursion.

## role
- act as a pure SML actor that consumes a stream of `gbnf::token` objects.
- build the `gbnf::grammar` struct, which is a flat Data-Oriented Design (DOD) array of `gbnf::element` blocks.
- enforce strict structural rules and grouping bounds using SML submachines, eliminating procedural recursion limits.

## architecture shift: token-pump and hierarchical SML
in the legacy design (`src/emel/gbnf/parser/detail.hpp`), grammatical parsing was implemented as a single procedural class that manually advanced `const char *` pointers and called functions like `parse_sequence` and `parse_alternates`. this violated the spirit of the actor model by hiding complex control flow inside a single synchronous `run_parse` action.

the new architecture uses the SML state machine as a true **Pushdown Automaton**:
1. **token events**: the external orchestrator (or a wrapper `gbnf::compiler`) calls the `gbnf::lexer` to generate an array of tokens, then iterates over them, dispatching a `token_event` for each one into the parser SM.
2. **hierarchical machines**: grammatical structures (like `sequence` vs `alternates` or nested parentheses `()`) are modeled as `boost::sml` composite states. transitioning into a grouped sub-expression simply pushes the SM into a submachine, tracking parenthesis depth automatically via SML states rather than C++ recursion.
3. **bump allocation**: the target `gbnf::grammar` struct already uses a fixed-size `std::array<element, 65536>`. the SML context uses this as a bump allocator. actions simply write `gbnf::element` objects into the array and advance the `element_count`, failing cleanly with `EMEL_ERR_INVALID_ARGUMENT` if the array is full.

## events
- `event::start_parse`
  - inputs: `grammar_out` (the destination flat array struct) and optional callbacks.
  - outputs: initializes the parser's symbol table and transitions to `idle`.
- `event::token_received`
  - inputs: a single `gbnf::token`.
  - outputs: drives the grammar state machine. appends `gbnf::element`s to the `grammar` bump allocator, manages group submachines, or flags a syntax error.
- `event::end_parse`
  - inputs: none.
  - outputs: validates that all rules are correctly defined, no parentheses remain unclosed, and emits `parsing_done`.

## state model (hierarchical)

```text
uninitialized ──► idle (expecting rule name)
                   │
                   ├──► (receive `RuleName`) ──► expecting_definition (`::=`)
                   │
                   ├──► (receive `::=`) ──► sm<alternates_parser> ──► (receive `
` or `EOF`) ──► idle
```

### submachine: `alternates_parser`
handles `|` separating sequences, as well as `()` groupings and repetition operators (`*`, `+`, `?`, `{m,n}`).

```text
parsing_sequence ──► (receive Literal) ──► append `element`
                 ├──► (receive CharClass) ──► append `element`
                 ├──► (receive RuleRef) ──► append `element`
                 ├──► (receive `*`, `+`, `?`) ──► mutate previous `element` into repetition nodes
                 ├──► (receive `(`) ──► sm<alternates_parser> (nested sub-group)
                 │
                 └──► (receive `|`) ──► append `alt` node ──► parsing_sequence
```

## responsibilities

1. **flat array compilation**:
   - unlike Jinja which builds a tree of heap-allocated objects, GBNF compiles directly into a flat DOD array of `gbnf::element` nodes.
   - the SML actions use the target `grammar` object as a bump allocator (`grammar->elements[grammar->element_count++] = ...`).
   
2. **explicit recursion bounds**:
   - the SML context must track `nesting_depth` (the number of nested `()` groups). if an expression nests beyond `k_max_nesting_depth` (e.g. 32), the `push_group` action must trigger a transition to `errored`.
   - the context tracks `symbol_table` sizes. if the number of rules exceeds `k_max_gbnf_rules`, the machine transitions to `errored`.

3. **deterministic error recovery**:
   - if a syntax error occurs (e.g., an unexpected `|` without a sequence), the machine immediately transitions to `errored`. subsequent `token_received` events are routed to `unexpected_event`.
