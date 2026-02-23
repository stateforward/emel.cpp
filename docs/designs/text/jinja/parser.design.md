---
title: jinja/parser architecture design
status: draft
---

# jinja/parser architecture design

this document defines the jinja parser actor. it replaces the legacy procedural recursive-descent parser with a hierarchical `boost::sml` state machine. because parsing is a one-time initialization step (e.g. during model loading), it utilizes standard heap allocation (`std::make_unique` and `std::vector`) for building the AST, adhering to the rule that one-time initialization may allocate.

## role
- act as a pure SML actor that consumes a stream of `jinja::token` objects.
- build an abstract syntax tree (`program`) using standard C++ heap allocations for AST nodes.
- enforce strict grammatical rules and nesting bounds using SML submachines and context stacks, eliminating procedural call stack recursion.

## architecture shift: token-pump and hierarchical SML
in the legacy design, parsing was a single procedural action (`run_parse`) that invoked a `recursive_descent_parser`. this violated the spirit of the actor model (by hiding complex control flow in an action) and relied heavily on the C++ call stack.

the new architecture uses the SML state machine as a true **Pushdown Automaton**:
1. **token events**: the external orchestrator (e.g. `text/conditioner`) calls the lexer to get an array of tokens, then iterates over them, dispatching a `token_event` for each one into the parser SM.
2. **hierarchical machines**: grammatical structures (like `expression` vs `statement`) are modeled as `boost::sml` composite states. transitioning into an expression simply pushes the SM into a submachine, naturally tracking nesting without C++ recursion.
3. **heap allocation**: since parsing only happens once during setup, actions construct AST nodes dynamically using `std::make_unique` and `std::vector`. this provides full flexibility and simplifies cleanup without violating inference hot-path constraints.

## events
- `event::start_parse`
  - inputs: `program_out` (destination AST) and optional callbacks.
  - outputs: initializes the parser's context stacks and transitions from `uninitialized` to `idle`.
- `event::token_received`
  - inputs: a single `jinja::token` and its metadata (type, string value, offset).
  - outputs: drives the grammar state machine. creates AST nodes on the heap, pushes/pops the parsing stack, or sets a syntax error.
- `event::end_parse`
  - inputs: none. signals the end of the token stream.
  - outputs: validates that all parsing stacks are empty (no unclosed tags) and finalizes the `program` structure. emits `parsing_done` or `parsing_error`.

## state model (hierarchical)

```text
uninitialized ──► idle (top-level text context)
                   │
                   ├──► (receive `{{`) ──► sm<expression_parser> ──► (receive `}}`) ──► idle
                   │
                   └──► (receive `{%`) ──► sm<statement_parser> ──► (receive `%}`) ──► idle
```

### submachine: `expression_parser`
handles the Pratt-style operator precedence and function calls.

```text
expecting_operand ──► (receive Identifier) ──► expecting_operator
                    ├──► (receive Numeric) ──► expecting_operator
                    └──► (receive `(`) ──► sm<expression_parser> (nested)

expecting_operator ──► (receive `+`) ──► expecting_operand
                     ├──► (receive `(`) ──► sm<call_args_parser>
                     └──► (receive `]`) ──► (pop stack)
```

## responsibilities

1. **one-time heap allocation**:
   - AST nodes (like `binary_expression` or `identifier`) are dynamically allocated via `std::make_unique`.
   - `ast_list` uses `std::vector<std::unique_ptr<ast_node>>`.
   - memory is released naturally when the `program` struct is destroyed.
   
2. **explicit bounds**:
   - the SML context must track `stack_depth`. if an expression nests too deeply (e.g. `(((((...))))`), the `push_stack` action must trigger a transition to `errored` to prevent malicious nesting attacks.

3. **deterministic error recovery**:
   - if a syntax error occurs (e.g. an unexpected token type for the current state), the machine immediately transitions to `errored` and sets `error_pos` in the context. subsequent `token_received` events are ignored or result in `sml::unexpected_event` routing.
