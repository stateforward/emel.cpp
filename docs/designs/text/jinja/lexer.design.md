---
title: jinja/lexer architecture design
status: draft
---

# jinja/lexer architecture design

this document defines the jinja lexer actor. it replaces the legacy procedural `while`-loop tokenizer with a pure `boost::sml` state machine, acting as a character-pump automaton to cleanly identify boundaries, handle escapes, and emit tokens without complex nested loops or procedural lookahead spaghetti.

## role
- act as a pure SML actor that consumes a stream of `char` (or `event::char_received`).
- emit a sequence of `jinja::token` objects via synchronous callbacks or by accumulating them in a context array.
- gracefully handle malformed templates (e.g., unterminated strings) using strict state transitions rather than complex procedural bounds checking.

## architecture shift: the character-pump automaton
in the legacy design, the lexer was a procedural class that ran a massive `while` loop, manually checking indices (`pos + 1`, `pos + 2`) and advancing pointers. this is fragile and violates the actor model paradigm.

the new architecture turns the lexer into a true **Finite State Machine**:
1. **character events**: the orchestrator iterates over the `std::string_view` of the template and dispatches an `event::char_received` for each character (plus an `event::eof` at the end).
2. **state-driven tokenization**: the lexer starts in a `text` state. if it receives `{`, it transitions to a `potential_block_open` state. if the next char is `%`, it emits a `text` token (if any was accumulated) and transitions to the `in_statement` state.
3. **zero-allocation tracking**: the context holds a fixed-size `scratch_buffer` for accumulating the current token's characters (or simply tracks the `start_pos` and `current_pos` indices to yield `std::string_view` tokens).

## events
- `event::start_lex`
  - inputs: optional synchronous callbacks for emitting tokens.
  - outputs: initializes context pointers and transitions to `idle`.
- `event::char_received`
  - inputs: `char c`, `size_t pos`.
  - outputs: drives the state machine. may accumulate characters, transition states, or emit completed tokens.
- `event::eof`
  - inputs: none.
  - outputs: forces the emission of the final token (if valid). if the machine is in an unterminated state (like `in_string`), it transitions to `errored`.

## state model

```text
uninitialized ──► idle (accumulating text)
                    │
                    ├──► (receive `{`) ──► potential_open
                    │                        ├──► (receive `%`) ──► in_statement
                    │                        ├──► (receive `{`) ──► in_expression
                    │                        └──► (receive `#`) ──► in_comment
                    │
                    └──► (receive EOF) ──► done
```

### sub-states (e.g., `in_expression`)
when inside an expression or statement, the machine parses numbers, identifiers, strings, and operators:

```text
in_expression ──► (receive `"`) ──► in_string ──► (receive `"`) ──► in_expression
              ├──► (receive digit) ──► in_number ──► (receive non-digit) ──► in_expression
              ├──► (receive letter) ──► in_identifier ──► (receive non-word) ──► in_expression
              └──► (receive `}`) ──► potential_close ──► (receive `}`) ──► idle
```

## responsibilities & constraints

1. **index-based string views**:
   - to avoid heap-allocating `std::string` for every token, the lexer context should merely track the `start_pos` of the current token.
   - when a token is complete, it emits a `jinja::token` containing the `token_type` and the `std::string_view` derived from the original template string.

2. **escape sequence handling**:
   - when inside `in_string`, receiving `\\` transitions to an `in_escape` state. the next character is decoded and the machine returns to `in_string`. if `EOF` is received during `in_escape` or `in_string`, the lexer transitions to `errored`.

3. **deterministic error recovery**:
   - invalid syntax (e.g., an illegal character in a variable name or an unterminated string) causes an immediate transition to `errored`.
   - the error position is recorded in the context, and subsequent `char_received` events are ignored via `unexpected_event` routing.
