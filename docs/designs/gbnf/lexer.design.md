---
title: gbnf/lexer architecture design
status: draft
---

# gbnf/lexer architecture design

this document defines the gbnf lexer actor. it replaces the legacy procedural `while`-loop lexical analysis currently embedded within the recursive-descent parser, acting as a pure `boost::sml` character-pump automaton to emit GBNF tokens safely and without heap allocation.

## role
- act as a pure SML actor that consumes a stream of `char` (or `event::char_received`).
- emit a sequence of `gbnf::token` objects (e.g. `rule_name`, `literal`, `character_class`, `operator`).
- gracefully handle malformed grammars (e.g., unterminated strings or classes) using strict state transitions.

## architecture shift: separating lexical analysis
in the legacy design (`src/emel/gbnf/parser/detail.hpp`), lexical analysis (reading strings, skipping spaces, finding `::=`) was tightly coupled with grammatical parsing in a single massive procedural class. this violates the actor model paradigm and makes boundary checking fragile.

the new architecture splits the work into a true **Finite State Machine**:
1. **character pump**: the orchestrator iterates over the `std::string_view` of the grammar and dispatches an `event::char_received` for each character.
2. **state-driven tokenization**: the lexer transitions between states like `idle`, `in_string`, `in_char_class`, `expecting_definition` based on input characters.
3. **zero-allocation string views**: the context holds no dynamic memory. it merely tracks the `start_pos` of the current token. upon completion of a token, it yields a `gbnf::token` with a `std::string_view` into the original text.

## events
- `event::start_lex`
  - inputs: optional synchronous callbacks for emitting tokens.
  - outputs: initializes context pointers and transitions to `idle`.
- `event::char_received`
  - inputs: `char c`, `size_t pos`.
  - outputs: drives the state machine. may accumulate bounds, transition states, or emit completed tokens.
- `event::eof`
  - inputs: none.
  - outputs: forces the emission of the final token (if valid). if the machine is in an unterminated state, it transitions to `errored`.

## state model (draft)

```text
uninitialized ──► idle (skipping whitespace)
                    │
                    ├──► (receive `"`) ──► in_string ──► (receive `"`) ──► emit `literal`
                    │
                    ├──► (receive `[`) ──► in_char_class ──► (receive `]`) ──► emit `char_class`
                    │
                    ├──► (receive `a-z`) ──► in_identifier ──► (receive non-word) ──► emit `rule_name`
                    │
                    └──► (receive `:` ──► `:` ──► `=`) ──► emit `definition_operator`
```

## responsibilities & constraints

1. **index-based string views**:
   - to avoid heap-allocating `std::string` for every token, the lexer context should merely track the `start_pos` of the current token.
   
2. **escape sequence handling**:
   - when inside `in_string` or `in_char_class`, receiving `\\` transitions to an `in_escape` state. the next character is decoded and the machine returns to its previous state. if `EOF` is received during `in_escape`, the lexer transitions to `errored`.

3. **deterministic error recovery**:
   - invalid syntax causes an immediate transition to `errored`. subsequent `char_received` events are ignored via `unexpected_event` routing.
