---
title: jinja architecture design
status: draft
---

# jinja architecture design

this document defines the jinja templating subsystem. it provides a high-performance, allocation-bounded jinja2 template engine used for chat formatting and prompt construction within the `text/formatter` and `text/conditioner` pipelines.

## role
- provide a lightweight, dependency-free jinja2 implementation tailored for LLM chat templates.
- execute entirely within the run-to-completion (RTC) actor model.
- enforce strict memory and computation bounds during both parsing and formatting to prevent denial-of-service from malicious templates.

## architecture: explicit decoupling
the jinja subsystem is divided into three distinct phases to ensure strict separation of concerns and allow offline compilation of templates in the future:
1. **`lexer`**: a pure, synchronous utility that converts a raw template string into a bounded array of tokens.
2. **`parser::sm`**: an SML actor that consumes a token stream and builds an abstract syntax tree (`program`).
3. **`formatter::sm`**: an SML actor that evaluates a `program` against a set of `globals` and emits formatted UTF-8 bytes.

## lexer
the lexer is a pure utility (not an SML actor) that takes a `std::string_view` and returns a `lexer_result` containing a sequence of tokens.
- **responsibilities**: handle whitespace trimming (`{%-`, `-%}`), detect blocks, comments, and expressions, and reject invalid syntax (like unterminated strings).
- **invariants**: operates sequentially without heap allocating individual tokens, typically returning a flat array/vector of bounded tokens.

## parser (parser::sm)
the parser is an SML actor that converts the token stream into a structured `program` (AST).

### events
- `event::parse`
  - inputs: `template_text`, `program_out` (AST destination), `error_out`, and optional synchronous callbacks (`dispatch_done`, `dispatch_error`).
  - outputs: tokenizes and parses the template text, populates the provided `program`, and invokes the appropriate callback before returning.

### state model
```text
uninitialized ──► initialized
                      │
initialized ──► parse_decision ──► (done | errored)
  ▲                                    │
  └────────────────────────────────────┘
```
- `initialized` — awaiting parse intent.
- `parse_decision` — synchronously runs the internal parsing logic (via recursive descent) and branches based on success/failure.
- `done` — parsing complete and AST populated.
- `errored` — syntax error or invalid argument.
- unexpected events route to `unexpected`.

## formatter (formatter::sm)
the formatter is an SML actor that evaluates a parsed `program` and emits the final formatted text. it supports variables, control flow (if/for/macro), filters, and tests.

### events
- `event::format`
  - inputs: `program` (parsed AST), `globals` (object containing user variables and model metadata), caller-provided `output` buffer + capacity, `output_length`, `error_out`, and optional synchronous callbacks (`dispatch_done`, `dispatch_error`).
  - outputs: evaluates the AST, writes UTF-8 bytes directly to the `output` buffer, updates `output_length`, and invokes the callback.

### state model
```text
uninitialized ──► initialized
                      │
initialized ──► setup ──► eval_stmt ──► format_decision ──► (done | errored)
  ▲                         │  ▲                                │
  │                         ▼  │                                │
  │                     eval_expr ──► write_output              │
  │                                     │                       │
  └─────────────────────────────────────┴───────────────────────┘
```
- `initialized` — idle state awaiting format intent.
- `setup` — initializes the evaluation context and seeds the initial statement work.
- `eval_stmt` — steps through statements in the current scope.
- `eval_expr` — evaluates pending expressions (variable resolution, math, filters).
- `write_output` — emits the evaluated values to the output buffer.
- `format_decision` — checks if the evaluation completed successfully or hit an error/limit.
- `done` — formatting complete.
- `errored` — hit an execution limit (e.g., max scope depth, out of buffer capacity, division by zero) or invalid arguments.
- unexpected events route to `unexpected`.

### responsibilities & constraints
1. **bounded evaluation**: the formatter strictly limits evaluation depth, scope counts, call stack limits, and array/object sizes to prevent infinite loops or memory exhaustion (enforced by constants like `k_max_scopes`, `k_max_array_items`, `k_max_callables`).
2. **zero-allocation formatting**: outputs are written directly to caller-provided buffers (`format_io`). intermediate strings are not dynamically heap-allocated during template evaluation; they are either string views of the original template or written into a fixed-size scratch buffer in the SML context.
3. **synchronous callbacks**: results are communicated via `dispatch_done` and `dispatch_error` callbacks before the SML dispatch returns, eliminating the need for the caller to inspect the state machine's internal context.
