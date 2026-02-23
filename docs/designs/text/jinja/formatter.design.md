---
title: jinja/formatter architecture design
status: draft
---

# jinja/formatter architecture design

this document defines the jinja formatter actor (formerly `renderer::sm`). it evaluates a parsed `jinja::program` against a set of variables and emits formatted text. it enforces strict memory and computation bounds using statically provable limits (e.g., stack depth, scope counts) to ensure safe, zero-allocation execution within the inference hot path.

## role
- act as a pure SML actor that evaluates a Jinja AST (`program`).
- emit UTF-8 formatted text into caller-provided buffers (`format_io`) without heap allocation.
- prevent denial-of-service (DoS) from malicious or infinitely recursive templates by enforcing strict execution bounds.

## architecture shift: renaming and bounding
previously named the `renderer`, this component has been renamed to `formatter` to align with its role within the `text/conditioner` pipeline (where a `text/formatter` prepares prompt strings before tokenization, while a `text/renderer` is responsible for decoding tokens back into domain output).

the core architectural challenge in the formatter is its evaluation loop:
```text
eval_stmt ──► eval_expr ──► write_output ──► eval_stmt
```
because these are **anonymous transitions** (transitions taken automatically without an external event), SML's Run-To-Completion (RTC) semantics will loop them until the machine reaches quiescence. if a template contains an infinite loop or unbounded recursion (e.g., `{% macro inf() %}{{ infinite() }}{% endmacro %}{{ inf() }}`), the SML dispatch would never return, locking the thread.

### the stack depth solution
to satisfy the `sml.rules.md` requirement that *"anonymous transition graphs MUST be acyclic or MUST have a statically provable bound on firings per top-level event"*, the formatter enforces strict hierarchical limits rather than a flat instruction counter.

1. **call stack depth**: the context tracks `call_depth`. every time a macro or function is evaluated, the depth increments. if it exceeds `k_max_call_depth`, the action sets an error flag.
2. **scope bounds**: the context tracks `scope_count`. every `{% for %}` loop or block pushes a new variable scope. if this exceeds `k_max_scopes`, it errors out.
3. **step limits (fail-safe)**: as a final safeguard against massive (but not necessarily deep) templates, a `steps_remaining` counter is decremented on every statement evaluation. if it hits 0, execution halts.

these limits ensure the anonymous SML loops are mathematically guaranteed to terminate and return control to the orchestrator.

## events
- `event::format`
  - inputs: `program` (the parsed AST), `globals` (injected variables/context), `output` buffer pointer and `output_capacity`, `error_out`, and optional synchronous callbacks.
  - outputs: executes the AST, populates the `output` buffer, updates `output_length` and `output_truncated` flags, and invokes the callback before returning.

## state model

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

- `setup` — binds the `program`, initializes the scope stack and bounds counters, and prepares the initial statements.
- `eval_stmt` — evaluates control flow (`if`, `for`, `set`) and enqueues expression evaluation.
- `eval_expr` — resolves variables, performs math, applies filters, and executes macros.
- `write_output` — writes evaluated expression values into the caller's output buffer.
- `format_decision` — inspects the context's error state (e.g., bounds exceeded, invalid types, buffer full).
- `done` / `errored` — terminal states for the RTC chain.

## responsibilities & constraints

1. **zero-allocation rendering**:
   - the formatter must never use `new`, `malloc`, or `std::string` during evaluation.
   - temporary strings (like the result of `replace` or `join` filters) are stored in a fixed-size `string_buffer` within the SML context. if this buffer fills up (`k_max_string_bytes`), the formatter immediately errors.
   - output is written sequentially to the caller-provided `format_io` buffer.

2. **graceful truncation**:
   - if the evaluated text exceeds the caller's `output_capacity`, the formatter must not crash. it stops writing, sets `output_truncated = true` in the event payload, and gracefully finishes the evaluation or errors out depending on the strictness policy.

3. **deterministic error routing**:
   - any bounds violation (stack depth, scope count, step limit) or runtime error (division by zero, missing filter) sets `ctx.phase_error`. the state machine guards inspect this and cleanly route to `format_decision` -> `errored` without throwing C++ exceptions.
