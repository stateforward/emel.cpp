# Phase 131 Context: Move Generator Actor And Tests

## Goal

Move the generator parent actor, initializer child, prefill child, and tests under text-domain
ownership while preserving request and output behavior.

## Scope

- `src/emel/generator/**` moves to `src/emel/text/generator/**`.
- `tests/generator/**` moves to `tests/text/generator/**`.
- Namespaces move from `emel::generator` to `emel::text::generator`.
- Includes move from `emel/generator/**` to `emel/text/generator/**`.

## Non-Goals

- No sampling, formatter, tokenizer, kernel, parity, or benchmark semantic changes.
- No new public C ABI.
- No new model family or fixture support.
