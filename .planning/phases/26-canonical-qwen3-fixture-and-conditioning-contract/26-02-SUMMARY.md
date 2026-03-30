# Phase 26-02 Summary

## Outcome

Widened the formatter and conditioner contract from one flat prompt string to explicit structured
chat messages with explicit formatter option fields.

## Implementation

- Added `emel::text::formatter::chat_message` and replaced `format_request.input` with
  `std::span<const chat_message> messages` plus explicit `add_generation_prompt` and
  `enable_thinking` flags.
- Updated `emel::text::conditioner::event::prepare` and
  `emel::text::conditioner::action::dispatch_format` to carry the same structured-message payload
  and option booleans through the formatter seam.
- Reworked formatter and conditioner regression tests so the maintained contract is expressed in
  terms of message spans and explicit option propagation rather than flat input strings.

## Test Coverage

- `cmake --build build/zig --target emel_tests_bin -j4`
- `./build/zig/emel_tests_bin --test-case='*formatter*contract*,*conditioner*structured*' --no-breaks`

## Deviations From Plan

- `[Rule 3 - Blocking]` `src/emel/generator/actions.hpp` needed a temporary explicit bridge from
  the legacy flat generator prompt to a single structured user message so Wave 2 would keep the
  tree compiling before Wave 3 widens `generator::event::generate` itself.

## Result

- Formatter and conditioner now share one explicit structured-message ABI.
- Formatter option fields are explicit and test-covered.
- No shared Qwen-specific formatter helper was added to production code.
