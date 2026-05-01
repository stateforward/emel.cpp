---
phase: 133-generator-sml-rule-repair
reviewed: 2026-04-28T21:27:30Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - src/emel/text/generator/sm.hpp
  - tests/text/generator/lifecycle_tests.cpp
findings:
  critical: 0
  warning: 1
  info: 0
  total: 1
status: issues_found
---

# Phase 133: Code Review Report

**Reviewed:** 2026-04-28T21:27:30Z
**Depth:** standard
**Files Reviewed:** 2
**Status:** issues_found

## Summary

Reviewed the generator state machine and lifecycle tests for Phase 133. The initialize wrapper no
longer contains the removed model/conditioner readiness branch, and the current implementation routes
missing dependencies through explicit SML guards, transitions, `action::reject_initialize`, and the
initialize error publication states. The focused generator/runtime ctest shard passed.

One test coverage issue remains: the added regression checks source spelling but does not execute the
exact missing-dependency path that the removed wrapper branch used to handle.

## Warnings

### WR-01: Missing Behavior Regression for Removed Initialize Branch

**File:** `tests/text/generator/lifecycle_tests.cpp:2021`
**Issue:** `generator_initialize_public_wrapper_has_no_runtime_branch` only verifies that the wrapper
source does not contain exact strings such as `if (` or `switch`. The existing invalid initialize
test at line 1038 uses a fully constructed fixture and nulls `tokenizer_sm`, so it does not exercise
the removed wrapper condition where `context_.model == nullptr` or `context_.conditioner == nullptr`.
A regression in the missing-dependency error publication path could still pass this phase's focused
test.
**Fix:** Add a behavioral regression that default-constructs the generator and verifies initialize
publishes `invalid_request` through the SML path:

```cpp
TEST_CASE("generator_initialize_rejects_missing_injected_dependencies") {
  emel::text::generator::sm generator{};
  emel::text::tokenizer::sm tokenizer{};
  std::array<emel::logits::sampler::fn, 1> samplers = {
      emel::logits::sampler::fn::from<sampler_select_argmax>(),
  };
  callback_tracker tracker{};
  emel::error::type error = emel::error::cast(emel::text::generator::error::none);
  emel::text::generator::event::initialize request{
    &tokenizer,
    tokenizer_bind_dispatch,
    tokenizer_tokenize_dispatch,
    std::span<emel::logits::sampler::fn>{samplers},
  };
  request.max_prompt_tokens = 8;
  request.max_generated_tokens = 4;
  request.max_blocks = 8;
  request.block_tokens = 4;
  request.error_out = &error;
  request.on_error = emel::callback<void(const emel::text::generator::events::initialize_error &)>(
      &tracker, on_initialize_error);

  CHECK_FALSE(generator.process_event(request));
  CHECK(generator.is(boost::sml::state<emel::text::generator::uninitialized>));
  CHECK(tracker.initialize_error_called);
  CHECK(error == emel::error::cast(emel::text::generator::error::invalid_request));
}
```

---

_Reviewed: 2026-04-28T21:27:30Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
