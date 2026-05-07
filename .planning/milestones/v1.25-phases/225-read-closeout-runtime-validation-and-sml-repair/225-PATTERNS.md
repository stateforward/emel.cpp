# Phase 225: read-closeout-runtime-validation-and-sml-repair - Pattern Map

**Mapped:** 2026-05-06
**Files analyzed:** 23
**Analogs found:** 23 / 23
**Rule sources:** `AGENTS.md`, `docs/rules/sml.rules.md`, `docs/rules/cpp.rules.md`

## File Classification

| New/Modified File | Role | Data Flow | Closest Rule-Safe Analog | Match Quality |
|-------------------|------|-----------|--------------------------|---------------|
| `src/emel/model/loader/events.hpp` | model | request-response | `src/emel/model/loader/events.hpp`, `src/emel/io/loader/events.hpp` | role-match |
| `src/emel/model/loader/guards.hpp` | middleware | request-response | `src/emel/model/loader/guards.hpp`, `src/emel/io/loader/guards.hpp` | role-match |
| `src/emel/model/loader/actions.hpp` | service | event-driven | `src/emel/io/loader/actions.hpp`, `src/emel/model/tensor/actions.hpp` | partial; reject legacy loop |
| `src/emel/model/loader/sm.hpp` | controller | event-driven | `src/emel/model/loader/sm.hpp`, `src/emel/model/tensor/sm.hpp` | role-match |
| `src/emel/io/loader/events.hpp` | model | request-response | `src/emel/io/loader/events.hpp` | role-match |
| `src/emel/io/loader/guards.hpp` | middleware | request-response | `src/emel/io/loader/guards.hpp` | exact |
| `src/emel/io/loader/actions.hpp` | service | event-driven | `src/emel/io/loader/actions.hpp` | role-match |
| `src/emel/io/loader/sm.hpp` | controller | event-driven | `src/emel/io/loader/sm.hpp` | exact |
| `src/emel/io/read/events.hpp` | model | file-I/O | `src/emel/io/read/events.hpp` | role-match |
| `src/emel/io/read/guards.hpp` | middleware | file-I/O | `src/emel/io/read/guards.hpp` | exact |
| `src/emel/io/read/actions.hpp` | service | file-I/O | `src/emel/io/read/actions.hpp` | exact |
| `src/emel/io/read/sm.hpp` | controller | file-I/O | `src/emel/io/read/sm.hpp` | exact |
| `tests/model/loader/lifecycle_tests.cpp` | test | request-response | `tests/model/loader/lifecycle_tests.cpp` | exact |
| `tests/io/loader/lifecycle_tests.cpp` | test | request-response | `tests/io/loader/lifecycle_tests.cpp` | exact |
| `tests/io/read/lifecycle_tests.cpp` | test | file-I/O | `tests/io/read/lifecycle_tests.cpp` | exact |
| `tests/model/tensor/lifecycle_tests.cpp` | test | request-response | `tests/model/tensor/lifecycle_tests.cpp` | role-match |
| `.planning/ROADMAP.md` | config | batch | `.planning/ROADMAP.md` | exact |
| `.planning/REQUIREMENTS.md` | config | batch | `.planning/REQUIREMENTS.md` | exact |
| `.planning/milestones/v1.25-ROADMAP.md` | config | batch | `.planning/ROADMAP.md` | role-match |
| `.planning/milestones/v1.25-REQUIREMENTS.md` | config | batch | `.planning/REQUIREMENTS.md` | role-match |
| `.planning/v1.25-MILESTONE-AUDIT.md` | config | batch | `.planning/v1.25-MILESTONE-AUDIT.md` | exact |
| `.planning/phases/225-read-closeout-runtime-validation-and-sml-repair/225-VALIDATION.md` | config | batch | `.planning/phases/225-read-closeout-runtime-validation-and-sml-repair/225-VALIDATION.md` | exact |
| `.planning/phases/225-read-closeout-runtime-validation-and-sml-repair/225-VERIFICATION.md` | config | batch | `.planning/milestones/v1.25-phases/223-read-closeout-truth-and-validation-reconciliation/223-VERIFICATION.md` | role-match |

## Pattern Assignments

### `src/emel/model/loader/events.hpp` (model, request-response)

**Rule-safe analog:** `src/emel/model/loader/events.hpp`

**Imports and ownership pattern** (lines 7-13):
```cpp
#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/io/loader/events.hpp"
#include "emel/model/data.hpp"
#include "emel/model/loader/errors.hpp"
#include "emel/model/tensor/events.hpp"
#include "emel/model/tensor/sm.hpp"
```

**Request-owned scratch pattern** (lines 46-68):
```cpp
struct load {
  emel::model::data &model_data;
  parse_model_fn &parse_model;
  emel::model::tensor::sm *tensor_loader = nullptr;
  emel::io::loader::sm *io_loader = nullptr;
  emel::io::loader::event::strategy_kind io_strategy =
      emel::io::loader::event::strategy_kind::none;
  std::span<emel::model::tensor::effect_request> effect_requests = {};
  std::span<emel::model::tensor::effect_result> effect_results = {};
  emel::callback<void(const events::load_done &)> on_done = {};
  emel::callback<void(const events::load_error &)> on_error = {};
};
```

Use this for new batch read/copy spans: keep spans and same-RTC carriers in the request/runtime event, not persistent `context`.

**Outcome evidence pattern** (lines 148-164):
```cpp
struct load_done {
  const event::load &request;
  uint64_t bytes_total = 0;
  uint64_t bytes_done = 0;
  bool used_mmap = false;
  emel::io::loader::event::strategy_kind used_io_strategy =
      emel::io::loader::event::strategy_kind::none;
};

struct load_error {
  const event::load &request;
  emel::io::loader::event::strategy_kind requested_io_strategy =
      emel::io::loader::event::strategy_kind::none;
  emel::io::loader::event::strategy_kind used_io_strategy =
      emel::io::loader::event::strategy_kind::none;
};
```

### `src/emel/model/loader/guards.hpp` (middleware, request-response)

**Rule-safe analog:** `src/emel/model/loader/guards.hpp`

**Runtime choice in guards** (lines 173-215):
```cpp
struct io_strategy_none {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return ev.request.io_strategy ==
           emel::io::loader::event::strategy_kind::none;
  }
};

struct tensor_plan_done_with_io_strategy_with_loader {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return tensor_plan_done_raised{}(ev) && io_strategy_present{}(ev) &&
           io_loader_present{}(ev);
  }
};
```

**I/O result classification** (lines 218-280):
```cpp
struct io_load_done_all {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return ev.io_events != nullptr && ev.io_events->load_done.raised &&
           !ev.io_events->load_error.raised &&
           ev.io_events->load_done.done_count ==
               ev.io_events->load_done.expected_count;
  }
};

struct io_load_error_raised {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return ev.io_events != nullptr && ev.io_events->load_error.raised;
  }
};
```

For Phase 225, extend this pattern with batch guards such as `io_load_batch_done_all` and explicit error-kind predicates. Do not move strategy/result choice into actions or detail helpers.

### `src/emel/model/loader/actions.hpp` (service, event-driven)

**Rule-safe analogs:** `src/emel/io/loader/actions.hpp`, `src/emel/model/tensor/actions.hpp`

**Cross-actor single dispatch pattern** from `src/emel/io/loader/actions.hpp` (lines 100-120):
```cpp
struct effect_dispatch_read_tensor {
  void operator()(const detail::load_tensor_runtime &ev,
                  context &ctx) const noexcept {
    emel::io::read::event::read_tensor_request request{
        .tensor_id = ev.request.tensor.tensor_id,
        .file_index = ev.request.tensor.file_index,
        .file_offset = ev.request.tensor.file_offset,
        .byte_size = ev.request.tensor.byte_size,
        .file_path = ev.request.tensor.file_path,
        .source_buffer = ev.request.tensor.source_buffer,
        .source_buffer_bytes = ev.request.tensor.source_buffer_bytes,
        .source_error = ev.request.tensor.source_error,
        .target_buffer = ev.request.tensor.target,
        .target_buffer_bytes = ev.request.tensor.target_bytes,
    };
    emel::io::read::event::read_tensor read{request};
    static_cast<void>(ctx.io_read->process_event(read));
  }
};
```

Use this only as a single child-dispatch shape. For model-loader, dispatch one batch request to `io/loader`; do not dispatch one child event per tensor.

**Same-RTC result-carrier pattern** from `src/emel/model/tensor/actions.hpp` (lines 619-639):
```cpp
struct effect_attempt_request_read_load_dispatch {
  void operator()(const detail::request_read_load_runtime &ev,
                  context &ctx) const noexcept {
    emel::io::read::event::read_tensor_request inner{
        .tensor_id = ev.request.tensor_id,
        .file_index = ev.request.file_index,
        .file_offset = ev.request.file_offset,
        .byte_size = ev.request.byte_size,
        .file_path = ev.request.file_path,
        .source_buffer = ev.request.source_buffer,
        .source_buffer_bytes = ev.request.source_buffer_bytes,
    };
    emel::io::read::event::read_tensor read{inner};
    static_cast<void>(ctx.io_read->process_event(read, ev.status.io_read));
    ev.status.buffer = ev.status.io_read.target_buffer;
    ev.status.buffer_bytes = ev.status.io_read.bytes_copied;
  }
};
```

Use a same-RTC batch result object if model-loader needs aggregate status without callbacks. The result object must live in the wrapper/runtime event, not machine context.

**Rejected legacy pattern** from `src/emel/model/loader/actions.hpp` (lines 249-272):
```cpp
struct effect_dispatch_io_loads {
  void operator()(const event::load_runtime &ev, context &) const noexcept {
    const uint32_t effect_count = ev.tensor_events.plan_done.effect_count;
    effect_reset_io_load_events(*ev.io_events, effect_count);

    for (uint32_t index = 0u; index < effect_count; ++index) {
      ...
      emel::io::loader::event::load_tensor load{tensor, policy};
      static_cast<void>(ev.request.io_loader->process_event(load));
    }
  }
};
```

Do not reuse this core pattern. It is the audited SML readiness gap: per-tensor child orchestration is hidden in an action loop.

### `src/emel/model/loader/sm.hpp` (controller, event-driven)

**Rule-safe analog:** `src/emel/model/loader/sm.hpp`

**Transition-table structure** (lines 42-49, 104-143):
```cpp
struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;
    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
        sml::state<request_decision> <= *sml::state<ready> + sml::event<event::load_runtime>
          / action::begin_load

      , sml::state<state_tensor_bind_decision> <= sml::state<loading_tensors>
          + sml::completion<event::load_runtime>
          / action::effect_dispatch_tensor_bind_storage
      , sml::state<state_tensor_plan_dispatch> <= sml::state<state_tensor_bind_decision>
          + sml::completion<event::load_runtime> [ guard::tensor_bind_done_raised{} ]
      , sml::state<state_io_load_dispatch> <= sml::state<state_tensor_plan_decision>
          + sml::completion<event::load_runtime>
          [ guard::tensor_plan_done_with_io_strategy_with_loader{} ]
      , sml::state<state_tensor_apply_dispatch> <= sml::state<state_io_load_decision>
          + sml::completion<event::load_runtime> [ guard::io_load_done_all{} ]
          / action::effect_mark_io_strategy_used
```

Keep destination-first rows, visual sections, and completion-carried runtime event data. Add any Phase 225 batch states here, with guards deciding batch availability and outcome.

**Wrapper-local carrier pattern** (lines 394-422):
```cpp
bool process_event(const event::load &ev) {
  event::load_ctx ctx{};
  events::io_load_done io_load_done{};
  events::io_load_error io_load_error{};
  event::io_phase_events io_events{
      .load_done = io_load_done,
      .load_error = io_load_error,
  };
  event::load_runtime runtime{ev, ctx, ..., &io_events};
  const bool accepted = base_type::process_event(runtime);
  return accepted && ctx.err == emel::error::cast(error::none);
}
```

Use wrapper-local stack carriers for per-dispatch data. Do not put request pointers, counts, error phase flags, or output pointers in `action::context`.

### `src/emel/io/loader/events.hpp` (model, request-response)

**Rule-safe analog:** `src/emel/io/loader/events.hpp`

**Public request and outcome shape** (lines 19-51, 58-69):
```cpp
enum class strategy_kind : uint8_t {
  none = 0u,
  mapped_file = 1u,
  read_copy = 2u,
  external_buffer = 3u,
};

struct tensor_load_span {
  int32_t tensor_id = 0;
  uint16_t file_index = 0u;
  uint64_t file_offset = 0u;
  uint64_t byte_size = 0u;
  const void *source_buffer = nullptr;
  uint64_t source_buffer_bytes = 0u;
  void *target = nullptr;
  uint64_t target_bytes = 0u;
};

struct load_tensor_done {
  const event::load_tensor &request;
  event::strategy_kind strategy = event::strategy_kind::none;
};
```

Batch events should preserve this shape: request-owned spans, explicit strategy, `_done` and `_error` events, and no owning dynamic containers.

### `src/emel/io/loader/guards.hpp` (middleware, request-response)

**Rule-safe analog:** `src/emel/io/loader/guards.hpp`

**Strategy routing guards** (lines 37-66):
```cpp
struct strategy_read_copy {
  bool operator()(const detail::load_tensor_runtime &ev) const noexcept {
    return ev.request.policy.strategy == event::strategy_kind::read_copy;
  }
};

struct strategy_read_copy_with_actor {
  bool operator()(const detail::load_tensor_runtime &ev,
                  const action::context &ctx) const noexcept {
    return strategy_read_copy{}(ev) && read_actor_present{}(ctx);
  }
};
```

Add batch equivalents here, for example `strategy_read_copy_batch_with_actor`. The guard name and transition table must make the route explicit.

### `src/emel/io/loader/actions.hpp` (service, event-driven)

**Rule-safe analog:** `src/emel/io/loader/actions.hpp`

**Callback status recording** (lines 76-97):
```cpp
namespace read_callbacks {
inline void
on_read_done(void *object,
             const emel::io::read::events::read_tensor_done &ev) noexcept {
  auto *status = static_cast<detail::runtime_status *>(object);
  status->err = emel::error::cast(error::none);
  status->ok = true;
  status->bytes_copied = ev.bytes_copied;
  status->buffer = ev.target_buffer;
}
}
```

For batch mode, prefer one same-RTC aggregate result over per-span callbacks if it avoids callback arrays or dynamic storage.

### `src/emel/io/loader/sm.hpp` (controller, event-driven)

**Rule-safe analog:** `src/emel/io/loader/sm.hpp`

**Explicit strategy route** (lines 28-62):
```cpp
return sml::make_transition_table(
  //------------------------------------------------------------------------------//
  // Loading strategy boundary. Concrete strategies are explicit future routes.
    sml::state<state_request_decision> <= *sml::state<state_ready>
      + sml::event<detail::load_tensor_runtime>
      [ guard::tensor_span_valid{} ]
      / action::effect_begin_load_tensor
  , sml::state<state_read_dispatch_decision> <=
      sml::state<state_request_decision>
      + sml::completion<detail::load_tensor_runtime>
      [ guard::strategy_read_copy_with_actor{} ]
      / action::effect_dispatch_read_tensor
  , sml::state<state_unsupported_strategy_error_decision> <=
      sml::state<state_request_decision>
      + sml::completion<detail::load_tensor_runtime>
      / action::effect_mark_unsupported_strategy
);
```

Use this as the primary analog for `io/loader` batch routing: add a separate batch runtime event and route it with explicit guarded transitions.

### `src/emel/io/read/events.hpp` (model, file-I/O)

**Rule-safe analog:** `src/emel/io/read/events.hpp`

**Public read/copy contract** (lines 21-39):
```cpp
struct read_tensor_request {
  int32_t tensor_id = 0;
  uint16_t file_index = 0u;
  uint64_t file_offset = 0u;
  uint64_t byte_size = 0u;
  std::string_view file_path = {};
  const void *source_buffer = nullptr;
  uint64_t source_buffer_bytes = 0u;
  emel::error::type source_error = emel::error::cast(error::none);
  void *target_buffer = nullptr;
  uint64_t target_buffer_bytes = 0u;
};
```

**Same-RTC result pattern** (lines 71-77):
```cpp
struct read_tensor_result {
  bool accepted = false;
  bool ok = false;
  emel::error::type err = emel::error::cast(error::none);
  uint64_t bytes_copied = 0u;
  void *target_buffer = nullptr;
};
```

Batch requests should reuse this contract per span, but the public batch event should carry a span over preallocated request/result arrays rather than own a dynamic container.

### `src/emel/io/read/guards.hpp` (middleware, file-I/O)

**Rule-safe analog:** `src/emel/io/read/guards.hpp`

**Validation chain predicates** (lines 12-24, 91-104, 160-189):
```cpp
struct request_span_valid {
  bool operator()(const detail::read_tensor_runtime &ev,
                  const action::context &) const noexcept {
    return ev.request.request.byte_size > 0u &&
           static_cast<bool>(ev.request.on_done);
  }
};

struct target_buffer_valid {
  bool operator()(const detail::read_tensor_runtime &ev,
                  const action::context &) const noexcept {
    return ev.request.request.target_buffer != nullptr &&
           ev.request.request.target_buffer_bytes >=
               ev.request.request.byte_size;
  }
};

struct file_read_succeeded {
  bool operator()(const detail::read_tensor_runtime &ev,
                  const action::context &) const noexcept {
    const auto &request = ev.request.request;
    return request.source_error == emel::error::cast(error::none) &&
           request.file_offset <= request.source_buffer_bytes &&
           request.byte_size <=
               request.source_buffer_bytes - request.file_offset;
  }
};
```

For batch read/copy, keep aggregate validation outcomes in guards and transitions. Per-span bounds checks may be data-plane work only after the batch route has already been selected.

### `src/emel/io/read/actions.hpp` (service, file-I/O)

**Rule-safe analog:** `src/emel/io/read/actions.hpp`

**Already-selected copy action** (lines 92-104):
```cpp
struct effect_mark_read_tensor_done {
  void operator()(const detail::read_tensor_runtime &ev,
                  context &) const noexcept {
    const auto &request = ev.request.request;
    const auto *source =
        static_cast<const unsigned char *>(request.source_buffer);
    auto *target = static_cast<unsigned char *>(request.target_buffer);
    std::memcpy(target, source + request.file_offset,
                static_cast<std::size_t>(request.byte_size));
    ev.status.err = emel::error::cast(error::none);
    ev.status.bytes_copied = ev.request.request.byte_size;
    ev.status.ok = true;
  }
};
```

This is safe because it executes the already-chosen read/copy path. A batch copy loop may live in `io/read` only as bounded data-plane iteration after guards/transitions have selected batch read/copy behavior.

### `src/emel/io/read/sm.hpp` (controller, file-I/O)

**Rule-safe analog:** `src/emel/io/read/sm.hpp`

**Explicit validation chain** (lines 40-169):
```cpp
return sml::make_transition_table(
  //------------------------------------------------------------------------------//
    sml::state<state_request_decision> <= *sml::state<state_ready>
      + sml::event<detail::read_tensor_runtime>
      / action::effect_begin_read_tensor
  , sml::state<state_file_path_decision> <=
      sml::state<state_request_decision>
      + sml::completion<detail::read_tensor_runtime>
      [ guard::request_span_valid{} ]
  , sml::state<state_done_callback> <=
      sml::state<state_file_read_decision>
      + sml::completion<detail::read_tensor_runtime>
      [ guard::file_read_succeeded{} ]
      / action::effect_mark_read_tensor_done
);
```

**Result-capture wrapper** (lines 324-337):
```cpp
bool process_event(const event::read_tensor &ev,
                   events::read_tensor_result &result) {
  detail::read_attempt_status status{};
  event::read_tensor captured{ev.request};
  captured.on_done = {nullptr, detail::ignore_read_tensor_done};
  captured.on_error = {nullptr, detail::ignore_read_tensor_error};
  detail::read_tensor_runtime runtime{captured, status};
  const bool accepted = base_type::process_event(runtime);
  result.accepted = accepted;
  result.ok = status.ok;
  result.err = status.err;
  result.bytes_copied = status.bytes_copied;
  result.target_buffer = ev.request.target_buffer;
  return accepted && status.ok;
}
```

Use this result-capture wrapper as the analog for batch result capture. Do not expose mutable internal carriers through public API types.

## Test Pattern Assignments

### `tests/model/loader/lifecycle_tests.cpp` (test, request-response)

**Rule-safe analog:** `tests/model/loader/lifecycle_tests.cpp`

**Maintained read/copy success test** (lines 1020-1065):
```cpp
TEST_CASE("model loader loads read/copy tensors through maintained actors") {
  emel::io::read::sm read_actor{};
  emel::io::loader::sm io_loader{{.io_read = &read_actor}};
  ...
  request.io_loader = &io_loader;
  request.io_strategy = emel::io::loader::event::strategy_kind::read_copy;
  ...
  CHECK(accepted);
  CHECK(owner.used_io_strategy ==
        emel::io::loader::event::strategy_kind::read_copy);
  CHECK(tensor_loader.effect_results[0].handle == target.data());
  CHECK(std::memcmp(target, "cdef", 4u) == 0);
}
```

Extend this test to prove the repaired batch path still traverses public `model/loader -> io/loader -> io/read` dispatch and still reports `used_io_strategy` only after success.

**Guardrail scan pattern** (lines 1391-1413):
```cpp
TEST_CASE("maintained tool read copy surfaces avoid direct io read events") {
  const std::array tool_sources{
      "tools/bench/generation_bench.cpp",
      "tools/bench/diarization/sortformer_fixture.hpp",
      "tools/embedded_size/emel_probe/main.cpp",
      "tools/paritychecker/parity_engines.cpp",
  };
  for (const auto *source_path : tool_sources) {
    const std::string source = read_text_file(repo_root() / source_path);
    CHECK(source.find("emel/io/read/detail.hpp") == std::string::npos);
    CHECK(source.find("emel::io::source::load_file_bytes") !=
          std::string::npos);
  }
}
```

Add a guardrail that fails if `src/emel/model/loader/actions.hpp` still contains an action-loop `io_loader->process_event(...)` pattern.

### `tests/io/loader/lifecycle_tests.cpp` (test, request-response)

**Rule-safe analog:** `tests/io/loader/lifecycle_tests.cpp`

**Read/copy route test** (lines 184-218):
```cpp
TEST_CASE("io loader dispatches read/copy requests through the read actor") {
  emel::io::read::sm read_actor{};
  emel::io::loader::sm loader{{.io_read = &read_actor}};
  ...
  CHECK(loader.process_event(request));
  CHECK(owner.strategy == emel::io::loader::event::strategy_kind::read_copy);
  CHECK(owner.buffer == target.data());
  CHECK(target[0] == 'c');
  CHECK(loader.is(stateforward::sml::state<emel::io::loader::state_ready>));
}
```

Add batch-route tests here for success, unsupported strategy/read actor absent, and recovery to ready.

**Strategy guardrail scan** (lines 305-318):
```cpp
CHECK(actions_source.find("mmap(") == std::string::npos);
CHECK(actions_source.find("pread(") == std::string::npos);
CHECK(actions_source.find("std::ifstream") == std::string::npos);
CHECK(sm_source.find("strategy_read_copy") != std::string::npos);
CHECK(sm_source.find("strategy_read_copy_with_actor") != std::string::npos);
CHECK(sm_source.find("effect_dispatch_read_tensor") != std::string::npos);
```

Extend with batch route names and continue rejecting file-I/O and mmap implementation details in `io/loader`.

### `tests/io/read/lifecycle_tests.cpp` (test, file-I/O)

**Rule-safe analog:** `tests/io/read/lifecycle_tests.cpp`

**Public process-event and state inspection** (lines 86-107):
```cpp
TEST_CASE("io read copies requested bytes into caller-owned target buffer") {
  emel::io::read::sm strategy{};
  uint8_t target[4]{};
  constexpr char source[] = "abcdef";
  auto request = make_request("emel_io_read_success.bin", target, 3u);
  request.source_buffer = source;
  request.source_buffer_bytes = sizeof(source) - 1u;
  request.file_offset = 2u;
  request.byte_size = 3u;
  emel::io::read::event::read_tensor read_request{request};
  REQUIRE(strategy.process_event(read_request));
  CHECK(std::memcmp(target, "cde", 3u) == 0);
  CHECK(strategy.is(stateforward::sml::state<emel::io::read::state_ready>));
}
```

**Scope guardrail** (lines 363-386):
```cpp
CHECK(source.find("mmap") == std::string_view::npos);
CHECK(source.find("async") == std::string_view::npos);
CHECK(source.find("staged") == std::string_view::npos);
CHECK(source.find("chunked") == std::string_view::npos);
CHECK(source.find("::open(") == std::string_view::npos);
CHECK(source.find("::read(") == std::string_view::npos);
```

Batch read tests should remain source-span copy tests, not OS file tests.

### `tests/model/tensor/lifecycle_tests.cpp` (test, request-response)

**Rule-safe analog:** `tests/model/tensor/lifecycle_tests.cpp`

**Tensor-owned read/copy route** (lines 950-983):
```cpp
TEST_CASE("model_tensor_request_read_load_dispatches_through_io_read") {
  emel::io::read::sm io_read_actor{};
  emel::model::tensor::sm machine = make_tensor_sm_with_io_read(io_read_actor);
  ...
  CHECK(machine.process_event(request));
  CHECK(owner.buffer == target);
  CHECK(std::memcmp(target, "cdef", 4u) == 0);
  CHECK(state.lifecycle_state ==
        emel::model::tensor::event::lifecycle::resident);
}
```

Keep this as regression coverage that model-loader batch repair does not move tensor residency ownership out of `model/tensor`.

## Planning Artifact Patterns

### `.planning/ROADMAP.md` and `.planning/milestones/v1.25-ROADMAP.md` (config, batch)

**Rule-safe analog:** `.planning/ROADMAP.md`

**Active closeout path truth** (lines 63-71):
```markdown
Current closeout artifacts:
- `.planning/REQUIREMENTS.md`
- `.planning/v1.25-MILESTONE-AUDIT.md`

Prior archive snapshot:
- `.planning/milestones/v1.25-ROADMAP.md`
- `.planning/milestones/v1.25-REQUIREMENTS.md`
- `.planning/milestones/v1.25-MILESTONE-AUDIT.md`
- `.planning/milestones/v1.25-phases/`
```

**Phase 225 success criteria** (lines 407-419):
```markdown
1. `ctest --test-dir build/zig --output-on-failure -R emel_tests_model_and_batch`
   runs to completion or the dyld/libSystem launch blocker is eliminated with a
   source-backed maintained substitute explicitly recorded in validation.
2. Maintained read/copy `model/loader -> io/loader` orchestration no longer relies on an
   action loop calling `io_loader->process_event(...)`.
4. Closeout artifact paths in active and archived roadmap/requirements/audit docs point
   at files that exist after the v1.25 archive layout.
```

For the archived roadmap, replace stale root closeout paths in `.planning/milestones/v1.25-ROADMAP.md` lines 61-63 with archived paths under `.planning/milestones/`.

### `.planning/REQUIREMENTS.md` and `.planning/milestones/v1.25-REQUIREMENTS.md` (config, batch)

**Rule-safe analog:** `.planning/REQUIREMENTS.md`

**Pending refreshed gap pattern** (lines 53-65, 109-117):
```markdown
- [ ] **VAL-01**: Doctest coverage proves supported read behavior and representative failure
  handling through `process_event(...)` and SML state inspection.
- [ ] **VAL-03**: Public docs, generated architecture docs, planning artifacts, lint snapshots,
  benchmark snapshots, benchmark outputs, and model artifacts are updated from maintained
  commands when required and describe read-strategy support truthfully.

| TIO-03 | Phase 225 | Pending |
| VAL-04 | Phase 225 | Pending |
| VAL-01 | Phase 225 | Pending |
| VAL-03 | Phase 225 | Pending |
```

When implementation completes, update active requirements to validated only with source/test evidence. Archived requirements currently point readers to `.planning/REQUIREMENTS.md` at lines 1-7; if archived closeout path truth is part of the plan, update that pointer to distinguish current root requirements from archived v1.25 copies.

### `.planning/v1.25-MILESTONE-AUDIT.md` (config, batch)

**Rule-safe analog:** `.planning/v1.25-MILESTONE-AUDIT.md`

**Gap format to close** (frontmatter lines 13-38):
```yaml
gaps:
  requirements:
    - id: "VAL-01"
      status: "partial"
      verification_status: "passed in archived artifacts; partial in current rerun"
  integration:
    - from: "model/loader"
      to: "io/loader"
      issue: "Maintained read_copy path is wired, but src/emel/model/loader/actions.hpp dispatches per-tensor io_loader->process_event(...) inside an action loop."
  tech_debt:
    - phase: "v1.25 archive"
```

Update this audit only after validation. Do not mark requirements satisfied from roadmap/requirements artifacts alone.

### `.planning/phases/225-read-closeout-runtime-validation-and-sml-repair/225-VALIDATION.md` and `225-VERIFICATION.md` (config, batch)

**Rule-safe analogs:** `225-VALIDATION.md`, archived `223-VERIFICATION.md`

**Validation map pattern** from `225-VALIDATION.md` (lines 67-72):
```markdown
| 225-01-01 | 01 | 1 | TIO-03, VAL-04 | — | N/A | source/doctest | `ctest --test-dir build/zig --output-on-failure -R emel_tests_model_and_batch` | ✅ | ⬜ pending |
| 225-01-02 | 01 | 1 | VAL-01 | — | N/A | doctest | `ctest --test-dir build/zig --output-on-failure -R 'emel_tests_(model_and_batch|io)'` | ✅ | ⬜ pending |
| 225-01-03 | 01 | 1 | VAL-03 | — | N/A | docs/consistency | `node .codex/get-shit-done/bin/gsd-tools.cjs validate consistency` | ✅ | ⬜ pending |
```

**Verification evidence pattern** from archived `223-VERIFICATION.md` (lines 21-39):
```markdown
## Verification Commands

- `scripts/generate_docs.sh --check` passed.
- `ctest --test-dir build/zig --output-on-failure -R emel_tests_io` passed.
- `ctest --test-dir build/zig --output-on-failure -R emel_tests_model_and_batch`
  passed.
- `scripts/check_domain_boundaries.sh` passed.
- `node .codex/get-shit-done/bin/gsd-tools.cjs validate consistency` passed
  with the pre-existing Phase 211 warning.
- Changed-file scoped `scripts/quality_gates.sh` passed.
```

For Phase 225, include exact dyld output only if it recurs and clearly label source-backed substitute evidence.

## Shared Patterns

### Explicit SML Routing
**Source:** `src/emel/io/loader/sm.hpp` lines 28-62 and `src/emel/io/read/sm.hpp` lines 40-169  
**Apply to:** all `sm.hpp` files touched in this phase

Use destination-first rows, visual sections, guards for runtime selection, and completion transitions for small phase-level progress.

### Same-RTC Dispatch Carriers
**Source:** `src/emel/io/read/detail.hpp` lines 24-31 and `src/emel/io/read/sm.hpp` lines 324-337  
**Apply to:** model-loader batch result carriers and read batch result capture

Keep dispatch-local status on the wrapper stack or inside internal-only runtime events. Do not store it in persistent actor context or public mutable event payloads.

### Public Runtime Evidence
**Source:** `tests/model/loader/lifecycle_tests.cpp` lines 1020-1065 and 1416-1439  
**Apply to:** model-loader tests, maintained tool guardrails, verification docs

Assert `used_io_strategy == read_copy` only after public runtime execution succeeds. Error paths should keep `used_io_strategy == none`.

### Scope Guardrails
**Source:** `tests/io/read/lifecycle_tests.cpp` lines 363-386 and `tests/io/loader/lifecycle_tests.cpp` lines 305-318  
**Apply to:** I/O tests and model-loader guardrail tests

Continue rejecting mmap, async, staged/chunked, OS read calls, direct `io/read/detail.hpp`, and actor-detail reach-through in maintained lanes.

## Rejected Legacy Patterns

| File | Lines | Rejected Pattern | Reason |
|------|-------|------------------|--------|
| `src/emel/model/loader/actions.hpp` | 249-272 | Looping over planned effects and calling `io_loader->process_event(load)` inside an action | Violates Phase 225 gap closure; orchestration work is hidden inside an action loop. |
| `src/emel/model/loader/actions.hpp` | 281-287, 307-313 | Per-effect loops for result materialization | May remain only as bounded data preparation, but must not become route selection or child dispatch. |
| `.planning/milestones/v1.25-ROADMAP.md` | 61-63 | Archived roadmap closeout paths pointing at root `.planning/REQUIREMENTS.md` and `.planning/v1.25-MILESTONE-AUDIT.md` | Audit says archived copies exist under `.planning/milestones/`; path truth must be reconciled. |
| Any `detail.hpp`/`detail.cpp` | n/a | Moving behavior choice into helpers | Rule files explicitly reject helper relocation as a compliance fix. |

## No Analog Found

| File/Pattern | Role | Data Flow | Reason |
|--------------|------|-----------|--------|
| New `load_tensor_batch` / `read_tensor_batch` public event names | model | request-response/file-I/O | No existing batch read/copy public event exists; adapt single-tensor public events and same-RTC result carriers. |
| Model-loader single batch dispatch to `io/loader` | service | event-driven | Current closest analog is the rejected per-tensor action loop; use `io/loader` single dispatch and tensor same-RTC result patterns instead. |

## Metadata

**Analog search scope:** `src/emel/model/loader`, `src/emel/io/loader`, `src/emel/io/read`, `src/emel/model/tensor`, `src/emel/gbnf`, `tests/model`, `tests/io`, `.planning`
**Files scanned:** 90+ source/test/planning paths via `rg --files` and targeted `rg -n`
**Pattern extraction date:** 2026-05-06
**Project skills checked:** `.claude/skills/atmux-create-role`, `atmux-assign`, `atmux-capture`, `atmux-send`
