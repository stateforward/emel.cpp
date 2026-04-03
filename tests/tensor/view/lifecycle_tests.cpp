#include <cstdint>

#include <doctest/doctest.h>

#include "emel/graph/tensor/events.hpp"
#include "emel/graph/tensor/sm.hpp"
#include "emel/model/data.hpp"
#include "emel/model/tensor/events.hpp"
#include "emel/model/tensor/sm.hpp"
#include "emel/tensor/view/events.hpp"
#include "emel/tensor/view/sm.hpp"

namespace {

struct graph_tensor_view_policy {
  using tensor_machine_type = emel::graph::tensor::sm;
  using tensor_state_type = emel::graph::tensor::event::tensor_state;
  static constexpr int32_t max_tensors = emel::graph::tensor::detail::max_tensors;
  static constexpr int32_t none_error_code =
      static_cast<int32_t>(emel::error::cast(emel::graph::tensor::error::none));
  static constexpr int32_t invalid_request_error_code =
      static_cast<int32_t>(emel::error::cast(emel::graph::tensor::error::invalid_request));
  static constexpr int32_t internal_error_code =
      static_cast<int32_t>(emel::error::cast(emel::graph::tensor::error::internal_error));

  static bool capture_tensor_state(tensor_machine_type & machine,
                                   const int32_t tensor_id,
                                   tensor_state_type & state_out,
                                   int32_t & err_out) noexcept {
    return machine.process_event(emel::graph::tensor::event::capture_tensor_state{
      .tensor_id = tensor_id,
      .state_out = &state_out,
      .error_out = &err_out,
    });
  }
};

struct model_tensor_view_policy {
  using tensor_machine_type = emel::model::tensor::sm;
  using tensor_state_type = emel::model::tensor::event::tensor_state;
  static constexpr int32_t max_tensors = emel::model::tensor::detail::max_tensors;
  static constexpr int32_t none_error_code =
      static_cast<int32_t>(emel::error::cast(emel::model::tensor::error::none));
  static constexpr int32_t invalid_request_error_code =
      static_cast<int32_t>(emel::error::cast(emel::model::tensor::error::invalid_request));
  static constexpr int32_t internal_error_code =
      static_cast<int32_t>(emel::error::cast(emel::model::tensor::error::internal_error));

  static bool capture_tensor_state(tensor_machine_type & machine,
                                   const int32_t tensor_id,
                                   tensor_state_type & state_out,
                                   int32_t & err_out) noexcept {
    return machine.process_event(emel::model::tensor::event::capture_tensor_state{
      .tensor_id = tensor_id,
      .state_out = &state_out,
      .error_out = &err_out,
    });
  }
};

struct failing_with_error_policy {
  using tensor_machine_type = emel::graph::tensor::sm;
  using tensor_state_type = emel::graph::tensor::event::tensor_state;
  static constexpr int32_t max_tensors = emel::graph::tensor::detail::max_tensors;
  static constexpr int32_t none_error_code =
      static_cast<int32_t>(emel::error::cast(emel::graph::tensor::error::none));
  static constexpr int32_t invalid_request_error_code =
      static_cast<int32_t>(emel::error::cast(emel::graph::tensor::error::invalid_request));
  static constexpr int32_t internal_error_code =
      static_cast<int32_t>(emel::error::cast(emel::graph::tensor::error::internal_error));

  static bool capture_tensor_state(tensor_machine_type &,
                                   const int32_t,
                                   tensor_state_type &,
                                   int32_t & err_out) noexcept {
    err_out = internal_error_code;
    return false;
  }
};

struct failing_without_error_policy {
  using tensor_machine_type = emel::graph::tensor::sm;
  using tensor_state_type = emel::graph::tensor::event::tensor_state;
  static constexpr int32_t max_tensors = emel::graph::tensor::detail::max_tensors;
  static constexpr int32_t none_error_code =
      static_cast<int32_t>(emel::error::cast(emel::graph::tensor::error::none));
  static constexpr int32_t invalid_request_error_code =
      static_cast<int32_t>(emel::error::cast(emel::graph::tensor::error::invalid_request));
  static constexpr int32_t internal_error_code =
      static_cast<int32_t>(emel::error::cast(emel::graph::tensor::error::internal_error));

  static bool capture_tensor_state(tensor_machine_type &,
                                   const int32_t,
                                   tensor_state_type &,
                                   int32_t & err_out) noexcept {
    err_out = none_error_code;
    return false;
  }
};

using graph_tensor_sm = emel::graph::tensor::sm;
using graph_tensor_view_sm = emel::tensor::view::sm<graph_tensor_view_policy>;
using model_tensor_sm = emel::model::tensor::sm;
using model_tensor_view_sm = emel::tensor::view::sm<model_tensor_view_policy>;
using failing_with_error_view_sm = emel::tensor::view::sm<failing_with_error_policy>;
using failing_without_error_view_sm = emel::tensor::view::sm<failing_without_error_policy>;

void * fake_buffer(const uintptr_t value) {
  return reinterpret_cast<void *>(value);
}

emel::model::data::tensor_record make_tensor_record() {
  emel::model::data::tensor_record tensor{};
  tensor.type = 7;
  tensor.file_offset = 2048u;
  tensor.data_size = 128u;
  tensor.file_index = 1u;
  return tensor;
}

}  // namespace

TEST_CASE("tensor_view_capture_tensor_view_reads_tensor_state") {
  graph_tensor_sm tensors{};
  graph_tensor_view_sm view{};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::tensor::view::error::none));

  REQUIRE(tensors.process_event(emel::graph::tensor::event::reserve_tensor{
    .tensor_id = 21,
    .buffer = fake_buffer(0x7000u),
    .buffer_bytes = 96u,
    .consumer_refs = 2,
    .is_leaf = false,
    .error_out = &err,
  }));
  REQUIRE(err == static_cast<int32_t>(emel::error::cast(emel::tensor::view::error::none)));

  emel::graph::tensor::event::tensor_state state{};
  REQUIRE(view.process_event(emel::tensor::view::event::capture_tensor_view<graph_tensor_view_policy>{
    .tensor_machine = &tensors,
    .tensor_id = 21,
    .state_out = &state,
    .error_out = &err,
  }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::tensor::view::error::none)));
  CHECK(state.lifecycle_state == emel::graph::tensor::event::lifecycle::empty);
  CHECK(state.seed_refs == 2u);
  CHECK(state.live_refs == 2u);
}

TEST_CASE("tensor_view_capture_tensor_view_rejects_invalid_request") {
  graph_tensor_sm tensors{};
  graph_tensor_view_sm view{};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::tensor::view::error::none));
  emel::graph::tensor::event::tensor_state state{};

  CHECK_FALSE(view.process_event(
      emel::tensor::view::event::capture_tensor_view<graph_tensor_view_policy>{
    .tensor_machine = nullptr,
    .tensor_id = 0,
    .state_out = &state,
    .error_out = &err,
  }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::tensor::view::error::invalid_request)));

  CHECK_FALSE(view.process_event(
      emel::tensor::view::event::capture_tensor_view<graph_tensor_view_policy>{
    .tensor_machine = &tensors,
    .tensor_id = -1,
    .state_out = &state,
    .error_out = &err,
  }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::tensor::view::error::invalid_request)));

  CHECK_FALSE(view.process_event(
      emel::tensor::view::event::capture_tensor_view<graph_tensor_view_policy>{
    .tensor_machine = &tensors,
    .tensor_id = 0,
    .state_out = nullptr,
    .error_out = &err,
  }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::tensor::view::error::invalid_request)));
}

TEST_CASE("tensor_view_capture_tensor_view_propagates_tensor_error") {
  graph_tensor_sm tensors{};
  graph_tensor_view_sm view{};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::tensor::view::error::none));

  REQUIRE(tensors.process_event(emel::graph::tensor::event::reserve_tensor{
    .tensor_id = 31,
    .buffer = fake_buffer(0x7100u),
    .buffer_bytes = 128u,
    .consumer_refs = 1,
    .is_leaf = true,
    .error_out = &err,
  }));
  CHECK_FALSE(tensors.process_event(emel::graph::tensor::event::publish_filled_tensor{
    .tensor_id = 31,
    .error_out = &err,
  }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::graph::tensor::error::internal_error)));

  emel::graph::tensor::event::tensor_state state{};
  REQUIRE(view.process_event(emel::tensor::view::event::capture_tensor_view<graph_tensor_view_policy>{
    .tensor_machine = &tensors,
    .tensor_id = 31,
    .state_out = &state,
    .error_out = &err,
  }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::tensor::view::error::none)));
  CHECK(state.lifecycle_state == emel::graph::tensor::event::lifecycle::internal_error);
}

TEST_CASE("tensor_view_capture_tensor_view_observes_publish_then_release") {
  graph_tensor_sm tensors{};
  graph_tensor_view_sm view{};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::tensor::view::error::none));

  REQUIRE(tensors.process_event(emel::graph::tensor::event::reserve_tensor{
    .tensor_id = 44,
    .buffer = fake_buffer(0x7200u),
    .buffer_bytes = 128u,
    .consumer_refs = 1,
    .is_leaf = false,
    .error_out = &err,
  }));
  REQUIRE(tensors.process_event(emel::graph::tensor::event::publish_filled_tensor{
    .tensor_id = 44,
    .error_out = &err,
  }));

  emel::graph::tensor::event::tensor_state state{};
  REQUIRE(view.process_event(emel::tensor::view::event::capture_tensor_view<graph_tensor_view_policy>{
    .tensor_machine = &tensors,
    .tensor_id = 44,
    .state_out = &state,
    .error_out = &err,
  }));
  CHECK(state.lifecycle_state == emel::graph::tensor::event::lifecycle::filled);

  REQUIRE(tensors.process_event(emel::graph::tensor::event::release_tensor_ref{
    .tensor_id = 44,
    .error_out = &err,
  }));
  REQUIRE(view.process_event(emel::tensor::view::event::capture_tensor_view<graph_tensor_view_policy>{
    .tensor_machine = &tensors,
    .tensor_id = 44,
    .state_out = &state,
    .error_out = &err,
  }));
  CHECK(state.lifecycle_state == emel::graph::tensor::event::lifecycle::empty);
}

TEST_CASE("tensor_view_unexpected_event_keeps_machine_dispatchable") {
  graph_tensor_sm tensors{};
  graph_tensor_view_sm view{};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::tensor::view::error::none));
  emel::graph::tensor::event::tensor_state state{};

  CHECK(view.process_event(
      emel::tensor::view::events::capture_tensor_view_done<graph_tensor_view_policy>{}));

  CHECK(view.process_event(emel::tensor::view::event::capture_tensor_view<graph_tensor_view_policy>{
    .tensor_machine = &tensors,
    .tensor_id = 0,
    .state_out = &state,
    .error_out = &err,
  }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::tensor::view::error::none)));
}

TEST_CASE("tensor_view_capture_tensor_view_reads_model_tensor_state") {
  model_tensor_sm tensors{};
  model_tensor_view_sm view{};
  const auto tensor_record = make_tensor_record();
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::tensor::view::error::none));

  auto bind =
      emel::model::tensor::event::bind_tensor{4, tensor_record, fake_buffer(0x7300u), 128u};
  bind.error_out = &err;
  REQUIRE(tensors.process_event(bind));

  emel::model::tensor::event::tensor_state state{};
  REQUIRE(view.process_event(emel::tensor::view::event::capture_tensor_view<model_tensor_view_policy>{
    .tensor_machine = &tensors,
    .tensor_id = 4,
    .state_out = &state,
    .error_out = &err,
  }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::tensor::view::error::none)));
  CHECK(state.lifecycle_state == emel::model::tensor::event::lifecycle::resident);
  CHECK(state.buffer == fake_buffer(0x7300u));
  CHECK(state.data_size == 128u);
  CHECK(state.file_offset == 2048u);
}

TEST_CASE("tensor_view_capture_tensor_view_maps_tensor_internal_error") {
  graph_tensor_sm tensors{};
  failing_with_error_view_sm view{};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::tensor::view::error::none));
  emel::graph::tensor::event::tensor_state state{};

  CHECK_FALSE(
      view.process_event(emel::tensor::view::event::capture_tensor_view<failing_with_error_policy>{
        .tensor_machine = &tensors,
        .tensor_id = 0,
        .state_out = &state,
        .error_out = &err,
      }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::tensor::view::error::internal_error)));
}

TEST_CASE("tensor_view_capture_tensor_view_maps_failed_without_error_to_internal_error") {
  graph_tensor_sm tensors{};
  failing_without_error_view_sm view{};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::tensor::view::error::none));
  emel::graph::tensor::event::tensor_state state{};

  CHECK_FALSE(
      view.process_event(emel::tensor::view::event::capture_tensor_view<failing_without_error_policy>{
        .tensor_machine = &tensors,
        .tensor_id = 0,
        .state_out = &state,
        .error_out = &err,
      }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::tensor::view::error::internal_error)));
}

TEST_CASE("tensor_view_unexpected_runtime_event_marks_internal_error") {
  graph_tensor_view_sm view{};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::tensor::view::error::none));
  emel::tensor::view::detail::runtime_status ctx{};

  struct unexpected_runtime_event {
    emel::tensor::view::detail::runtime_status & ctx;
    int32_t & error_code_out;
  };

  CHECK(view.process_event(unexpected_runtime_event{ctx, err}));
  CHECK(ctx.err == emel::error::cast(emel::tensor::view::error::internal_error));
  CHECK_FALSE(ctx.ok);
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::tensor::view::error::internal_error)));
}
