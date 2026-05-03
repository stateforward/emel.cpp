#include <cstdint>

#include <array>
#include <span>

#include <doctest/doctest.h>

#include "emel/model/data.hpp"
#include "emel/model/tensor/actions.hpp"
#include "emel/model/tensor/events.hpp"
#include "emel/model/tensor/guards.hpp"
#include "emel/model/tensor/sm.hpp"

namespace {

emel::model::data::tensor_record make_tensor_record() {
  emel::model::data::tensor_record tensor{};
  tensor.type = 9;
  tensor.file_offset = 4096u;
  tensor.data_size = 512u;
  tensor.file_index = 3u;
  return tensor;
}

const void *fake_buffer(const uintptr_t value) {
  return reinterpret_cast<const void *>(value);
}

struct owner_state {
  bool bind_done = false;
  bool bind_error = false;
  bool plan_done = false;
  bool plan_error = false;
  bool apply_done = false;
  bool apply_error = false;
  emel::error::type err = emel::error::cast(emel::model::tensor::error::none);
  uint32_t effect_count = 0u;
};

void on_bind_storage_done(
    void *object,
    const emel::model::tensor::events::bind_storage_done &) noexcept {
  auto *owner = static_cast<owner_state *>(object);
  owner->bind_done = true;
  owner->bind_error = false;
}

void on_bind_storage_error(
    void *object,
    const emel::model::tensor::events::bind_storage_error &ev) noexcept {
  auto *owner = static_cast<owner_state *>(object);
  owner->bind_done = false;
  owner->bind_error = true;
  owner->err = ev.err;
}

void on_plan_load_done(
    void *object,
    const emel::model::tensor::events::plan_load_done &ev) noexcept {
  auto *owner = static_cast<owner_state *>(object);
  owner->plan_done = true;
  owner->plan_error = false;
  owner->effect_count = ev.effect_count;
}

void on_plan_load_error(
    void *object,
    const emel::model::tensor::events::plan_load_error &ev) noexcept {
  auto *owner = static_cast<owner_state *>(object);
  owner->plan_done = false;
  owner->plan_error = true;
  owner->err = ev.err;
}

void on_apply_effect_results_done(
    void *object,
    const emel::model::tensor::events::apply_effect_results_done &) noexcept {
  auto *owner = static_cast<owner_state *>(object);
  owner->apply_done = true;
  owner->apply_error = false;
}

void on_apply_effect_results_error(
    void *object, const emel::model::tensor::events::apply_effect_results_error
                      &ev) noexcept {
  auto *owner = static_cast<owner_state *>(object);
  owner->apply_done = false;
  owner->apply_error = true;
  owner->err = ev.err;
}

} // namespace

TEST_CASE("model_tensor_bind_capture_evict_cycle") {
  emel::model::tensor::sm machine{};
  auto tensor_record = make_tensor_record();
  int32_t err =
      static_cast<int32_t>(emel::error::cast(emel::model::tensor::error::none));

  auto bind = emel::model::tensor::event::bind_tensor{
      12, tensor_record, fake_buffer(0x9000u), 512u};
  bind.error_out = &err;
  REQUIRE(machine.process_event(bind));
  CHECK(err == static_cast<int32_t>(
                   emel::error::cast(emel::model::tensor::error::none)));

  emel::model::tensor::event::tensor_state state{};
  REQUIRE(
      machine.process_event(emel::model::tensor::event::capture_tensor_state{
          .tensor_id = 12,
          .state_out = &state,
          .error_out = &err,
      }));
  CHECK(state.lifecycle_state ==
        emel::model::tensor::event::lifecycle::resident);
  CHECK(state.buffer == fake_buffer(0x9000u));
  CHECK(state.buffer_bytes == 512u);
  CHECK(state.file_offset == 4096u);
  CHECK(state.data_size == 512u);
  CHECK(state.file_index == 3u);
  CHECK(state.tensor_type == 9);

  REQUIRE(machine.process_event(emel::model::tensor::event::evict_tensor{
      .tensor_id = 12,
      .error_out = &err,
  }));
  REQUIRE(
      machine.process_event(emel::model::tensor::event::capture_tensor_state{
          .tensor_id = 12,
          .state_out = &state,
          .error_out = &err,
      }));
  CHECK(state.lifecycle_state ==
        emel::model::tensor::event::lifecycle::evicted);
  CHECK(state.buffer == nullptr);
  CHECK(state.buffer_bytes == 0u);
  CHECK(state.file_offset == 4096u);
}

TEST_CASE("model_tensor_rejects_invalid_requests") {
  emel::model::tensor::sm machine{};
  auto tensor_record = make_tensor_record();
  int32_t err =
      static_cast<int32_t>(emel::error::cast(emel::model::tensor::error::none));
  emel::model::tensor::event::tensor_state state{};

  auto invalid_bind =
      emel::model::tensor::event::bind_tensor{0, tensor_record, nullptr, 512u};
  invalid_bind.error_out = &err;
  CHECK_FALSE(machine.process_event(invalid_bind));
  CHECK_FALSE(
      machine.process_event(emel::model::tensor::event::capture_tensor_state{
          .tensor_id = 0,
          .state_out = nullptr,
          .error_out = &err,
      }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(
                   emel::model::tensor::error::invalid_request)));

  CHECK_FALSE(machine.process_event(emel::model::tensor::event::evict_tensor{
      .tensor_id = 5,
      .error_out = &err,
  }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(
                   emel::model::tensor::error::invalid_request)));

  auto resident_bind = emel::model::tensor::event::bind_tensor{
      5, tensor_record, fake_buffer(0x9100u), 512u};
  resident_bind.error_out = &err;
  REQUIRE(machine.process_event(resident_bind));
  REQUIRE(
      machine.process_event(emel::model::tensor::event::capture_tensor_state{
          .tensor_id = 5,
          .state_out = &state,
          .error_out = &err,
      }));
  CHECK(state.lifecycle_state ==
        emel::model::tensor::event::lifecycle::resident);
}

TEST_CASE("model_tensor_unexpected_event_keeps_machine_dispatchable") {
  emel::model::tensor::sm machine{};
  auto tensor_record = make_tensor_record();
  int32_t err =
      static_cast<int32_t>(emel::error::cast(emel::model::tensor::error::none));

  CHECK(machine.process_event(emel::model::tensor::events::bind_tensor_done{}));
  auto bind = emel::model::tensor::event::bind_tensor{
      2, tensor_record, fake_buffer(0x9200u), 256u};
  bind.error_out = &err;
  CHECK(machine.process_event(bind));
  CHECK(err == static_cast<int32_t>(
                   emel::error::cast(emel::model::tensor::error::none)));
}

TEST_CASE("model_tensor_bind_plan_apply_storage_lifecycle") {
  emel::model::tensor::sm machine{};
  owner_state owner{};
  std::array<emel::model::data::tensor_record, 2> tensors{};
  tensors[0].file_offset = 4096u;
  tensors[0].data_size = 32u;
  tensors[0].file_index = 1u;
  tensors[0].type = 7;
  tensors[1].file_offset = 8192u;
  tensors[1].data_size = 64u;
  tensors[1].file_index = 2u;
  tensors[1].type = 8;

  emel::model::tensor::event::bind_storage bind{std::span{tensors}};
  bind.on_done = {&owner, on_bind_storage_done};
  bind.on_error = {&owner, on_bind_storage_error};
  CHECK(machine.process_event(bind));
  CHECK(owner.bind_done);
  CHECK_FALSE(owner.bind_error);

  std::array<emel::model::tensor::effect_request, 2> effects{};
  emel::model::tensor::event::plan_load plan{std::span{effects}};
  plan.on_done = {&owner, on_plan_load_done};
  plan.on_error = {&owner, on_plan_load_error};
  CHECK(machine.process_event(plan));
  CHECK(owner.plan_done);
  CHECK_FALSE(owner.plan_error);
  CHECK(owner.effect_count == 2u);
  CHECK(effects[0].offset == 4096u);
  CHECK(effects[0].size == 32u);
  CHECK(effects[1].offset == 8192u);
  CHECK(effects[1].size == 64u);

  std::array<emel::model::tensor::effect_result, 2> results{
      emel::model::tensor::effect_result{
          .kind = emel::model::tensor::effect_kind::k_none,
          .handle = const_cast<void *>(fake_buffer(0xA000u)),
          .err = emel::error::cast(emel::model::tensor::error::none),
      },
      emel::model::tensor::effect_result{
          .kind = emel::model::tensor::effect_kind::k_none,
          .handle = const_cast<void *>(fake_buffer(0xB000u)),
          .err = emel::error::cast(emel::model::tensor::error::none),
      },
  };
  emel::model::tensor::event::apply_effect_results apply{
      std::span<const emel::model::tensor::effect_result>{results}};
  apply.on_done = {&owner, on_apply_effect_results_done};
  apply.on_error = {&owner, on_apply_effect_results_error};
  CHECK(machine.process_event(apply));
  CHECK(owner.apply_done);
  CHECK_FALSE(owner.apply_error);
  CHECK(tensors[0].data == fake_buffer(0xA000u));
  CHECK(tensors[1].data == fake_buffer(0xB000u));

  emel::model::tensor::event::tensor_state state{};
  CHECK(machine.process_event(emel::model::tensor::event::capture_tensor_state{
      .tensor_id = 1,
      .state_out = &state,
  }));
  CHECK(state.lifecycle_state ==
        emel::model::tensor::event::lifecycle::resident);
  CHECK(state.buffer == fake_buffer(0xB000u));
  CHECK(state.buffer_bytes == 64u);
  CHECK(state.file_offset == 8192u);
  CHECK(state.file_index == 2u);
  CHECK(state.tensor_type == 8);
}

TEST_CASE("model_tensor_storage_load_rejects_invalid_inputs") {
  emel::model::tensor::sm machine{};
  owner_state owner{};
  std::array<emel::model::data::tensor_record, 2> tensors{};

  emel::model::tensor::event::bind_storage empty_bind{
      std::span{tensors}.subspan(0u, 0u)};
  empty_bind.on_done = {&owner, on_bind_storage_done};
  empty_bind.on_error = {&owner, on_bind_storage_error};
  CHECK_FALSE(machine.process_event(empty_bind));
  CHECK_FALSE(owner.bind_done);
  CHECK(owner.bind_error);
  CHECK(owner.err ==
        emel::error::cast(emel::model::tensor::error::invalid_request));

  emel::model::tensor::event::bind_storage bind{std::span{tensors}};
  bind.on_done = {&owner, on_bind_storage_done};
  bind.on_error = {&owner, on_bind_storage_error};
  CHECK(machine.process_event(bind));

  std::array<emel::model::tensor::effect_request, 1> effects{};
  emel::model::tensor::event::plan_load plan{std::span{effects}};
  plan.on_done = {&owner, on_plan_load_done};
  plan.on_error = {&owner, on_plan_load_error};
  CHECK_FALSE(machine.process_event(plan));
  CHECK_FALSE(owner.plan_done);
  CHECK(owner.plan_error);
  CHECK(owner.err == emel::error::cast(emel::model::tensor::error::capacity));
}

TEST_CASE("model_tensor_apply_results_rejects_count_mismatch") {
  emel::model::tensor::sm machine{};
  owner_state owner{};
  std::array<emel::model::data::tensor_record, 1> tensors{};

  emel::model::tensor::event::bind_storage bind{std::span{tensors}};
  bind.on_done = {&owner, on_bind_storage_done};
  bind.on_error = {&owner, on_bind_storage_error};
  CHECK(machine.process_event(bind));

  std::array<emel::model::tensor::effect_request, 1> effects{};
  emel::model::tensor::event::plan_load plan{std::span{effects}};
  plan.on_done = {&owner, on_plan_load_done};
  plan.on_error = {&owner, on_plan_load_error};
  CHECK(machine.process_event(plan));

  std::array<emel::model::tensor::effect_result, 0> results{};
  emel::model::tensor::event::apply_effect_results apply{
      std::span<const emel::model::tensor::effect_result>{results}};
  apply.on_done = {&owner, on_apply_effect_results_done};
  apply.on_error = {&owner, on_apply_effect_results_error};
  CHECK_FALSE(machine.process_event(apply));
  CHECK_FALSE(owner.apply_done);
  CHECK(owner.apply_error);
  CHECK(owner.err ==
        emel::error::cast(emel::model::tensor::error::invalid_request));
}

TEST_CASE("model_tensor_apply_results_maps_effect_errors_to_backend_error") {
  emel::model::tensor::sm machine{};
  owner_state owner{};
  std::array<emel::model::data::tensor_record, 1> tensors{};

  emel::model::tensor::event::bind_storage bind{std::span{tensors}};
  bind.on_done = {&owner, on_bind_storage_done};
  bind.on_error = {&owner, on_bind_storage_error};
  CHECK(machine.process_event(bind));

  std::array<emel::model::tensor::effect_request, 1> effects{};
  emel::model::tensor::event::plan_load plan{std::span{effects}};
  plan.on_done = {&owner, on_plan_load_done};
  plan.on_error = {&owner, on_plan_load_error};
  CHECK(machine.process_event(plan));

  std::array<emel::model::tensor::effect_result, 1> results{
      emel::model::tensor::effect_result{
          .kind = emel::model::tensor::effect_kind::k_none,
          .handle = nullptr,
          .err = emel::error::cast(emel::model::tensor::error::out_of_memory),
      },
  };
  emel::model::tensor::event::apply_effect_results apply{
      std::span<const emel::model::tensor::effect_result>{results}};
  apply.on_done = {&owner, on_apply_effect_results_done};
  apply.on_error = {&owner, on_apply_effect_results_error};
  CHECK_FALSE(machine.process_event(apply));
  CHECK_FALSE(owner.apply_done);
  CHECK(owner.apply_error);
  CHECK(owner.err ==
        emel::error::cast(emel::model::tensor::error::backend_error));
}

TEST_CASE("model_tensor_bulk_guard_and_unexpected_action_predicates") {
  emel::model::tensor::action::context action_ctx{};
  owner_state owner{};
  std::array<emel::model::data::tensor_record, 1> tensors{};
  emel::model::tensor::event::bind_storage bind{std::span{tensors}};

  CHECK(emel::model::tensor::guard::bind_storage_done_callback_absent{}(
      bind, action_ctx));
  CHECK(emel::model::tensor::guard::bind_storage_error_callback_absent{}(
      bind, action_ctx));
  bind.on_done = {&owner, on_bind_storage_done};
  bind.on_error = {&owner, on_bind_storage_error};
  CHECK(emel::model::tensor::guard::bind_storage_done_callback_present{}(
      bind, action_ctx));
  CHECK(emel::model::tensor::guard::bind_storage_error_callback_present{}(
      bind, action_ctx));
  CHECK(emel::model::tensor::guard::storage_bind_valid{}(bind));

  emel::model::tensor::event::bind_storage null_bind{
      std::span<emel::model::data::tensor_record>{}};
  CHECK_FALSE(emel::model::tensor::guard::storage_bind_valid{}(null_bind));
  CHECK(emel::model::tensor::guard::storage_bind_invalid{}(null_bind));

  emel::model::tensor::event::bind_storage empty_bind{
      std::span{tensors}.subspan(0u, 0u)};
  CHECK_FALSE(emel::model::tensor::guard::storage_bind_valid{}(empty_bind));

  emel::model::tensor::event::bind_storage oversized_bind{
      std::span<emel::model::data::tensor_record>{
          tensors.data(),
          static_cast<size_t>(emel::model::tensor::detail::max_tensors) + 1u}};
  CHECK_FALSE(
      emel::model::tensor::guard::storage_bind_valid{}(oversized_bind));

  int32_t error_code =
      static_cast<int32_t>(emel::error::cast(emel::model::tensor::error::none));
  auto error_tensor_record = make_tensor_record();
  auto error_bind_tensor = emel::model::tensor::event::bind_tensor{
      3, error_tensor_record, fake_buffer(0xC000u), 64u};
  emel::model::tensor::detail::runtime_status error_status{};
  emel::model::tensor::detail::bind_tensor_runtime error_runtime{
      error_bind_tensor, error_status, &error_code};

  error_status.err =
      emel::error::cast(emel::model::tensor::error::model_invalid);
  CHECK(emel::model::tensor::guard::error_model_invalid{}(error_runtime,
                                                          action_ctx));
  error_status.err =
      emel::error::cast(emel::model::tensor::error::out_of_memory);
  CHECK(emel::model::tensor::guard::error_out_of_memory{}(error_runtime,
                                                          action_ctx));
  error_status.err =
      emel::error::cast(emel::model::tensor::error::internal_error);
  CHECK(emel::model::tensor::guard::error_internal_error{}(error_runtime,
                                                           action_ctx));
  error_status.err = emel::error::cast(emel::model::tensor::error::untracked);
  CHECK(
      emel::model::tensor::guard::error_untracked{}(error_runtime, action_ctx));
  error_status.err = static_cast<emel::error::type>(0x4000u);
  CHECK(emel::model::tensor::guard::error_unknown{}(error_runtime,
                                                    action_ctx));
  const std::array<emel::error::type, 8> known_errors{
      emel::error::cast(emel::model::tensor::error::none),
      emel::error::cast(emel::model::tensor::error::invalid_request),
      emel::error::cast(emel::model::tensor::error::capacity),
      emel::error::cast(emel::model::tensor::error::backend_error),
      emel::error::cast(emel::model::tensor::error::model_invalid),
      emel::error::cast(emel::model::tensor::error::out_of_memory),
      emel::error::cast(emel::model::tensor::error::internal_error),
      emel::error::cast(emel::model::tensor::error::untracked),
  };
  for (const auto err_value : known_errors) {
    error_status.err = err_value;
    CHECK_FALSE(
        emel::model::tensor::guard::error_unknown{}(error_runtime, action_ctx));
  }

  std::array<emel::model::tensor::effect_request, 1> effects{};
  emel::model::tensor::event::plan_load plan{std::span{effects}};

  CHECK(emel::model::tensor::guard::plan_load_done_callback_absent{}(
      plan, action_ctx));
  CHECK(emel::model::tensor::guard::plan_load_error_callback_absent{}(
      plan, action_ctx));
  plan.on_done = {&owner, on_plan_load_done};
  plan.on_error = {&owner, on_plan_load_error};
  CHECK(emel::model::tensor::guard::plan_load_done_callback_present{}(
      plan, action_ctx));
  CHECK(emel::model::tensor::guard::plan_load_error_callback_present{}(
      plan, action_ctx));
  CHECK_FALSE(emel::model::tensor::guard::storage_bound{}(action_ctx));
  CHECK(
      emel::model::tensor::guard::plan_load_invalid_request{}(plan, action_ctx));
  action_ctx.bound_records = std::span{tensors};
  CHECK(emel::model::tensor::guard::storage_bound{}(action_ctx));
  CHECK(
      emel::model::tensor::guard::plan_load_valid{}(plan, action_ctx));
  CHECK_FALSE(emel::model::tensor::guard::plan_load_invalid_capacity{}(
      plan, action_ctx));
  emel::model::tensor::event::plan_load no_capacity_plan{
      std::span{effects}.subspan(0u, 0u)};
  CHECK_FALSE(emel::model::tensor::guard::plan_load_valid{}(
      no_capacity_plan, action_ctx));
  CHECK(emel::model::tensor::guard::plan_load_invalid_capacity{}(
      no_capacity_plan, action_ctx));

  std::array<emel::model::tensor::effect_result, 1> results{};
  emel::model::tensor::event::apply_effect_results apply{
      std::span<const emel::model::tensor::effect_result>{results}};

  CHECK(emel::model::tensor::guard::apply_effect_results_done_callback_absent{}(
      apply, action_ctx));
  CHECK(
      emel::model::tensor::guard::apply_effect_results_error_callback_absent{}(
          apply, action_ctx));
  apply.on_done = {&owner, on_apply_effect_results_done};
  apply.on_error = {&owner, on_apply_effect_results_error};
  CHECK(
      emel::model::tensor::guard::apply_effect_results_done_callback_present{}(
          apply, action_ctx));
  CHECK(
      emel::model::tensor::guard::apply_effect_results_error_callback_present{}(
          apply, action_ctx));
  CHECK(emel::model::tensor::guard::apply_results_valid{}(apply, action_ctx));
  CHECK_FALSE(
      emel::model::tensor::guard::apply_results_invalid{}(apply, action_ctx));
  CHECK(
      emel::model::tensor::guard::apply_effect_errors_absent{}(apply,
                                                               action_ctx));
  results[0].err =
      emel::error::cast(emel::model::tensor::error::out_of_memory);
  CHECK(
      emel::model::tensor::guard::apply_effect_errors_present{}(apply,
                                                                action_ctx));
  std::array<emel::model::tensor::effect_result, 0> no_results{};
  emel::model::tensor::event::apply_effect_results no_results_apply{
      std::span<const emel::model::tensor::effect_result>{no_results}};
  CHECK_FALSE(emel::model::tensor::guard::apply_results_valid{}(
      no_results_apply, action_ctx));
  CHECK(emel::model::tensor::guard::apply_results_invalid{}(
      no_results_apply, action_ctx));

  int32_t err =
      static_cast<int32_t>(emel::error::cast(emel::model::tensor::error::none));
  auto tensor_record = make_tensor_record();
  auto bind_tensor = emel::model::tensor::event::bind_tensor{
      3, tensor_record, fake_buffer(0xC000u), 64u};
  emel::model::tensor::detail::runtime_status status{};
  emel::model::tensor::detail::bind_tensor_runtime single_runtime{bind_tensor,
                                                                  status, &err};

  CHECK(emel::model::tensor::guard::bind_tensor_request_valid{}(single_runtime,
                                                                action_ctx));
  CHECK(emel::model::tensor::guard::operation_not_dispatched{}(single_runtime));
  CHECK_FALSE(
      emel::model::tensor::guard::operation_succeeded{}(single_runtime));
  status.accepted = true;
  status.err = emel::error::cast(emel::model::tensor::error::none);
  CHECK(emel::model::tensor::guard::operation_succeeded{}(single_runtime));
  CHECK_FALSE(
      emel::model::tensor::guard::operation_not_dispatched{}(single_runtime));
  status.err = emel::error::cast(emel::model::tensor::error::invalid_request);
  CHECK_FALSE(
      emel::model::tensor::guard::operation_succeeded{}(single_runtime));

  auto invalid_id_bind = emel::model::tensor::event::bind_tensor{
      -1, tensor_record, fake_buffer(0xC000u), 64u};
  emel::model::tensor::detail::bind_tensor_runtime invalid_id_bind_runtime{
      invalid_id_bind, status, &err};
  CHECK_FALSE(emel::model::tensor::guard::bind_tensor_request_valid{}(
      invalid_id_bind_runtime, action_ctx));

  auto null_buffer_bind =
      emel::model::tensor::event::bind_tensor{3, tensor_record, nullptr, 64u};
  emel::model::tensor::detail::bind_tensor_runtime null_buffer_bind_runtime{
      null_buffer_bind, status, &err};
  CHECK_FALSE(emel::model::tensor::guard::bind_tensor_request_valid{}(
      null_buffer_bind_runtime, action_ctx));

  auto zero_bytes_bind = emel::model::tensor::event::bind_tensor{
      3, tensor_record, fake_buffer(0xC000u), 0u};
  emel::model::tensor::detail::bind_tensor_runtime zero_bytes_bind_runtime{
      zero_bytes_bind, status, &err};
  CHECK_FALSE(emel::model::tensor::guard::bind_tensor_request_valid{}(
      zero_bytes_bind_runtime, action_ctx));

  auto zero_size_record = make_tensor_record();
  zero_size_record.data_size = 0u;
  auto zero_size_bind = emel::model::tensor::event::bind_tensor{
      3, zero_size_record, fake_buffer(0xC000u), 64u};
  emel::model::tensor::detail::bind_tensor_runtime zero_size_bind_runtime{
      zero_size_bind, status, &err};
  CHECK_FALSE(emel::model::tensor::guard::bind_tensor_request_valid{}(
      zero_size_bind_runtime, action_ctx));

  emel::model::tensor::event::evict_tensor evict{.tensor_id = 3};
  emel::model::tensor::detail::evict_tensor_runtime evict_runtime{evict, status,
                                                                  &err};
  action_ctx.tensors.lifecycle[3] =
      emel::model::tensor::event::lifecycle::resident;
  CHECK(emel::model::tensor::guard::evict_tensor_request_valid{}(evict_runtime,
                                                                 action_ctx));
  action_ctx.tensors.lifecycle[3] =
      emel::model::tensor::event::lifecycle::unbound;
  CHECK_FALSE(emel::model::tensor::guard::evict_tensor_request_valid{}(
      evict_runtime, action_ctx));
  emel::model::tensor::event::evict_tensor invalid_evict{.tensor_id = -1};
  emel::model::tensor::detail::evict_tensor_runtime invalid_evict_runtime{
      invalid_evict, status, &err};
  CHECK_FALSE(emel::model::tensor::guard::evict_tensor_request_valid{}(
      invalid_evict_runtime, action_ctx));

  emel::model::tensor::event::tensor_state captured_state{};
  emel::model::tensor::event::capture_tensor_state capture{
      .tensor_id = 3,
      .state_out = &captured_state,
  };
  emel::model::tensor::detail::capture_tensor_state_runtime capture_runtime{
      capture, status, &err};
  CHECK(emel::model::tensor::guard::capture_tensor_state_request_valid{}(
      capture_runtime, action_ctx));
  emel::model::tensor::event::capture_tensor_state null_capture{
      .tensor_id = 3,
      .state_out = nullptr,
  };
  emel::model::tensor::detail::capture_tensor_state_runtime
      null_capture_runtime{null_capture, status, &err};
  CHECK_FALSE(emel::model::tensor::guard::capture_tensor_state_request_valid{}(
      null_capture_runtime, action_ctx));
  emel::model::tensor::event::capture_tensor_state invalid_capture{
      .tensor_id = -1,
      .state_out = &captured_state,
  };
  emel::model::tensor::detail::capture_tensor_state_runtime
      invalid_capture_runtime{invalid_capture, status, &err};
  CHECK_FALSE(emel::model::tensor::guard::capture_tensor_state_request_valid{}(
      invalid_capture_runtime, action_ctx));

  emel::model::tensor::action::on_unexpected(single_runtime, action_ctx);
  CHECK(status.err ==
        emel::error::cast(emel::model::tensor::error::internal_error));
  CHECK_FALSE(status.ok);
}
