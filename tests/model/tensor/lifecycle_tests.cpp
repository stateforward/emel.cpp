#include <cstdint>

#include <array>
#include <span>
#include <string>

#include <doctest/doctest.h>

#include "emel/docs/detail.hpp"
#include "emel/io/loader/events.hpp"
#include "emel/model/data.hpp"
#include "emel/model/tensor/events.hpp"
#include "emel/model/tensor/sm.hpp"

namespace {

template <class... types, class fn>
constexpr void for_each_type(stateforward::sml::aux::type_list<types...>,
                             fn &&visitor) {
  (visitor.template operator()<types>(), ...);
}

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

TEST_CASE("model_tensor_sm_excludes_unreachable_plan_load_decision_state") {
  using machine_t = stateforward::sml::sm<emel::model::tensor::model>;
  using states = typename machine_t::states;

  bool has_unreachable_state = false;
  for_each_type(states{}, [&]<class state_type>() {
    const std::string name = emel::docs::detail::raw_type_name<state_type>();
    if (name.find("state_plan_load_decision") != std::string_view::npos) {
      has_unreachable_state = true;
    }
  });

  CHECK_FALSE(has_unreachable_state);
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
  CHECK(effects[0].kind == emel::model::tensor::effect_kind::k_none);
  CHECK(effects[0].strategy == emel::io::loader::event::strategy_kind::none);
  CHECK(effects[0].tensor_id == 0);
  CHECK(effects[0].file_index == 1u);
  CHECK(effects[0].offset == 4096u);
  CHECK(effects[0].size == 32u);
  CHECK(effects[1].kind == emel::model::tensor::effect_kind::k_none);
  CHECK(effects[1].strategy == emel::io::loader::event::strategy_kind::none);
  CHECK(effects[1].tensor_id == 1);
  CHECK(effects[1].file_index == 2u);
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
      std::span<const emel::model::tensor::effect_result>{results},
      std::span{tensors}};
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

TEST_CASE("model_tensor_plan_load_marks_io_strategy_effect_requests") {
  emel::model::tensor::sm machine{};
  owner_state owner{};
  std::array<emel::model::data::tensor_record, 1> tensors{};
  tensors[0].file_offset = 16384u;
  tensors[0].data_size = 96u;
  tensors[0].data = fake_buffer(0xE000u);
  tensors[0].file_index = 5u;
  tensors[0].type = 11;

  emel::model::tensor::event::bind_storage bind{std::span{tensors}};
  bind.on_done = {&owner, on_bind_storage_done};
  bind.on_error = {&owner, on_bind_storage_error};
  REQUIRE(machine.process_event(bind));

  std::array<emel::model::tensor::effect_request, 1> effects{};
  emel::model::tensor::event::plan_load plan{std::span{effects}};
  plan.strategy = emel::io::loader::event::strategy_kind::mapped_file;
  plan.on_done = {&owner, on_plan_load_done};
  plan.on_error = {&owner, on_plan_load_error};

  CHECK(machine.process_event(plan));
  CHECK(owner.plan_done);
  CHECK_FALSE(owner.plan_error);
  CHECK(owner.effect_count == 1u);
  CHECK(effects[0].kind == emel::model::tensor::effect_kind::k_io_load);
  CHECK(effects[0].strategy ==
        emel::io::loader::event::strategy_kind::mapped_file);
  CHECK(effects[0].tensor_id == 0);
  CHECK(effects[0].file_index == 5u);
  CHECK(effects[0].offset == 16384u);
  CHECK(effects[0].size == 96u);
  CHECK(effects[0].target == fake_buffer(0xE000u));
}

TEST_CASE("model_tensor_bulk_binding_owns_bound_record_metadata") {
  emel::model::tensor::sm machine{};
  owner_state owner{};
  std::array<emel::model::data::tensor_record, 1> tensors{};
  tensors[0].file_offset = 4096u;
  tensors[0].data_size = 32u;
  tensors[0].data = fake_buffer(0xA000u);
  tensors[0].file_index = 1u;
  tensors[0].type = 7;

  emel::model::tensor::event::bind_storage bind{std::span{tensors}};
  bind.on_done = {&owner, on_bind_storage_done};
  bind.on_error = {&owner, on_bind_storage_error};
  REQUIRE(machine.process_event(bind));

  tensors[0].file_offset = 8192u;
  tensors[0].data_size = 64u;
  tensors[0].data = fake_buffer(0xB000u);
  tensors[0].file_index = 2u;
  tensors[0].type = 8;

  std::array<emel::model::tensor::effect_request, 1> effects{};
  emel::model::tensor::event::plan_load plan{std::span{effects}};
  plan.on_done = {&owner, on_plan_load_done};
  plan.on_error = {&owner, on_plan_load_error};
  CHECK(machine.process_event(plan));
  CHECK(owner.plan_done);
  CHECK_FALSE(owner.plan_error);
  CHECK(effects[0].offset == 4096u);
  CHECK(effects[0].size == 32u);
  CHECK(effects[0].target == fake_buffer(0xA000u));
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

TEST_CASE("model_tensor_invalid_rebind_clears_prior_bulk_binding") {
  emel::model::tensor::sm machine{};
  owner_state owner{};
  std::array<emel::model::data::tensor_record, 1> tensors{};
  tensors[0].file_offset = 4096u;
  tensors[0].data_size = 32u;
  tensors[0].file_index = 1u;
  tensors[0].type = 7;

  emel::model::tensor::event::bind_storage bind{std::span{tensors}};
  bind.on_done = {&owner, on_bind_storage_done};
  bind.on_error = {&owner, on_bind_storage_error};
  REQUIRE(machine.process_event(bind));

  emel::model::tensor::event::bind_storage invalid_bind{
      std::span{tensors}.subspan(0u, 0u)};
  invalid_bind.on_done = {&owner, on_bind_storage_done};
  invalid_bind.on_error = {&owner, on_bind_storage_error};
  owner.bind_done = false;
  owner.bind_error = false;
  CHECK_FALSE(machine.process_event(invalid_bind));
  CHECK_FALSE(owner.bind_done);
  CHECK(owner.bind_error);
  CHECK(owner.err ==
        emel::error::cast(emel::model::tensor::error::invalid_request));

  std::array<emel::model::tensor::effect_request, 1> effects{};
  emel::model::tensor::event::plan_load plan{std::span{effects}};
  plan.on_done = {&owner, on_plan_load_done};
  plan.on_error = {&owner, on_plan_load_error};
  owner.plan_done = false;
  owner.plan_error = false;
  CHECK_FALSE(machine.process_event(plan));
  CHECK_FALSE(owner.plan_done);
  CHECK(owner.plan_error);
  CHECK(owner.err ==
        emel::error::cast(emel::model::tensor::error::invalid_request));

  emel::model::tensor::event::tensor_state state{};
  CHECK(machine.process_event(emel::model::tensor::event::capture_tensor_state{
      .tensor_id = 0,
      .state_out = &state,
  }));
  CHECK(state.lifecycle_state ==
        emel::model::tensor::event::lifecycle::unbound);
  CHECK(state.buffer == nullptr);
  CHECK(state.buffer_bytes == 0u);
  CHECK(state.file_offset == 0u);
  CHECK(state.data_size == 0u);
  CHECK(state.file_index == 0u);
  CHECK(state.tensor_type == 0);
}

TEST_CASE("model_tensor_bulk_storage_supports_absent_callbacks") {
  std::array<emel::model::data::tensor_record, 1> tensors{};
  tensors[0].file_offset = 4096u;
  tensors[0].data_size = 32u;
  tensors[0].file_index = 1u;
  tensors[0].type = 7;

  {
    emel::model::tensor::sm machine{};
    emel::model::tensor::event::bind_storage bind{std::span{tensors}};
    REQUIRE(machine.process_event(bind));

    std::array<emel::model::tensor::effect_request, 1> effects{};
    emel::model::tensor::event::plan_load plan{std::span{effects}};
    REQUIRE(machine.process_event(plan));

    std::array<emel::model::tensor::effect_result, 1> results{
        emel::model::tensor::effect_result{
            .kind = emel::model::tensor::effect_kind::k_none,
            .handle = const_cast<void *>(fake_buffer(0xA000u)),
            .err = emel::error::cast(emel::model::tensor::error::none),
        },
    };
    emel::model::tensor::event::apply_effect_results apply{
        std::span<const emel::model::tensor::effect_result>{results},
        std::span{tensors}};
    CHECK(machine.process_event(apply));
    CHECK(tensors[0].data == fake_buffer(0xA000u));
  }

  {
    emel::model::tensor::sm machine{};
    std::array<emel::model::tensor::effect_request, 1> effects{};
    emel::model::tensor::event::plan_load plan{std::span{effects}};
    CHECK_FALSE(machine.process_event(plan));
  }

  {
    emel::model::tensor::sm machine{};
    emel::model::tensor::event::bind_storage bind{std::span{tensors}};
    REQUIRE(machine.process_event(bind));

    std::array<emel::model::tensor::effect_request, 0> effects{};
    emel::model::tensor::event::plan_load plan{std::span{effects}};
    CHECK_FALSE(machine.process_event(plan));
  }

  {
    emel::model::tensor::sm machine{};
    emel::model::tensor::event::bind_storage bind{std::span{tensors}};
    REQUIRE(machine.process_event(bind));

    std::array<emel::model::tensor::effect_request, 1> effects{};
    emel::model::tensor::event::plan_load plan{std::span{effects}};
    REQUIRE(machine.process_event(plan));

    emel::model::tensor::event::bind_storage invalid_rebind{
        std::span{tensors}.subspan(0u, 0u)};
    CHECK_FALSE(machine.process_event(invalid_rebind));

    std::array<emel::model::tensor::effect_result, 1> results{
        emel::model::tensor::effect_result{
            .kind = emel::model::tensor::effect_kind::k_none,
            .handle = const_cast<void *>(fake_buffer(0xB000u)),
            .err = emel::error::cast(emel::model::tensor::error::none),
        },
    };
    emel::model::tensor::event::apply_effect_results apply{
        std::span<const emel::model::tensor::effect_result>{results},
        std::span{tensors}};
    CHECK(machine.process_event(apply));
    CHECK(tensors[0].data == fake_buffer(0xB000u));
  }

  {
    emel::model::tensor::sm machine{};
    emel::model::tensor::event::bind_storage bind{std::span{tensors}};
    REQUIRE(machine.process_event(bind));

    std::array<emel::model::tensor::effect_request, 1> effects{};
    emel::model::tensor::event::plan_load plan{std::span{effects}};
    REQUIRE(machine.process_event(plan));

    std::array<emel::model::tensor::effect_result, 0> results{};
    emel::model::tensor::event::apply_effect_results apply{
        std::span<const emel::model::tensor::effect_result>{results}};
    CHECK_FALSE(machine.process_event(apply));
  }

  {
    emel::model::tensor::sm machine{};
    emel::model::tensor::event::bind_storage bind{std::span{tensors}};
    REQUIRE(machine.process_event(bind));

    std::array<emel::model::tensor::effect_request, 1> effects{};
    emel::model::tensor::event::plan_load plan{std::span{effects}};
    REQUIRE(machine.process_event(plan));

    std::array<emel::model::tensor::effect_result, 1> results{
        emel::model::tensor::effect_result{
            .kind = emel::model::tensor::effect_kind::k_none,
            .handle = nullptr,
            .err = emel::error::cast(emel::model::tensor::error::out_of_memory),
        },
    };
    emel::model::tensor::event::apply_effect_results apply{
        std::span<const emel::model::tensor::effect_result>{results}};
    CHECK_FALSE(machine.process_event(apply));
  }
}

TEST_CASE("model_tensor_single_tensor_errors_support_absent_error_output") {
  emel::model::tensor::sm machine{};
  auto tensor_record = make_tensor_record();

  auto invalid_bind =
      emel::model::tensor::event::bind_tensor{0, tensor_record, nullptr, 512u};
  CHECK_FALSE(machine.process_event(invalid_bind));
}

TEST_CASE("model_tensor_rejects_rebind_while_awaiting_effects") {
  emel::model::tensor::sm machine{};
  owner_state owner{};
  std::array<emel::model::data::tensor_record, 1> tensors{};
  tensors[0].file_offset = 4096u;
  tensors[0].data_size = 32u;
  tensors[0].file_index = 1u;
  tensors[0].type = 7;

  emel::model::tensor::event::bind_storage bind{std::span{tensors}};
  bind.on_done = {&owner, on_bind_storage_done};
  bind.on_error = {&owner, on_bind_storage_error};
  REQUIRE(machine.process_event(bind));

  std::array<emel::model::tensor::effect_request, 1> effects{};
  emel::model::tensor::event::plan_load plan{std::span{effects}};
  plan.on_done = {&owner, on_plan_load_done};
  plan.on_error = {&owner, on_plan_load_error};
  REQUIRE(machine.process_event(plan));

  std::array<emel::model::data::tensor_record, 1> replacement{};
  replacement[0].file_offset = 8192u;
  replacement[0].data_size = 64u;
  replacement[0].file_index = 2u;
  replacement[0].type = 8;
  owner.bind_done = false;
  owner.bind_error = false;
  emel::model::tensor::event::bind_storage rebind{std::span{replacement}};
  rebind.on_done = {&owner, on_bind_storage_done};
  rebind.on_error = {&owner, on_bind_storage_error};
  CHECK_FALSE(machine.process_event(rebind));
  CHECK_FALSE(owner.bind_done);
  CHECK(owner.bind_error);
  CHECK(owner.err ==
        emel::error::cast(emel::model::tensor::error::invalid_request));

  std::array<emel::model::tensor::effect_result, 1> results{
      emel::model::tensor::effect_result{
          .kind = emel::model::tensor::effect_kind::k_none,
          .handle = const_cast<void *>(fake_buffer(0xA000u)),
          .err = emel::error::cast(emel::model::tensor::error::none),
      },
  };
  emel::model::tensor::event::apply_effect_results apply{
      std::span<const emel::model::tensor::effect_result>{results},
      std::span{tensors}};
  apply.on_done = {&owner, on_apply_effect_results_done};
  apply.on_error = {&owner, on_apply_effect_results_error};
  CHECK(machine.process_event(apply));
  CHECK(owner.apply_done);
  CHECK_FALSE(owner.apply_error);
  CHECK(tensors[0].data == fake_buffer(0xA000u));
  CHECK(replacement[0].data == nullptr);
}

TEST_CASE("model_tensor_rebind_clears_stale_tensor_slots") {
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
  REQUIRE(machine.process_event(bind));

  std::array<emel::model::tensor::effect_request, 2> effects{};
  emel::model::tensor::event::plan_load plan{std::span{effects}};
  plan.on_done = {&owner, on_plan_load_done};
  plan.on_error = {&owner, on_plan_load_error};
  REQUIRE(machine.process_event(plan));

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
  REQUIRE(machine.process_event(apply));

  std::array<emel::model::data::tensor_record, 1> smaller{};
  smaller[0].file_offset = 12288u;
  smaller[0].data_size = 128u;
  smaller[0].file_index = 3u;
  smaller[0].type = 9;
  emel::model::tensor::event::bind_storage rebind{std::span{smaller}};
  rebind.on_done = {&owner, on_bind_storage_done};
  rebind.on_error = {&owner, on_bind_storage_error};
  REQUIRE(machine.process_event(rebind));

  emel::model::tensor::event::tensor_state state{};
  CHECK(machine.process_event(emel::model::tensor::event::capture_tensor_state{
      .tensor_id = 1,
      .state_out = &state,
  }));
  CHECK(state.lifecycle_state ==
        emel::model::tensor::event::lifecycle::unbound);
  CHECK(state.buffer == nullptr);
  CHECK(state.buffer_bytes == 0u);
  CHECK(state.file_offset == 0u);
  CHECK(state.data_size == 0u);
  CHECK(state.file_index == 0u);
  CHECK(state.tensor_type == 0);
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
