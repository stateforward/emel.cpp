#include <array>
#include <boost/sml.hpp>
#include <type_traits>
#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/kv/cache/actions.hpp"
#include "emel/kv/cache/sm.hpp"

namespace {

struct noop_queue {
  using container_type = void;

  template <class Event>
  void push(const Event &) noexcept {}
};

struct validate_error_queue {
  using container_type = void;

  template <class Event>
  void push(const Event & ev) noexcept {
    if constexpr (std::is_same_v<Event, emel::kv::cache::event::validate>) {
      if (ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_BACKEND;
      }
    }
  }
};

struct prepare_slots_error_queue {
  using container_type = void;

  template <class Event>
  void push(const Event & ev) noexcept {
    if constexpr (std::is_same_v<Event, emel::kv::cache::event::prepare_slots>) {
      if (ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_BACKEND;
      }
    }
  }
};

struct apply_step_error_queue {
  using container_type = void;

  template <class Event>
  void push(const Event & ev) noexcept {
    if constexpr (std::is_same_v<Event, emel::kv::cache::event::apply_step>) {
      if (ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_BACKEND;
      }
    }
  }
};

struct rollback_step_error_queue {
  using container_type = void;

  template <class Event>
  void push(const Event & ev) noexcept {
    if constexpr (std::is_same_v<Event, emel::kv::cache::event::rollback_step>) {
      if (ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_BACKEND;
      }
    }
  }
};

struct publish_error_queue {
  using container_type = void;

  template <class Event>
  void push(const Event & ev) noexcept {
    if constexpr (std::is_same_v<Event, emel::kv::cache::event::publish>) {
      if (ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_BACKEND;
      }
    }
  }
};

template <class Machine>
void drive_to_prepared(Machine & machine, emel::kv::cache::event::prepare & prepare) {
  CHECK(machine.process_event(prepare));

  emel::kv::cache::events::request_ref request{};
  request.prepare = &prepare;
  CHECK(machine.process_event(emel::kv::cache::events::validate_done{.request = request}));
  CHECK(machine.process_event(emel::kv::cache::events::prepare_slots_done{.request = request}));
  CHECK(machine.process_event(emel::kv::cache::events::publish_done{.request = request}));
  CHECK(machine.process_event(emel::kv::cache::events::kv_done{}));
}

}  // namespace

TEST_CASE("kv_cache_testing_policy_prepare_success_path") {
  emel::kv::cache::action::context ctx{};
  noop_queue queue{};
  emel::kv::cache::Process process{queue};
  boost::sml::sm<emel::kv::cache::model, boost::sml::testing, emel::kv::cache::Process>
    machine{ctx, process};

  emel::kv::cache::event::prepare prepare{};
  drive_to_prepared(machine, prepare);
}

TEST_CASE("kv_cache_testing_policy_prepare_error_path") {
  emel::kv::cache::action::context ctx{};
  noop_queue queue{};
  emel::kv::cache::Process process{queue};
  boost::sml::sm<emel::kv::cache::model, boost::sml::testing, emel::kv::cache::Process>
    machine{ctx, process};

  emel::kv::cache::event::prepare prepare{};
  CHECK(machine.process_event(prepare));
  emel::kv::cache::events::request_ref request{};
  request.prepare = &prepare;
  CHECK(machine.process_event(emel::kv::cache::events::validate_error{
    .err = EMEL_ERR_BACKEND,
    .request = request,
  }));
  CHECK(machine.process_event(emel::kv::cache::events::kv_error{.err = EMEL_ERR_BACKEND}));
}

TEST_CASE("kv_cache_testing_policy_apply_success_path") {
  emel::kv::cache::action::context ctx{};
  noop_queue queue{};
  emel::kv::cache::Process process{queue};
  boost::sml::sm<emel::kv::cache::model, boost::sml::testing, emel::kv::cache::Process>
    machine{ctx, process};

  emel::kv::cache::event::prepare prepare{};
  drive_to_prepared(machine, prepare);

  emel::kv::cache::event::apply_ubatch apply{};
  int32_t kv_tokens = -1;
  apply.kv_tokens_out = &kv_tokens;
  CHECK(machine.process_event(apply));

  emel::kv::cache::events::request_ref request{};
  request.apply = &apply;
  CHECK(machine.process_event(emel::kv::cache::events::validate_done{.request = request}));
  CHECK(machine.process_event(emel::kv::cache::events::apply_done{.request = request}));
  CHECK(machine.process_event(emel::kv::cache::events::publish_done{.request = request}));
  CHECK(machine.process_event(emel::kv::cache::events::kv_done{}));
  CHECK(kv_tokens >= 0);
}

TEST_CASE("kv_cache_testing_policy_apply_error_path") {
  emel::kv::cache::action::context ctx{};
  noop_queue queue{};
  emel::kv::cache::Process process{queue};
  boost::sml::sm<emel::kv::cache::model, boost::sml::testing, emel::kv::cache::Process>
    machine{ctx, process};

  emel::kv::cache::event::prepare prepare{};
  drive_to_prepared(machine, prepare);

  emel::kv::cache::event::apply_ubatch apply{};
  CHECK(machine.process_event(apply));

  emel::kv::cache::events::request_ref request{};
  request.apply = &apply;
  CHECK(machine.process_event(emel::kv::cache::events::validate_error{
    .err = EMEL_ERR_BACKEND,
    .request = request,
  }));
  CHECK(machine.process_event(emel::kv::cache::events::kv_error{.err = EMEL_ERR_BACKEND}));
}

TEST_CASE("kv_cache_testing_policy_rollback_success_path") {
  emel::kv::cache::action::context ctx{};
  noop_queue queue{};
  emel::kv::cache::Process process{queue};
  boost::sml::sm<emel::kv::cache::model, boost::sml::testing, emel::kv::cache::Process>
    machine{ctx, process};

  emel::kv::cache::event::prepare prepare{};
  drive_to_prepared(machine, prepare);

  emel::kv::cache::event::rollback rollback{};
  CHECK(machine.process_event(rollback));

  emel::kv::cache::events::request_ref request{};
  request.rollback = &rollback;
  CHECK(machine.process_event(emel::kv::cache::events::validate_done{.request = request}));
  CHECK(machine.process_event(emel::kv::cache::events::rollback_done{.request = request}));
  CHECK(machine.process_event(emel::kv::cache::events::publish_done{.request = request}));
  CHECK(machine.process_event(emel::kv::cache::events::kv_done{}));
}

TEST_CASE("kv_cache_testing_policy_prepare_validate_error_from_queue") {
  emel::kv::cache::action::context ctx{};
  validate_error_queue queue{};
  emel::kv::cache::Process process{queue};
  boost::sml::sm<emel::kv::cache::model, boost::sml::testing, emel::kv::cache::Process>
    machine{ctx, process};

  int32_t err = EMEL_OK;
  emel::kv::cache::event::prepare prepare{.error_out = &err};

  CHECK(machine.process_event(prepare));
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK(machine.process_event(emel::kv::cache::events::validate_error{
    .err = EMEL_ERR_BACKEND,
    .request = {.prepare = &prepare},
  }));
  CHECK(machine.process_event(emel::kv::cache::events::kv_error{.err = EMEL_ERR_BACKEND}));
}

TEST_CASE("kv_cache_testing_policy_prepare_slots_error_from_queue") {
  emel::kv::cache::action::context ctx{};
  prepare_slots_error_queue queue{};
  emel::kv::cache::Process process{queue};
  boost::sml::sm<emel::kv::cache::model, boost::sml::testing, emel::kv::cache::Process>
    machine{ctx, process};

  emel::kv::cache::event::prepare prepare{};

  CHECK(machine.process_event(prepare));
  emel::kv::cache::events::request_ref request{};
  request.prepare = &prepare;
  CHECK(machine.process_event(emel::kv::cache::events::validate_done{.request = request}));
  CHECK(machine.process_event(emel::kv::cache::events::prepare_slots_error{
    .err = EMEL_ERR_BACKEND,
    .request = request,
  }));
  CHECK(machine.process_event(emel::kv::cache::events::kv_error{.err = EMEL_ERR_BACKEND}));
}

TEST_CASE("kv_cache_testing_policy_apply_step_error_from_queue") {
  emel::kv::cache::action::context ctx{};
  apply_step_error_queue queue{};
  emel::kv::cache::Process process{queue};
  boost::sml::sm<emel::kv::cache::model, boost::sml::testing, emel::kv::cache::Process>
    machine{ctx, process};

  emel::kv::cache::event::prepare prepare{};
  drive_to_prepared(machine, prepare);

  emel::kv::cache::event::apply_ubatch apply{};
  CHECK(machine.process_event(apply));

  emel::kv::cache::events::request_ref request{};
  request.apply = &apply;
  CHECK(machine.process_event(emel::kv::cache::events::validate_done{.request = request}));
  CHECK(machine.process_event(emel::kv::cache::events::apply_error{
    .err = EMEL_ERR_BACKEND,
    .request = request,
  }));
  CHECK(machine.process_event(emel::kv::cache::events::kv_error{.err = EMEL_ERR_BACKEND}));
}

TEST_CASE("kv_cache_testing_policy_rollback_step_error_from_queue") {
  emel::kv::cache::action::context ctx{};
  rollback_step_error_queue queue{};
  emel::kv::cache::Process process{queue};
  boost::sml::sm<emel::kv::cache::model, boost::sml::testing, emel::kv::cache::Process>
    machine{ctx, process};

  emel::kv::cache::event::prepare prepare{};
  drive_to_prepared(machine, prepare);

  emel::kv::cache::event::rollback rollback{};
  CHECK(machine.process_event(rollback));

  emel::kv::cache::events::request_ref request{};
  request.rollback = &rollback;
  CHECK(machine.process_event(emel::kv::cache::events::validate_done{.request = request}));
  CHECK(machine.process_event(emel::kv::cache::events::rollback_error{
    .err = EMEL_ERR_BACKEND,
    .request = request,
  }));
  CHECK(machine.process_event(emel::kv::cache::events::kv_error{.err = EMEL_ERR_BACKEND}));
}

TEST_CASE("kv_cache_testing_policy_publish_error_from_queue") {
  emel::kv::cache::action::context ctx{};
  publish_error_queue queue{};
  emel::kv::cache::Process process{queue};
  boost::sml::sm<emel::kv::cache::model, boost::sml::testing, emel::kv::cache::Process>
    machine{ctx, process};

  emel::kv::cache::event::prepare prepare{};
  CHECK(machine.process_event(prepare));

  emel::kv::cache::events::request_ref request{};
  request.prepare = &prepare;
  CHECK(machine.process_event(emel::kv::cache::events::validate_done{.request = request}));
  CHECK(machine.process_event(emel::kv::cache::events::prepare_slots_done{.request = request}));
  CHECK(machine.process_event(emel::kv::cache::events::publish_error{
    .err = EMEL_ERR_BACKEND,
    .request = request,
  }));
  CHECK(machine.process_event(emel::kv::cache::events::kv_error{.err = EMEL_ERR_BACKEND}));
}

TEST_CASE("kv_cache_testing_policy_done_copies_prepare_outputs") {
  emel::kv::cache::action::context ctx{};
  noop_queue queue{};
  emel::kv::cache::Process process{queue};
  boost::sml::sm<emel::kv::cache::model, boost::sml::testing, emel::kv::cache::Process>
    machine{ctx, process};

  std::array<int32_t, 2> offsets = {{0, 0}};
  int32_t ubatch_count = 0;
  int32_t err = EMEL_OK;
  emel::kv::cache::event::prepare prepare{
    .slot_offsets_out = offsets.data(),
    .slot_offsets_capacity = static_cast<int32_t>(offsets.size()),
    .ubatch_count_out = &ubatch_count,
    .error_out = &err,
  };

  CHECK(machine.process_event(prepare));
  emel::kv::cache::events::request_ref request{};
  request.prepare = &prepare;
  CHECK(machine.process_event(emel::kv::cache::events::validate_done{.request = request}));
  CHECK(machine.process_event(emel::kv::cache::events::prepare_slots_done{.request = request}));
  ctx.planned_ubatch_count = 2;
  ctx.slot_offsets[0] = 4;
  ctx.slot_offsets[1] = 8;
  CHECK(machine.process_event(emel::kv::cache::events::publish_done{.request = request}));
  CHECK(err == EMEL_OK);
  CHECK(ubatch_count == 2);
  CHECK(offsets[0] == 4);
  CHECK(offsets[1] == 8);
  CHECK(machine.process_event(emel::kv::cache::events::kv_done{}));
}

TEST_CASE("kv_cache_testing_policy_done_reports_prepare_capacity_error") {
  emel::kv::cache::action::context ctx{};
  noop_queue queue{};
  emel::kv::cache::Process process{queue};
  boost::sml::sm<emel::kv::cache::model, boost::sml::testing, emel::kv::cache::Process>
    machine{ctx, process};

  std::array<int32_t, 1> offsets = {{0}};
  int32_t err = EMEL_OK;
  emel::kv::cache::event::prepare prepare{
    .slot_offsets_out = offsets.data(),
    .slot_offsets_capacity = static_cast<int32_t>(offsets.size()),
    .error_out = &err,
  };

  CHECK(machine.process_event(prepare));
  emel::kv::cache::events::request_ref request{};
  request.prepare = &prepare;
  CHECK(machine.process_event(emel::kv::cache::events::validate_done{.request = request}));
  CHECK(machine.process_event(emel::kv::cache::events::prepare_slots_done{.request = request}));
  ctx.planned_ubatch_count = 2;
  ctx.slot_offsets[0] = 4;
  ctx.slot_offsets[1] = 8;
  CHECK(machine.process_event(emel::kv::cache::events::publish_done{.request = request}));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(machine.process_event(emel::kv::cache::events::kv_error{.err = EMEL_ERR_INVALID_ARGUMENT}));
}

TEST_CASE("kv_cache_testing_policy_done_sets_rollback_error_out") {
  emel::kv::cache::action::context ctx{};
  noop_queue queue{};
  emel::kv::cache::Process process{queue};
  boost::sml::sm<emel::kv::cache::model, boost::sml::testing, emel::kv::cache::Process>
    machine{ctx, process};

  emel::kv::cache::event::prepare prepare{};
  drive_to_prepared(machine, prepare);

  int32_t err = EMEL_OK;
  emel::kv::cache::event::rollback rollback{.error_out = &err};
  CHECK(machine.process_event(rollback));

  emel::kv::cache::events::request_ref request{};
  request.rollback = &rollback;
  CHECK(machine.process_event(emel::kv::cache::events::validate_done{.request = request}));
  CHECK(machine.process_event(emel::kv::cache::events::rollback_done{.request = request}));
  CHECK(machine.process_event(emel::kv::cache::events::publish_done{.request = request}));
  CHECK(err == EMEL_OK);
  CHECK(machine.process_event(emel::kv::cache::events::kv_done{}));
}
