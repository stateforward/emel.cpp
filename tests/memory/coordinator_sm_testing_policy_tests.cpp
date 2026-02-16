#include <boost/sml.hpp>
#include <type_traits>
#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/memory/coordinator/sm.hpp"

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
    using namespace emel::memory::coordinator;
    if constexpr (std::is_same_v<Event, event::validate_update> ||
                  std::is_same_v<Event, event::validate_batch> ||
                  std::is_same_v<Event, event::validate_full>) {
      if (ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_BACKEND;
      }
    }
  }
};

struct prepare_error_queue {
  using container_type = void;

  template <class Event>
  void push(const Event & ev) noexcept {
    using namespace emel::memory::coordinator;
    if constexpr (std::is_same_v<Event, event::prepare_update_step> ||
                  std::is_same_v<Event, event::prepare_batch_step> ||
                  std::is_same_v<Event, event::prepare_full_step>) {
      if (ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_BACKEND;
      }
    }
  }
};

struct prepare_status_queue {
  using container_type = void;
  emel::memory::coordinator::event::memory_status status =
    emel::memory::coordinator::event::memory_status::success;

  template <class Event>
  void push(const Event & ev) noexcept {
    if constexpr (requires { ev.prepared_status_out; }) {
      if (ev.prepared_status_out != nullptr) {
        *ev.prepared_status_out = status;
      }
    }
  }
};

struct apply_error_queue {
  using container_type = void;

  template <class Event>
  void push(const Event & ev) noexcept {
    if constexpr (std::is_same_v<Event, emel::memory::coordinator::event::apply_update_step>) {
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
    using namespace emel::memory::coordinator;
    if constexpr (std::is_same_v<Event, event::publish_update> ||
                  std::is_same_v<Event, event::publish_batch> ||
                  std::is_same_v<Event, event::publish_full>) {
      if (ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_BACKEND;
      }
    }
  }
};

struct prepare_no_update_publish_error_queue : publish_error_queue {
  using container_type = void;
  emel::memory::coordinator::event::memory_status status =
    emel::memory::coordinator::event::memory_status::no_update;

  template <class Event>
  void push(const Event & ev) noexcept {
    if constexpr (requires { ev.prepared_status_out; }) {
      if (ev.prepared_status_out != nullptr) {
        *ev.prepared_status_out = status;
      }
    }
    publish_error_queue::push(ev);
  }
};

}  // namespace

TEST_CASE("memory_coordinator_testing_policy_update_success_path") {
  emel::memory::coordinator::action::context ctx{};
  noop_queue queue{};
  emel::memory::coordinator::Process process{queue};
  boost::sml::sm<
    emel::memory::coordinator::model,
    boost::sml::testing,
    emel::memory::coordinator::Process> machine{ctx, process};

  emel::memory::coordinator::event::prepare_update update{
    .optimize = true,
  };

  CHECK(machine.process_event(update));
  CHECK(machine.process_event(emel::memory::coordinator::events::validate_done{
    .update_request = &update,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::prepare_done{
    .prepared_status = emel::memory::coordinator::event::memory_status::success,
    .update_request = &update,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::apply_done{
    .prepared_status = emel::memory::coordinator::event::memory_status::success,
    .update_request = &update,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::publish_done{
    .prepared_status = emel::memory::coordinator::event::memory_status::success,
    .update_request = &update,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::memory_done{
    .status = emel::memory::coordinator::event::memory_status::success,
    .update_request = &update,
  }));
}

TEST_CASE("memory_coordinator_testing_policy_update_error_path") {
  emel::memory::coordinator::action::context ctx{};
  noop_queue queue{};
  emel::memory::coordinator::Process process{queue};
  boost::sml::sm<
    emel::memory::coordinator::model,
    boost::sml::testing,
    emel::memory::coordinator::Process> machine{ctx, process};

  emel::memory::coordinator::event::prepare_update update{
    .optimize = false,
  };

  CHECK(machine.process_event(update));
  CHECK(machine.process_event(emel::memory::coordinator::events::validate_error{
    .err = EMEL_ERR_BACKEND,
    .update_request = &update,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::memory_error{
    .err = EMEL_ERR_BACKEND,
    .status = emel::memory::coordinator::event::memory_status::failed_prepare,
    .update_request = &update,
  }));
}

TEST_CASE("memory_coordinator_testing_policy_batch_success_path") {
  emel::memory::coordinator::action::context ctx{};
  noop_queue queue{};
  emel::memory::coordinator::Process process{queue};
  boost::sml::sm<
    emel::memory::coordinator::model,
    boost::sml::testing,
    emel::memory::coordinator::Process> machine{ctx, process};

  emel::memory::coordinator::event::prepare_batch batch{
    .n_ubatch = 1,
    .n_ubatches_total = 2,
  };

  CHECK(machine.process_event(batch));
  CHECK(machine.process_event(emel::memory::coordinator::events::validate_done{
    .batch_request = &batch,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::prepare_done{
    .prepared_status = emel::memory::coordinator::event::memory_status::success,
    .batch_request = &batch,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::publish_done{
    .prepared_status = emel::memory::coordinator::event::memory_status::success,
    .batch_request = &batch,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::memory_done{
    .status = emel::memory::coordinator::event::memory_status::success,
    .batch_request = &batch,
  }));
}

TEST_CASE("memory_coordinator_testing_policy_full_success_path") {
  emel::memory::coordinator::action::context ctx{};
  noop_queue queue{};
  emel::memory::coordinator::Process process{queue};
  boost::sml::sm<
    emel::memory::coordinator::model,
    boost::sml::testing,
    emel::memory::coordinator::Process> machine{ctx, process};

  emel::memory::coordinator::event::prepare_full full{};

  CHECK(machine.process_event(full));
  CHECK(machine.process_event(emel::memory::coordinator::events::validate_done{
    .full_request = &full,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::prepare_done{
    .prepared_status = emel::memory::coordinator::event::memory_status::success,
    .full_request = &full,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::publish_done{
    .prepared_status = emel::memory::coordinator::event::memory_status::success,
    .full_request = &full,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::memory_done{
    .status = emel::memory::coordinator::event::memory_status::success,
    .full_request = &full,
  }));
}

TEST_CASE("memory_coordinator_testing_policy_update_validate_error_from_queue") {
  emel::memory::coordinator::action::context ctx{};
  validate_error_queue queue{};
  emel::memory::coordinator::Process process{queue};
  boost::sml::sm<
    emel::memory::coordinator::model,
    boost::sml::testing,
    emel::memory::coordinator::Process> machine{ctx, process};

  int32_t err = EMEL_OK;
  emel::memory::coordinator::event::prepare_update update{
    .optimize = true,
    .error_out = &err,
  };

  CHECK(machine.process_event(update));
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK(machine.process_event(emel::memory::coordinator::events::validate_error{
    .err = EMEL_ERR_BACKEND,
    .update_request = &update,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::memory_error{
    .err = EMEL_ERR_BACKEND,
    .status = emel::memory::coordinator::event::memory_status::failed_prepare,
    .update_request = &update,
  }));
}

TEST_CASE("memory_coordinator_testing_policy_update_prepare_error_phase") {
  emel::memory::coordinator::action::context ctx{};
  prepare_error_queue queue{};
  emel::memory::coordinator::Process process{queue};
  boost::sml::sm<
    emel::memory::coordinator::model,
    boost::sml::testing,
    emel::memory::coordinator::Process> machine{ctx, process};

  emel::memory::coordinator::event::prepare_update update{};

  CHECK(machine.process_event(update));
  CHECK(machine.process_event(emel::memory::coordinator::events::validate_done{
    .update_request = &update,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::prepare_error{
    .err = EMEL_ERR_BACKEND,
    .prepared_status = emel::memory::coordinator::event::memory_status::failed_prepare,
    .update_request = &update,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::memory_error{
    .err = EMEL_ERR_BACKEND,
    .status = emel::memory::coordinator::event::memory_status::failed_prepare,
    .update_request = &update,
  }));
}

TEST_CASE("memory_coordinator_testing_policy_update_prepare_invalid_status") {
  emel::memory::coordinator::action::context ctx{};
  prepare_status_queue queue{
    .status = emel::memory::coordinator::event::memory_status::failed_compute,
  };
  emel::memory::coordinator::Process process{queue};
  boost::sml::sm<
    emel::memory::coordinator::model,
    boost::sml::testing,
    emel::memory::coordinator::Process> machine{ctx, process};

  emel::memory::coordinator::event::prepare_update update{};

  CHECK(machine.process_event(update));
  CHECK(machine.process_event(emel::memory::coordinator::events::validate_done{
    .update_request = &update,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::prepare_error{
    .err = EMEL_ERR_BACKEND,
    .prepared_status = emel::memory::coordinator::event::memory_status::failed_compute,
    .update_request = &update,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::memory_error{
    .err = EMEL_ERR_BACKEND,
    .status = emel::memory::coordinator::event::memory_status::failed_compute,
    .update_request = &update,
  }));
}

TEST_CASE("memory_coordinator_testing_policy_update_prepare_no_update_publish_error") {
  emel::memory::coordinator::action::context ctx{};
  prepare_no_update_publish_error_queue queue{};
  emel::memory::coordinator::Process process{queue};
  boost::sml::sm<
    emel::memory::coordinator::model,
    boost::sml::testing,
    emel::memory::coordinator::Process> machine{ctx, process};

  emel::memory::coordinator::event::prepare_update update{};

  CHECK(machine.process_event(update));
  CHECK(machine.process_event(emel::memory::coordinator::events::validate_done{
    .update_request = &update,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::prepare_done{
    .prepared_status = emel::memory::coordinator::event::memory_status::no_update,
    .update_request = &update,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::publish_error{
    .err = EMEL_ERR_BACKEND,
    .prepared_status = emel::memory::coordinator::event::memory_status::no_update,
    .update_request = &update,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::memory_error{
    .err = EMEL_ERR_BACKEND,
    .status = emel::memory::coordinator::event::memory_status::no_update,
    .update_request = &update,
  }));
}

TEST_CASE("memory_coordinator_testing_policy_update_apply_error") {
  emel::memory::coordinator::action::context ctx{};
  apply_error_queue queue{};
  emel::memory::coordinator::Process process{queue};
  boost::sml::sm<
    emel::memory::coordinator::model,
    boost::sml::testing,
    emel::memory::coordinator::Process> machine{ctx, process};

  emel::memory::coordinator::event::prepare_update update{};

  CHECK(machine.process_event(update));
  CHECK(machine.process_event(emel::memory::coordinator::events::validate_done{
    .update_request = &update,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::prepare_done{
    .prepared_status = emel::memory::coordinator::event::memory_status::success,
    .update_request = &update,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::apply_error{
    .err = EMEL_ERR_BACKEND,
    .prepared_status = emel::memory::coordinator::event::memory_status::success,
    .update_request = &update,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::memory_error{
    .err = EMEL_ERR_BACKEND,
    .status = emel::memory::coordinator::event::memory_status::success,
    .update_request = &update,
  }));
}

TEST_CASE("memory_coordinator_testing_policy_batch_publish_error") {
  emel::memory::coordinator::action::context ctx{};
  publish_error_queue queue{};
  emel::memory::coordinator::Process process{queue};
  boost::sml::sm<
    emel::memory::coordinator::model,
    boost::sml::testing,
    emel::memory::coordinator::Process> machine{ctx, process};

  emel::memory::coordinator::event::prepare_batch batch{
    .n_ubatch = 1,
    .n_ubatches_total = 1,
  };

  CHECK(machine.process_event(batch));
  CHECK(machine.process_event(emel::memory::coordinator::events::validate_done{
    .batch_request = &batch,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::prepare_done{
    .prepared_status = emel::memory::coordinator::event::memory_status::success,
    .batch_request = &batch,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::publish_error{
    .err = EMEL_ERR_BACKEND,
    .prepared_status = emel::memory::coordinator::event::memory_status::success,
    .batch_request = &batch,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::memory_error{
    .err = EMEL_ERR_BACKEND,
    .status = emel::memory::coordinator::event::memory_status::success,
    .batch_request = &batch,
  }));
}

TEST_CASE("memory_coordinator_testing_policy_full_prepare_error") {
  emel::memory::coordinator::action::context ctx{};
  prepare_error_queue queue{};
  emel::memory::coordinator::Process process{queue};
  boost::sml::sm<
    emel::memory::coordinator::model,
    boost::sml::testing,
    emel::memory::coordinator::Process> machine{ctx, process};

  emel::memory::coordinator::event::prepare_full full{};

  CHECK(machine.process_event(full));
  CHECK(machine.process_event(emel::memory::coordinator::events::validate_done{
    .full_request = &full,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::prepare_error{
    .err = EMEL_ERR_BACKEND,
    .prepared_status = emel::memory::coordinator::event::memory_status::failed_prepare,
    .full_request = &full,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::memory_error{
    .err = EMEL_ERR_BACKEND,
    .status = emel::memory::coordinator::event::memory_status::failed_prepare,
    .full_request = &full,
  }));
}

TEST_CASE("memory_coordinator_testing_policy_errored_without_request") {
  emel::memory::coordinator::action::context ctx{};
  noop_queue queue{};
  emel::memory::coordinator::Process process{queue};
  boost::sml::sm<
    emel::memory::coordinator::model,
    boost::sml::testing,
    emel::memory::coordinator::Process> machine{ctx, process};

  emel::memory::coordinator::event::prepare_update update{};

  CHECK(machine.process_event(update));
  CHECK(machine.process_event(emel::memory::coordinator::events::validate_error{
    .err = EMEL_ERR_BACKEND,
    .update_request = nullptr,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::memory_error{
    .err = EMEL_ERR_BACKEND,
    .status = emel::memory::coordinator::event::memory_status::failed_prepare,
  }));
}
