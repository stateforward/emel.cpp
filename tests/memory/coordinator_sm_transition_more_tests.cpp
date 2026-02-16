#include <boost/sml.hpp>
#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/memory/coordinator/sm.hpp"

namespace {

struct noop_queue {
  using container_type = void;

  template <class Event>
  void push(const Event &) noexcept {}
};

TEST_CASE("memory_coordinator_sm_update_success_path") {
  emel::memory::coordinator::action::context ctx{};
  noop_queue queue{};
  emel::memory::coordinator::Process process{queue};
  boost::sml::sm<
    emel::memory::coordinator::model,
    boost::sml::testing,
    emel::memory::coordinator::Process>
    machine{ctx, process};
  emel::memory::coordinator::event::memory_status status =
    emel::memory::coordinator::event::memory_status::success;
  int32_t err = EMEL_OK;

  emel::memory::coordinator::event::prepare_update request{
    .optimize = false,
    .status_out = &status,
    .error_out = &err,
  };

  CHECK(machine.process_event(request));
  CHECK(machine.process_event(emel::memory::coordinator::events::validate_done{
    .update_request = &request,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::prepare_done{
    .prepared_status = emel::memory::coordinator::event::memory_status::success,
    .update_request = &request,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::apply_done{
    .prepared_status = emel::memory::coordinator::event::memory_status::success,
    .update_request = &request,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::publish_done{
    .prepared_status = emel::memory::coordinator::event::memory_status::success,
    .update_request = &request,
  }));
  (void)machine.process_event(emel::memory::coordinator::events::memory_done{
    .status = emel::memory::coordinator::event::memory_status::success,
    .update_request = &request,
  });
}

TEST_CASE("memory_coordinator_sm_update_prepare_error_path") {
  emel::memory::coordinator::action::context ctx{};
  noop_queue queue{};
  emel::memory::coordinator::Process process{queue};
  boost::sml::sm<
    emel::memory::coordinator::model,
    boost::sml::testing,
    emel::memory::coordinator::Process>
    machine{ctx, process};
  emel::memory::coordinator::event::memory_status status =
    emel::memory::coordinator::event::memory_status::success;
  int32_t err = EMEL_OK;

  emel::memory::coordinator::event::prepare_update request{
    .optimize = false,
    .status_out = &status,
    .error_out = &err,
  };

  CHECK(machine.process_event(request));
  CHECK(machine.process_event(emel::memory::coordinator::events::validate_done{
    .update_request = &request,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::prepare_error{
    .err = EMEL_ERR_BACKEND,
    .prepared_status = emel::memory::coordinator::event::memory_status::failed_prepare,
    .update_request = &request,
  }));
  (void)machine.process_event(emel::memory::coordinator::events::memory_error{
    .err = EMEL_ERR_BACKEND,
    .status = emel::memory::coordinator::event::memory_status::failed_prepare,
    .update_request = &request,
  });
}

TEST_CASE("memory_coordinator_sm_update_apply_error_path") {
  emel::memory::coordinator::action::context ctx{};
  noop_queue queue{};
  emel::memory::coordinator::Process process{queue};
  boost::sml::sm<
    emel::memory::coordinator::model,
    boost::sml::testing,
    emel::memory::coordinator::Process>
    machine{ctx, process};
  emel::memory::coordinator::event::memory_status status =
    emel::memory::coordinator::event::memory_status::success;
  int32_t err = EMEL_OK;

  emel::memory::coordinator::event::prepare_update request{
    .optimize = false,
    .status_out = &status,
    .error_out = &err,
  };

  CHECK(machine.process_event(request));
  CHECK(machine.process_event(emel::memory::coordinator::events::validate_done{
    .update_request = &request,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::prepare_done{
    .prepared_status = emel::memory::coordinator::event::memory_status::success,
    .update_request = &request,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::apply_error{
    .err = EMEL_ERR_BACKEND,
    .prepared_status = emel::memory::coordinator::event::memory_status::failed_prepare,
    .update_request = &request,
  }));
  (void)machine.process_event(emel::memory::coordinator::events::memory_error{
    .err = EMEL_ERR_BACKEND,
    .status = emel::memory::coordinator::event::memory_status::failed_prepare,
    .update_request = &request,
  });
}

TEST_CASE("memory_coordinator_sm_update_publish_error_path") {
  emel::memory::coordinator::action::context ctx{};
  noop_queue queue{};
  emel::memory::coordinator::Process process{queue};
  boost::sml::sm<
    emel::memory::coordinator::model,
    boost::sml::testing,
    emel::memory::coordinator::Process>
    machine{ctx, process};
  emel::memory::coordinator::event::memory_status status =
    emel::memory::coordinator::event::memory_status::success;
  int32_t err = EMEL_OK;

  emel::memory::coordinator::event::prepare_update request{
    .optimize = false,
    .status_out = &status,
    .error_out = &err,
  };

  CHECK(machine.process_event(request));
  CHECK(machine.process_event(emel::memory::coordinator::events::validate_done{
    .update_request = &request,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::prepare_done{
    .prepared_status = emel::memory::coordinator::event::memory_status::success,
    .update_request = &request,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::apply_done{
    .prepared_status = emel::memory::coordinator::event::memory_status::success,
    .update_request = &request,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::publish_error{
    .err = EMEL_ERR_BACKEND,
    .prepared_status = emel::memory::coordinator::event::memory_status::failed_prepare,
    .update_request = &request,
  }));
  (void)machine.process_event(emel::memory::coordinator::events::memory_error{
    .err = EMEL_ERR_BACKEND,
    .status = emel::memory::coordinator::event::memory_status::failed_prepare,
    .update_request = &request,
  });
}

TEST_CASE("memory_coordinator_sm_batch_success_path") {
  emel::memory::coordinator::action::context ctx{};
  noop_queue queue{};
  emel::memory::coordinator::Process process{queue};
  boost::sml::sm<
    emel::memory::coordinator::model,
    boost::sml::testing,
    emel::memory::coordinator::Process>
    machine{ctx, process};
  emel::memory::coordinator::event::memory_status status =
    emel::memory::coordinator::event::memory_status::success;
  int32_t err = EMEL_OK;

  emel::memory::coordinator::event::prepare_batch request{
    .n_ubatch = 1,
    .n_ubatches_total = 1,
    .status_out = &status,
    .error_out = &err,
  };

  CHECK(machine.process_event(request));
  CHECK(machine.process_event(emel::memory::coordinator::events::validate_done{
    .batch_request = &request,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::prepare_done{
    .prepared_status = emel::memory::coordinator::event::memory_status::success,
    .batch_request = &request,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::publish_done{
    .prepared_status = emel::memory::coordinator::event::memory_status::success,
    .batch_request = &request,
  }));
  (void)machine.process_event(emel::memory::coordinator::events::memory_done{
    .status = emel::memory::coordinator::event::memory_status::success,
    .batch_request = &request,
  });
}

TEST_CASE("memory_coordinator_sm_full_success_path") {
  emel::memory::coordinator::action::context ctx{};
  noop_queue queue{};
  emel::memory::coordinator::Process process{queue};
  boost::sml::sm<
    emel::memory::coordinator::model,
    boost::sml::testing,
    emel::memory::coordinator::Process>
    machine{ctx, process};
  emel::memory::coordinator::event::memory_status status =
    emel::memory::coordinator::event::memory_status::success;
  int32_t err = EMEL_OK;

  emel::memory::coordinator::event::prepare_full request{
    .status_out = &status,
    .error_out = &err,
  };

  CHECK(machine.process_event(request));
  CHECK(machine.process_event(emel::memory::coordinator::events::validate_done{
    .full_request = &request,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::prepare_done{
    .prepared_status = emel::memory::coordinator::event::memory_status::success,
    .full_request = &request,
  }));
  CHECK(machine.process_event(emel::memory::coordinator::events::publish_done{
    .prepared_status = emel::memory::coordinator::event::memory_status::success,
    .full_request = &request,
  }));
  (void)machine.process_event(emel::memory::coordinator::events::memory_done{
    .status = emel::memory::coordinator::event::memory_status::success,
    .full_request = &request,
  });
}

}  // namespace
