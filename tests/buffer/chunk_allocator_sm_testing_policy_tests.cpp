#include <boost/sml.hpp>
#include <doctest/doctest.h>

#include "emel/buffer/chunk_allocator/sm.hpp"
#include "emel/emel.h"

namespace {

struct noop_queue {
  using container_type = void;

  template <class Event>
  void push(const Event &) noexcept {}
};

}  // namespace

TEST_CASE("chunk_allocator_testing_policy_configure_success_path") {
  emel::buffer::chunk_allocator::action::context ctx{};
  noop_queue queue{};
  emel::buffer::chunk_allocator::Process process{queue};
  boost::sml::sm<
    emel::buffer::chunk_allocator::model,
    boost::sml::testing,
    emel::buffer::chunk_allocator::Process> machine{ctx, process};

  emel::buffer::chunk_allocator::event::configure configure{
    .alignment = 16,
    .max_chunk_size = 64,
  };

  CHECK(machine.process_event(configure));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::validate_configure_done{
    .request = &configure,
  }));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::apply_configure_done{
    .request = &configure,
  }));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::configure_done{
    .error_out = nullptr,
    .request = &configure,
  }));
}

TEST_CASE("chunk_allocator_testing_policy_configure_error_path") {
  emel::buffer::chunk_allocator::action::context ctx{};
  noop_queue queue{};
  emel::buffer::chunk_allocator::Process process{queue};
  boost::sml::sm<
    emel::buffer::chunk_allocator::model,
    boost::sml::testing,
    emel::buffer::chunk_allocator::Process> machine{ctx, process};

  emel::buffer::chunk_allocator::event::configure configure{};
  CHECK(machine.process_event(configure));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::validate_configure_error{
    .err = EMEL_ERR_BACKEND,
    .request = &configure,
  }));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::configure_error{
    .err = EMEL_ERR_BACKEND,
    .error_out = nullptr,
    .request = &configure,
  }));
}

TEST_CASE("chunk_allocator_testing_policy_allocate_success_path") {
  emel::buffer::chunk_allocator::action::context ctx{};
  noop_queue queue{};
  emel::buffer::chunk_allocator::Process process{queue};
  boost::sml::sm<
    emel::buffer::chunk_allocator::model,
    boost::sml::testing,
    emel::buffer::chunk_allocator::Process> machine{ctx, process};

  emel::buffer::chunk_allocator::event::allocate allocate{
    .size = 16,
  };

  CHECK(machine.process_event(allocate));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::validate_allocate_done{
    .request = &allocate,
  }));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::select_block_done{
    .request = &allocate,
  }));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::ensure_chunk_done{
    .request = &allocate,
  }));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::commit_allocate_done{
    .request = &allocate,
  }));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::allocate_done{
    .chunk = 0,
    .offset = 0,
    .size = 16,
    .error_out = nullptr,
    .request = &allocate,
  }));
}

TEST_CASE("chunk_allocator_testing_policy_allocate_error_path") {
  emel::buffer::chunk_allocator::action::context ctx{};
  noop_queue queue{};
  emel::buffer::chunk_allocator::Process process{queue};
  boost::sml::sm<
    emel::buffer::chunk_allocator::model,
    boost::sml::testing,
    emel::buffer::chunk_allocator::Process> machine{ctx, process};

  emel::buffer::chunk_allocator::event::allocate allocate{};

  CHECK(machine.process_event(allocate));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::validate_allocate_error{
    .err = EMEL_ERR_BACKEND,
    .request = &allocate,
  }));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::allocate_error{
    .err = EMEL_ERR_BACKEND,
    .error_out = nullptr,
    .request = &allocate,
  }));
}

TEST_CASE("chunk_allocator_testing_policy_release_success_path") {
  emel::buffer::chunk_allocator::action::context ctx{};
  noop_queue queue{};
  emel::buffer::chunk_allocator::Process process{queue};
  boost::sml::sm<
    emel::buffer::chunk_allocator::model,
    boost::sml::testing,
    emel::buffer::chunk_allocator::Process> machine{ctx, process};

  emel::buffer::chunk_allocator::event::release release{
    .chunk = 0,
    .offset = 0,
    .size = 16,
  };

  CHECK(machine.process_event(release));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::validate_release_done{
    .request = &release,
  }));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::merge_release_done{
    .request = &release,
  }));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::release_done{
    .error_out = nullptr,
    .request = &release,
  }));
}

TEST_CASE("chunk_allocator_testing_policy_release_error_path") {
  emel::buffer::chunk_allocator::action::context ctx{};
  noop_queue queue{};
  emel::buffer::chunk_allocator::Process process{queue};
  boost::sml::sm<
    emel::buffer::chunk_allocator::model,
    boost::sml::testing,
    emel::buffer::chunk_allocator::Process> machine{ctx, process};

  emel::buffer::chunk_allocator::event::release release{};

  CHECK(machine.process_event(release));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::validate_release_error{
    .err = EMEL_ERR_BACKEND,
    .request = &release,
  }));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::release_error{
    .err = EMEL_ERR_BACKEND,
    .error_out = nullptr,
    .request = &release,
  }));
}

TEST_CASE("chunk_allocator_testing_policy_reset_paths") {
  emel::buffer::chunk_allocator::action::context ctx{};
  noop_queue queue{};
  emel::buffer::chunk_allocator::Process process{queue};
  boost::sml::sm<
    emel::buffer::chunk_allocator::model,
    boost::sml::testing,
    emel::buffer::chunk_allocator::Process> machine{ctx, process};

  emel::buffer::chunk_allocator::event::reset reset{};

  CHECK(machine.process_event(reset));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::apply_reset_done{
    .request = &reset,
  }));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::reset_done{
    .error_out = nullptr,
    .request = &reset,
  }));

  CHECK(machine.process_event(reset));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::apply_reset_error{
    .err = EMEL_ERR_BACKEND,
    .request = &reset,
  }));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::reset_error{
    .err = EMEL_ERR_BACKEND,
    .error_out = nullptr,
    .request = &reset,
  }));
}

TEST_CASE("chunk_allocator_testing_policy_configure_apply_error_path") {
  emel::buffer::chunk_allocator::action::context ctx{};
  noop_queue queue{};
  emel::buffer::chunk_allocator::Process process{queue};
  boost::sml::sm<
    emel::buffer::chunk_allocator::model,
    boost::sml::testing,
    emel::buffer::chunk_allocator::Process> machine{ctx, process};
  int32_t err = EMEL_OK;

  emel::buffer::chunk_allocator::event::configure configure{
    .alignment = 16,
    .max_chunk_size = 64,
    .error_out = &err,
  };

  CHECK(machine.process_event(configure));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::validate_configure_done{
    .request = &configure,
  }));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::apply_configure_error{
    .err = EMEL_ERR_BACKEND,
    .request = &configure,
  }));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::configure_error{
    .err = EMEL_ERR_BACKEND,
    .error_out = &err,
    .request = &configure,
  }));
}

TEST_CASE("chunk_allocator_testing_policy_allocate_select_block_error_path") {
  emel::buffer::chunk_allocator::action::context ctx{};
  noop_queue queue{};
  emel::buffer::chunk_allocator::Process process{queue};
  boost::sml::sm<
    emel::buffer::chunk_allocator::model,
    boost::sml::testing,
    emel::buffer::chunk_allocator::Process> machine{ctx, process};
  int32_t err = EMEL_OK;

  emel::buffer::chunk_allocator::event::allocate allocate{
    .size = 32,
    .error_out = &err,
  };

  CHECK(machine.process_event(allocate));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::validate_allocate_done{
    .request = &allocate,
  }));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::select_block_error{
    .err = EMEL_ERR_BACKEND,
    .request = &allocate,
  }));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::allocate_error{
    .err = EMEL_ERR_BACKEND,
    .error_out = &err,
    .request = &allocate,
  }));
}

TEST_CASE("chunk_allocator_testing_policy_allocate_ensure_chunk_error_path") {
  emel::buffer::chunk_allocator::action::context ctx{};
  noop_queue queue{};
  emel::buffer::chunk_allocator::Process process{queue};
  boost::sml::sm<
    emel::buffer::chunk_allocator::model,
    boost::sml::testing,
    emel::buffer::chunk_allocator::Process> machine{ctx, process};
  int32_t err = EMEL_OK;

  emel::buffer::chunk_allocator::event::allocate allocate{
    .size = 32,
    .error_out = &err,
  };

  CHECK(machine.process_event(allocate));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::validate_allocate_done{
    .request = &allocate,
  }));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::select_block_done{
    .request = &allocate,
  }));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::ensure_chunk_error{
    .err = EMEL_ERR_BACKEND,
    .request = &allocate,
  }));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::allocate_error{
    .err = EMEL_ERR_BACKEND,
    .error_out = &err,
    .request = &allocate,
  }));
}

TEST_CASE("chunk_allocator_testing_policy_allocate_commit_error_path") {
  emel::buffer::chunk_allocator::action::context ctx{};
  noop_queue queue{};
  emel::buffer::chunk_allocator::Process process{queue};
  boost::sml::sm<
    emel::buffer::chunk_allocator::model,
    boost::sml::testing,
    emel::buffer::chunk_allocator::Process> machine{ctx, process};
  int32_t err = EMEL_OK;

  emel::buffer::chunk_allocator::event::allocate allocate{
    .size = 32,
    .error_out = &err,
  };

  CHECK(machine.process_event(allocate));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::validate_allocate_done{
    .request = &allocate,
  }));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::select_block_done{
    .request = &allocate,
  }));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::ensure_chunk_done{
    .request = &allocate,
  }));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::commit_allocate_error{
    .err = EMEL_ERR_BACKEND,
    .request = &allocate,
  }));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::allocate_error{
    .err = EMEL_ERR_BACKEND,
    .error_out = &err,
    .request = &allocate,
  }));
}

TEST_CASE("chunk_allocator_testing_policy_release_merge_error_path") {
  emel::buffer::chunk_allocator::action::context ctx{};
  noop_queue queue{};
  emel::buffer::chunk_allocator::Process process{queue};
  boost::sml::sm<
    emel::buffer::chunk_allocator::model,
    boost::sml::testing,
    emel::buffer::chunk_allocator::Process> machine{ctx, process};
  int32_t err = EMEL_OK;

  emel::buffer::chunk_allocator::event::release release{
    .chunk = 1,
    .offset = 0,
    .size = 8,
    .error_out = &err,
  };

  CHECK(machine.process_event(release));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::validate_release_done{
    .request = &release,
  }));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::merge_release_error{
    .err = EMEL_ERR_BACKEND,
    .request = &release,
  }));
  CHECK(machine.process_event(emel::buffer::chunk_allocator::events::release_error{
    .err = EMEL_ERR_BACKEND,
    .error_out = &err,
    .request = &release,
  }));
}
