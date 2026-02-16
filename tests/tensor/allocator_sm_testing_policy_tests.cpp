#include <boost/sml.hpp>
#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/tensor/allocator/sm.hpp"

namespace {

struct noop_queue {
  using container_type = void;

  template <class Event>
  void push(const Event &) noexcept {}
};

}  // namespace

TEST_CASE("tensor_allocator_testing_policy_allocate_success_path") {
  emel::tensor::allocator::action::context ctx{};
  noop_queue queue{};
  emel::tensor::allocator::Process process{queue};
  boost::sml::sm<
    emel::tensor::allocator::model,
    boost::sml::testing,
    emel::tensor::allocator::Process> machine{ctx, process};

  emel::tensor::allocator::event::allocate_tensors allocate{};

  CHECK(machine.process_event(allocate));
  CHECK(machine.process_event(emel::tensor::allocator::events::validate_done{
    .request = &allocate,
  }));
  CHECK(machine.process_event(emel::tensor::allocator::events::scan_done{
    .request = &allocate,
  }));
  CHECK(machine.process_event(emel::tensor::allocator::events::partition_done{
    .request = &allocate,
  }));
  CHECK(machine.process_event(emel::tensor::allocator::events::allocate_ranges_done{
    .request = &allocate,
  }));
  CHECK(machine.process_event(emel::tensor::allocator::events::initialize_tensors_done{
    .request = &allocate,
  }));
  CHECK(machine.process_event(emel::tensor::allocator::events::assemble_done{
    .request = &allocate,
  }));
  CHECK(machine.process_event(emel::tensor::allocator::events::allocate_done{
    .total_bytes = 0,
    .chunk_count = 0,
    .request = &allocate,
  }));
}

TEST_CASE("tensor_allocator_testing_policy_allocate_error_path") {
  emel::tensor::allocator::action::context ctx{};
  noop_queue queue{};
  emel::tensor::allocator::Process process{queue};
  boost::sml::sm<
    emel::tensor::allocator::model,
    boost::sml::testing,
    emel::tensor::allocator::Process> machine{ctx, process};

  emel::tensor::allocator::event::allocate_tensors allocate{};

  CHECK(machine.process_event(allocate));
  CHECK(machine.process_event(emel::tensor::allocator::events::validate_error{
    .err = EMEL_ERR_BACKEND,
    .request = &allocate,
  }));
  CHECK(machine.process_event(emel::tensor::allocator::events::allocate_error{
    .err = EMEL_ERR_BACKEND,
    .request = &allocate,
  }));
}

TEST_CASE("tensor_allocator_testing_policy_release_paths") {
  emel::tensor::allocator::action::context ctx{};
  noop_queue queue{};
  emel::tensor::allocator::Process process{queue};
  boost::sml::sm<
    emel::tensor::allocator::model,
    boost::sml::testing,
    emel::tensor::allocator::Process> machine{ctx, process};

  emel::tensor::allocator::event::release release{};

  CHECK(machine.process_event(release));
  CHECK(machine.process_event(emel::tensor::allocator::events::release_done{
    .request = &release,
  }));

  CHECK(machine.process_event(release));
  CHECK(machine.process_event(emel::tensor::allocator::events::release_error{
    .err = EMEL_ERR_BACKEND,
    .request = &release,
  }));
}
