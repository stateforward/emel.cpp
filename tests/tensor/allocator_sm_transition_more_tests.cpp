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

TEST_CASE("tensor_allocator_sm_success_path") {
  emel::tensor::allocator::action::context ctx{};
  noop_queue queue{};
  emel::tensor::allocator::Process process{queue};
  boost::sml::sm<
    emel::tensor::allocator::model,
    boost::sml::testing,
    emel::tensor::allocator::Process>
    machine{ctx, process};
  emel::tensor::allocator::event::tensor_desc tensor{
    .tensor_id = 0,
    .alloc_size = 4,
  };
  emel_error_detail detail{};
  int32_t err = EMEL_OK;

  emel::tensor::allocator::event::allocate_tensors request{
    .tensors = &tensor,
    .tensor_count = 1,
    .alignment = 16,
    .max_buffer_size = 1024,
    .no_alloc = false,
    .error_out = &err,
    .detail_out = &detail,
  };

  CHECK(machine.process_event(request));
  CHECK(machine.process_event(emel::tensor::allocator::events::validate_done{.request = &request}));
  CHECK(machine.process_event(emel::tensor::allocator::events::scan_done{.request = &request}));
  CHECK(machine.process_event(emel::tensor::allocator::events::partition_done{.request = &request}));
  CHECK(machine.process_event(emel::tensor::allocator::events::allocate_ranges_done{.request = &request}));
  CHECK(machine.process_event(emel::tensor::allocator::events::initialize_tensors_done{
    .request = &request,
  }));
  CHECK(machine.process_event(emel::tensor::allocator::events::assemble_done{.request = &request}));
  (void)machine.process_event(emel::tensor::allocator::events::allocate_done{
    .total_bytes = 4,
    .chunk_count = 1,
    .request = &request,
  });
}

TEST_CASE("tensor_allocator_sm_validate_error_path") {
  emel::tensor::allocator::action::context ctx{};
  noop_queue queue{};
  emel::tensor::allocator::Process process{queue};
  boost::sml::sm<
    emel::tensor::allocator::model,
    boost::sml::testing,
    emel::tensor::allocator::Process>
    machine{ctx, process};
  emel::tensor::allocator::event::tensor_desc tensor{
    .tensor_id = 0,
    .alloc_size = 4,
  };
  emel_error_detail detail{};
  int32_t err = EMEL_OK;

  emel::tensor::allocator::event::allocate_tensors request{
    .tensors = &tensor,
    .tensor_count = 1,
    .alignment = 16,
    .max_buffer_size = 1024,
    .no_alloc = false,
    .error_out = &err,
    .detail_out = &detail,
  };

  CHECK(machine.process_event(request));
  CHECK(machine.process_event(emel::tensor::allocator::events::validate_error{
    .err = EMEL_ERR_BACKEND,
    .request = &request,
  }));
  (void)machine.process_event(emel::tensor::allocator::events::allocate_error{
    .err = EMEL_ERR_BACKEND,
    .request = &request,
  });
}

TEST_CASE("tensor_allocator_sm_scan_error_path") {
  emel::tensor::allocator::action::context ctx{};
  noop_queue queue{};
  emel::tensor::allocator::Process process{queue};
  boost::sml::sm<
    emel::tensor::allocator::model,
    boost::sml::testing,
    emel::tensor::allocator::Process>
    machine{ctx, process};
  emel::tensor::allocator::event::tensor_desc tensor{
    .tensor_id = 0,
    .alloc_size = 4,
  };
  emel_error_detail detail{};
  int32_t err = EMEL_OK;

  emel::tensor::allocator::event::allocate_tensors request{
    .tensors = &tensor,
    .tensor_count = 1,
    .alignment = 16,
    .max_buffer_size = 1024,
    .no_alloc = false,
    .error_out = &err,
    .detail_out = &detail,
  };

  CHECK(machine.process_event(request));
  CHECK(machine.process_event(emel::tensor::allocator::events::validate_done{.request = &request}));
  CHECK(machine.process_event(emel::tensor::allocator::events::scan_error{
    .err = EMEL_ERR_BACKEND,
    .request = &request,
  }));
  (void)machine.process_event(emel::tensor::allocator::events::allocate_error{
    .err = EMEL_ERR_BACKEND,
    .request = &request,
  });
}

TEST_CASE("tensor_allocator_sm_partition_error_path") {
  emel::tensor::allocator::action::context ctx{};
  noop_queue queue{};
  emel::tensor::allocator::Process process{queue};
  boost::sml::sm<
    emel::tensor::allocator::model,
    boost::sml::testing,
    emel::tensor::allocator::Process>
    machine{ctx, process};
  emel::tensor::allocator::event::tensor_desc tensor{
    .tensor_id = 0,
    .alloc_size = 4,
  };
  emel_error_detail detail{};
  int32_t err = EMEL_OK;

  emel::tensor::allocator::event::allocate_tensors request{
    .tensors = &tensor,
    .tensor_count = 1,
    .alignment = 16,
    .max_buffer_size = 1024,
    .no_alloc = false,
    .error_out = &err,
    .detail_out = &detail,
  };

  CHECK(machine.process_event(request));
  CHECK(machine.process_event(emel::tensor::allocator::events::validate_done{.request = &request}));
  CHECK(machine.process_event(emel::tensor::allocator::events::scan_done{.request = &request}));
  CHECK(machine.process_event(emel::tensor::allocator::events::partition_error{
    .err = EMEL_ERR_BACKEND,
    .request = &request,
  }));
  (void)machine.process_event(emel::tensor::allocator::events::allocate_error{
    .err = EMEL_ERR_BACKEND,
    .request = &request,
  });
}

TEST_CASE("tensor_allocator_sm_allocate_ranges_error_path") {
  emel::tensor::allocator::action::context ctx{};
  noop_queue queue{};
  emel::tensor::allocator::Process process{queue};
  boost::sml::sm<
    emel::tensor::allocator::model,
    boost::sml::testing,
    emel::tensor::allocator::Process>
    machine{ctx, process};
  emel::tensor::allocator::event::tensor_desc tensor{
    .tensor_id = 0,
    .alloc_size = 4,
  };
  emel_error_detail detail{};
  int32_t err = EMEL_OK;

  emel::tensor::allocator::event::allocate_tensors request{
    .tensors = &tensor,
    .tensor_count = 1,
    .alignment = 16,
    .max_buffer_size = 1024,
    .no_alloc = false,
    .error_out = &err,
    .detail_out = &detail,
  };

  CHECK(machine.process_event(request));
  CHECK(machine.process_event(emel::tensor::allocator::events::validate_done{.request = &request}));
  CHECK(machine.process_event(emel::tensor::allocator::events::scan_done{.request = &request}));
  CHECK(machine.process_event(emel::tensor::allocator::events::partition_done{.request = &request}));
  CHECK(machine.process_event(emel::tensor::allocator::events::allocate_ranges_error{
    .err = EMEL_ERR_BACKEND,
    .request = &request,
  }));
  (void)machine.process_event(emel::tensor::allocator::events::allocate_error{
    .err = EMEL_ERR_BACKEND,
    .request = &request,
  });
}

TEST_CASE("tensor_allocator_sm_initialize_error_path") {
  emel::tensor::allocator::action::context ctx{};
  noop_queue queue{};
  emel::tensor::allocator::Process process{queue};
  boost::sml::sm<
    emel::tensor::allocator::model,
    boost::sml::testing,
    emel::tensor::allocator::Process>
    machine{ctx, process};
  emel::tensor::allocator::event::tensor_desc tensor{
    .tensor_id = 0,
    .alloc_size = 4,
  };
  emel_error_detail detail{};
  int32_t err = EMEL_OK;

  emel::tensor::allocator::event::allocate_tensors request{
    .tensors = &tensor,
    .tensor_count = 1,
    .alignment = 16,
    .max_buffer_size = 1024,
    .no_alloc = false,
    .error_out = &err,
    .detail_out = &detail,
  };

  CHECK(machine.process_event(request));
  CHECK(machine.process_event(emel::tensor::allocator::events::validate_done{.request = &request}));
  CHECK(machine.process_event(emel::tensor::allocator::events::scan_done{.request = &request}));
  CHECK(machine.process_event(emel::tensor::allocator::events::partition_done{.request = &request}));
  CHECK(machine.process_event(emel::tensor::allocator::events::allocate_ranges_done{
    .request = &request,
  }));
  CHECK(machine.process_event(emel::tensor::allocator::events::initialize_tensors_error{
    .err = EMEL_ERR_BACKEND,
    .request = &request,
  }));
  (void)machine.process_event(emel::tensor::allocator::events::allocate_error{
    .err = EMEL_ERR_BACKEND,
    .request = &request,
  });
}

TEST_CASE("tensor_allocator_sm_assemble_error_path") {
  emel::tensor::allocator::action::context ctx{};
  noop_queue queue{};
  emel::tensor::allocator::Process process{queue};
  boost::sml::sm<
    emel::tensor::allocator::model,
    boost::sml::testing,
    emel::tensor::allocator::Process>
    machine{ctx, process};
  emel::tensor::allocator::event::tensor_desc tensor{
    .tensor_id = 0,
    .alloc_size = 4,
  };
  emel_error_detail detail{};
  int32_t err = EMEL_OK;

  emel::tensor::allocator::event::allocate_tensors request{
    .tensors = &tensor,
    .tensor_count = 1,
    .alignment = 16,
    .max_buffer_size = 1024,
    .no_alloc = false,
    .error_out = &err,
    .detail_out = &detail,
  };

  CHECK(machine.process_event(request));
  CHECK(machine.process_event(emel::tensor::allocator::events::validate_done{.request = &request}));
  CHECK(machine.process_event(emel::tensor::allocator::events::scan_done{.request = &request}));
  CHECK(machine.process_event(emel::tensor::allocator::events::partition_done{.request = &request}));
  CHECK(machine.process_event(emel::tensor::allocator::events::allocate_ranges_done{
    .request = &request,
  }));
  CHECK(machine.process_event(emel::tensor::allocator::events::initialize_tensors_done{
    .request = &request,
  }));
  CHECK(machine.process_event(emel::tensor::allocator::events::assemble_error{
    .err = EMEL_ERR_BACKEND,
    .request = &request,
  }));
  (void)machine.process_event(emel::tensor::allocator::events::allocate_error{
    .err = EMEL_ERR_BACKEND,
    .request = &request,
  });
}

TEST_CASE("tensor_allocator_sm_manual_null_request_path") {
  emel::tensor::allocator::action::context ctx{};
  noop_queue queue{};
  emel::tensor::allocator::Process process{queue};
  boost::sml::sm<
    emel::tensor::allocator::model,
    boost::sml::testing,
    emel::tensor::allocator::Process>
    machine{ctx, process};
  emel_error_detail detail{};
  int32_t err = EMEL_OK;

  emel::tensor::allocator::event::allocate_tensors request{
    .tensors = nullptr,
    .tensor_count = 0,
    .alignment = 16,
    .max_buffer_size = 1024,
    .no_alloc = true,
    .error_out = &err,
    .detail_out = &detail,
  };

  CHECK(machine.process_event(request));
  CHECK(machine.process_event(emel::tensor::allocator::events::validate_done{.request = nullptr}));
  CHECK(machine.process_event(emel::tensor::allocator::events::scan_done{.request = nullptr}));
  CHECK(machine.process_event(emel::tensor::allocator::events::partition_done{.request = nullptr}));
  CHECK(machine.process_event(emel::tensor::allocator::events::allocate_ranges_done{
    .request = nullptr,
  }));
  CHECK(machine.process_event(emel::tensor::allocator::events::initialize_tensors_done{
    .request = nullptr,
  }));
  CHECK(machine.process_event(emel::tensor::allocator::events::assemble_done{.request = nullptr}));
  CHECK(machine.process_event(emel::tensor::allocator::events::allocate_done{
    .total_bytes = 0,
    .chunk_count = 0,
    .request = nullptr,
  }));
}

TEST_CASE("tensor_allocator_sm_release_done_path") {
  emel::tensor::allocator::action::context ctx{};
  noop_queue queue{};
  emel::tensor::allocator::Process process{queue};
  boost::sml::sm<
    emel::tensor::allocator::model,
    boost::sml::testing,
    emel::tensor::allocator::Process>
    machine{ctx, process};
  emel_error_detail detail{};
  int32_t err = EMEL_OK;

  emel::tensor::allocator::event::release release{
    .error_out = &err,
    .detail_out = &detail,
  };

  CHECK(machine.process_event(release));
  CHECK(machine.process_event(emel::tensor::allocator::events::release_done{
    .request = &release,
  }));
}

TEST_CASE("tensor_allocator_sm_release_error_path") {
  emel::tensor::allocator::action::context ctx{};
  noop_queue queue{};
  emel::tensor::allocator::Process process{queue};
  boost::sml::sm<
    emel::tensor::allocator::model,
    boost::sml::testing,
    emel::tensor::allocator::Process>
    machine{ctx, process};
  emel_error_detail detail{};
  int32_t err = EMEL_ERR_BACKEND;

  emel::tensor::allocator::event::release release{
    .error_out = &err,
    .detail_out = &detail,
  };

  CHECK(machine.process_event(release));
  CHECK(machine.process_event(emel::tensor::allocator::events::release_error{
    .err = EMEL_ERR_BACKEND,
    .request = &release,
  }));
  CHECK(machine.process_event(emel::tensor::allocator::events::release_error{
    .err = EMEL_ERR_BACKEND,
    .request = &release,
  }));
}

}  // namespace
