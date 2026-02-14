#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/tensor/allocator/actions.hpp"
#include "emel/tensor/allocator/guards.hpp"
#include "emel/tensor/allocator/sm.hpp"

namespace {

using tensor_desc = emel::tensor::allocator::event::tensor_desc;

struct backend_stub {
  int32_t alloc_calls = 0;
  int32_t free_calls = 0;
  int32_t init_calls = 0;
  int32_t view_calls = 0;
  int32_t assemble_calls = 0;
  bool fail_alloc = false;
  int32_t fail_alloc_after = -1;
  bool fail_init = false;
  bool fail_assemble = false;
};

void * stub_alloc_buffer(void * ctx, int32_t bytes) noexcept {
  auto * stub = static_cast<backend_stub *>(ctx);
  stub->alloc_calls += 1;
  if (
      stub->fail_alloc ||
      (stub->fail_alloc_after >= 0 && stub->alloc_calls > stub->fail_alloc_after) || bytes <= 0) {
    return nullptr;
  }
  return std::malloc(static_cast<size_t>(bytes));
}

void stub_free_buffer(void * ctx, void * buffer) noexcept {
  auto * stub = static_cast<backend_stub *>(ctx);
  stub->free_calls += 1;
  std::free(buffer);
}

int32_t stub_init_tensor(
    void * ctx, const tensor_desc *, void * buffer, int32_t offset_bytes) noexcept {
  auto * stub = static_cast<backend_stub *>(ctx);
  stub->init_calls += 1;
  if (stub->fail_init || buffer == nullptr || offset_bytes < 0) {
    return EMEL_ERR_BACKEND;
  }
  return EMEL_OK;
}

int32_t stub_init_view_tensor(void * ctx, const tensor_desc *) noexcept {
  auto * stub = static_cast<backend_stub *>(ctx);
  stub->view_calls += 1;
  return EMEL_OK;
}

int32_t stub_init_view_tensor_fail(void * ctx, const tensor_desc *) noexcept {
  auto * stub = static_cast<backend_stub *>(ctx);
  stub->view_calls += 1;
  return EMEL_ERR_BACKEND;
}

void * stub_assemble_buffers(void * ctx, void * const *, int32_t buffer_count) noexcept {
  auto * stub = static_cast<backend_stub *>(ctx);
  stub->assemble_calls += 1;
  if (stub->fail_assemble || buffer_count <= 1) {
    return nullptr;
  }
  return stub;
}

TEST_CASE("tensor_allocator_starts_idle") {
  emel::tensor::allocator::sm machine{};
  int32_t state_count = 0;
  machine.visit_current_states([&](auto) { state_count += 1; });
  CHECK(state_count == 1);
}

TEST_CASE("tensor_allocator_allocate_tensors_no_alloc_success") {
  emel::tensor::allocator::sm machine{};

  std::array<tensor_desc, 5> tensors = {{
    tensor_desc{
      .tensor_id = 1,
      .alloc_size = 32,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
    tensor_desc{
      .tensor_id = 2,
      .alloc_size = 20,
      .src_ids = {{1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
    tensor_desc{
      .tensor_id = 3,
      .alloc_size = 0,
      .src_ids = {{1, -1, -1, -1}},
      .is_view = true,
      .view_src_id = 1,
      .has_external_data = false,
    },
    tensor_desc{
      .tensor_id = 4,
      .alloc_size = 96,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = true,
    },
    tensor_desc{
      .tensor_id = 5,
      .alloc_size = 40,
      .src_ids = {{2, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
  }};

  int32_t total = 0;
  int32_t chunk_count = 0;
  int32_t error = -1;
  std::array<int32_t, 4> chunks = {{0, 0, 0, 0}};

  CHECK(machine.process_event(emel::tensor::allocator::event::allocate_tensors{
    .tensors = tensors.data(),
    .tensor_count = static_cast<int32_t>(tensors.size()),
    .alignment = 16,
    .max_buffer_size = 64,
    .no_alloc = true,
    .total_size_out = &total,
    .chunk_sizes_out = chunks.data(),
    .chunk_sizes_out_count = static_cast<int32_t>(chunks.size()),
    .chunk_count_out = &chunk_count,
    .error_out = &error,
  }));

  CHECK(error == EMEL_OK);
  CHECK(total == 112);
  CHECK(chunk_count == 2);
  CHECK(chunks[0] == 64);
  CHECK(chunks[1] == 48);
  CHECK(machine.total_bytes() == 112);
  CHECK(machine.chunk_count() == 2);
}

TEST_CASE("tensor_allocator_allows_single_tensor_larger_than_max_buffer") {
  emel::tensor::allocator::sm machine{};
  backend_stub backend{};
  std::array<tensor_desc, 2> tensors = {{
    tensor_desc{
      .tensor_id = 11,
      .alloc_size = 96,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
    tensor_desc{
      .tensor_id = 12,
      .alloc_size = 16,
      .src_ids = {{11, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
  }};

  int32_t total = 0;
  int32_t chunk_count = 0;
  std::array<int32_t, 2> chunks = {{0, 0}};
  void * result = nullptr;

  CHECK(machine.process_event(emel::tensor::allocator::event::allocate_tensors{
    .tensors = tensors.data(),
    .tensor_count = static_cast<int32_t>(tensors.size()),
    .alignment = 16,
    .max_buffer_size = 64,
    .no_alloc = false,
    .backend_ctx = &backend,
    .alloc_buffer = stub_alloc_buffer,
    .free_buffer = stub_free_buffer,
    .init_tensor = stub_init_tensor,
    .init_view_tensor = stub_init_view_tensor,
    .assemble_buffers = stub_assemble_buffers,
    .result_buffer_out = &result,
    .total_size_out = &total,
    .chunk_sizes_out = chunks.data(),
    .chunk_sizes_out_count = static_cast<int32_t>(chunks.size()),
    .chunk_count_out = &chunk_count,
  }));

  CHECK(total == 112);
  CHECK(chunk_count == 2);
  CHECK(chunks[0] == 96);
  CHECK(chunks[1] == 16);
  CHECK(result == &backend);
  CHECK(backend.alloc_calls == 2);
  CHECK(backend.init_calls == 2);
  CHECK(backend.assemble_calls == 1);
  CHECK(machine.process_event(emel::tensor::allocator::event::release{}));
}

TEST_CASE("tensor_allocator_reports_invalid_arguments_and_view_errors") {
  emel::tensor::allocator::sm machine{};

  int32_t error = EMEL_OK;
  CHECK_FALSE(machine.process_event(emel::tensor::allocator::event::allocate_tensors{
    .tensors = nullptr,
    .tensor_count = 1,
    .alignment = 16,
    .max_buffer_size = 64,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);

  std::array<tensor_desc, 1> bad_view = {{
    tensor_desc{
      .tensor_id = 21,
      .alloc_size = 0,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = true,
      .view_src_id = 999,
      .has_external_data = false,
    },
  }};

  backend_stub backend{};
  error = EMEL_OK;
  CHECK_FALSE(machine.process_event(emel::tensor::allocator::event::allocate_tensors{
    .tensors = bad_view.data(),
    .tensor_count = static_cast<int32_t>(bad_view.size()),
    .alignment = 16,
    .max_buffer_size = 64,
    .backend_ctx = &backend,
    .alloc_buffer = stub_alloc_buffer,
    .free_buffer = stub_free_buffer,
    .init_tensor = stub_init_tensor,
    .init_view_tensor = stub_init_view_tensor,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("tensor_allocator_reports_assemble_output_mismatch") {
  emel::tensor::allocator::sm machine{};
  backend_stub backend{};
  std::array<tensor_desc, 2> tensors = {{
    tensor_desc{
      .tensor_id = 31,
      .alloc_size = 32,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
    tensor_desc{
      .tensor_id = 32,
      .alloc_size = 64,
      .src_ids = {{31, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
  }};

  int32_t error = EMEL_OK;
  std::array<int32_t, 1> chunks = {{0}};
  CHECK_FALSE(machine.process_event(emel::tensor::allocator::event::allocate_tensors{
    .tensors = tensors.data(),
    .tensor_count = static_cast<int32_t>(tensors.size()),
    .alignment = 16,
    .max_buffer_size = 64,
    .backend_ctx = &backend,
    .alloc_buffer = stub_alloc_buffer,
    .free_buffer = stub_free_buffer,
    .init_tensor = stub_init_tensor,
    .result_buffer_out = nullptr,
    .chunk_sizes_out = chunks.data(),
    .chunk_sizes_out_count = 0,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("tensor_allocator_release_resets_runtime") {
  emel::tensor::allocator::sm machine{};
  backend_stub backend{};
  std::array<tensor_desc, 1> tensors = {{
    tensor_desc{
      .tensor_id = 41,
      .alloc_size = 32,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
  }};

  int32_t total = 0;
  CHECK(machine.process_event(emel::tensor::allocator::event::allocate_tensors{
    .tensors = tensors.data(),
    .tensor_count = static_cast<int32_t>(tensors.size()),
    .alignment = 16,
    .max_buffer_size = 64,
    .backend_ctx = &backend,
    .alloc_buffer = stub_alloc_buffer,
    .free_buffer = stub_free_buffer,
    .init_tensor = stub_init_tensor,
    .total_size_out = &total,
  }));
  CHECK(total == 32);

  CHECK(machine.process_event(emel::tensor::allocator::event::release{}));
  CHECK(machine.total_bytes() == 0);
  CHECK(machine.chunk_count() == 0);
  CHECK(backend.free_calls >= 1);
}

TEST_CASE("tensor_allocator_guard_and_detail_helpers_cover_edges") {
  emel::tensor::allocator::action::context c{};
  CHECK(emel::tensor::allocator::guard::no_error{}(emel::tensor::allocator::events::validate_done{}, c));
  CHECK(
      emel::tensor::allocator::guard::has_error{}(
          emel::tensor::allocator::events::validate_error{.err = EMEL_ERR_BACKEND}, c));

  CHECK(emel::tensor::allocator::action::detail::normalize_error(7, 9) == 7);
  CHECK(
      emel::tensor::allocator::action::detail::normalize_error(0, EMEL_ERR_INVALID_ARGUMENT) ==
      EMEL_ERR_INVALID_ARGUMENT);
  CHECK(emel::tensor::allocator::action::detail::normalize_error(0, 0) == EMEL_ERR_BACKEND);

  int32_t aligned = -1;
  CHECK(emel::tensor::allocator::action::detail::align_up(0, 16, aligned));
  CHECK(aligned == 0);
  CHECK_FALSE(emel::tensor::allocator::action::detail::align_up(
      std::numeric_limits<int32_t>::max(), 16, aligned));
}

TEST_CASE("tensor_allocator_validate_rejects_invalid_combinations") {
  std::array<tensor_desc, 1> tensors = {{
    tensor_desc{
      .tensor_id = 51,
      .alloc_size = 16,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
  }};

  emel::tensor::allocator::action::context c{};
  c.tensors = tensors.data();
  c.tensor_count = 1;
  c.alignment = 16;
  c.max_buffer_size = 64;
  c.no_alloc = true;

  int32_t phase_error = EMEL_OK;
  emel::tensor::allocator::action::run_validate(
      emel::tensor::allocator::event::validate{.error_out = &phase_error}, c);
  CHECK(phase_error == EMEL_OK);

  c.tensor_count = -1;
  emel::tensor::allocator::action::run_validate(
      emel::tensor::allocator::event::validate{.error_out = &phase_error}, c);
  CHECK(phase_error == EMEL_ERR_INVALID_ARGUMENT);

  c.tensor_count = emel::tensor::allocator::action::k_max_tensors + 1;
  emel::tensor::allocator::action::run_validate(
      emel::tensor::allocator::event::validate{.error_out = &phase_error}, c);
  CHECK(phase_error == EMEL_ERR_INVALID_ARGUMENT);

  c.tensor_count = 1;
  c.tensors = nullptr;
  emel::tensor::allocator::action::run_validate(
      emel::tensor::allocator::event::validate{.error_out = &phase_error}, c);
  CHECK(phase_error == EMEL_ERR_INVALID_ARGUMENT);

  c.tensors = tensors.data();
  c.alignment = 3;
  emel::tensor::allocator::action::run_validate(
      emel::tensor::allocator::event::validate{.error_out = &phase_error}, c);
  CHECK(phase_error == EMEL_ERR_INVALID_ARGUMENT);

  c.alignment = 16;
  c.max_buffer_size = 0;
  emel::tensor::allocator::action::run_validate(
      emel::tensor::allocator::event::validate{.error_out = &phase_error}, c);
  CHECK(phase_error == EMEL_ERR_INVALID_ARGUMENT);

  c.max_buffer_size = 64;
  c.chunk_sizes_out_count = -1;
  emel::tensor::allocator::action::run_validate(
      emel::tensor::allocator::event::validate{.error_out = &phase_error}, c);
  CHECK(phase_error == EMEL_ERR_INVALID_ARGUMENT);

  c.chunk_sizes_out_count = 1;
  c.chunk_sizes_out = nullptr;
  emel::tensor::allocator::action::run_validate(
      emel::tensor::allocator::event::validate{.error_out = &phase_error}, c);
  CHECK(phase_error == EMEL_ERR_INVALID_ARGUMENT);

  c.chunk_sizes_out_count = 0;
  c.no_alloc = false;
  c.alloc_buffer = nullptr;
  c.free_buffer = nullptr;
  c.init_tensor = nullptr;
  emel::tensor::allocator::action::run_validate(
      emel::tensor::allocator::event::validate{.error_out = &phase_error}, c);
  CHECK(phase_error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("tensor_allocator_scan_and_allocate_paths_cover_errors") {
  std::array<tensor_desc, 2> duplicate_ids = {{
    tensor_desc{
      .tensor_id = 61,
      .alloc_size = 16,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
    tensor_desc{
      .tensor_id = 61,
      .alloc_size = 16,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
  }};
  emel::tensor::allocator::action::context c{};
  c.tensors = duplicate_ids.data();
  c.tensor_count = static_cast<int32_t>(duplicate_ids.size());
  c.alignment = 16;
  int32_t phase_error = EMEL_OK;
  emel::tensor::allocator::action::run_scan_tensors(
      emel::tensor::allocator::event::scan_tensors{.error_out = &phase_error}, c);
  CHECK(phase_error == EMEL_ERR_INVALID_ARGUMENT);

  std::array<tensor_desc, 1> bad_view = {{
    tensor_desc{
      .tensor_id = 62,
      .alloc_size = 0,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = true,
      .view_src_id = -1,
      .has_external_data = false,
    },
  }};
  c = {};
  c.tensors = bad_view.data();
  c.tensor_count = 1;
  c.alignment = 16;
  emel::tensor::allocator::action::run_scan_tensors(
      emel::tensor::allocator::event::scan_tensors{.error_out = &phase_error}, c);
  CHECK(phase_error == EMEL_ERR_INVALID_ARGUMENT);

  std::array<tensor_desc, 1> overflow_size = {{
    tensor_desc{
      .tensor_id = 63,
      .alloc_size = std::numeric_limits<int32_t>::max(),
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
  }};
  c = {};
  c.tensors = overflow_size.data();
  c.tensor_count = 1;
  c.alignment = 16;
  emel::tensor::allocator::action::run_scan_tensors(
      emel::tensor::allocator::event::scan_tensors{.error_out = &phase_error}, c);
  CHECK(phase_error == EMEL_ERR_BACKEND);

  backend_stub backend{};
  c = {};
  c.no_alloc = false;
  c.backend_ctx = &backend;
  c.alloc_buffer = stub_alloc_buffer;
  c.free_buffer = stub_free_buffer;
  c.chunk_count = 1;
  c.chunk_sizes[0] = 0;
  emel::tensor::allocator::action::run_allocate_ranges(
      emel::tensor::allocator::event::allocate_ranges{.error_out = &phase_error}, c);
  CHECK(phase_error == EMEL_ERR_BACKEND);

  backend = {};
  backend.fail_alloc_after = 1;
  c = {};
  c.no_alloc = false;
  c.backend_ctx = &backend;
  c.alloc_buffer = stub_alloc_buffer;
  c.free_buffer = stub_free_buffer;
  c.chunk_count = 2;
  c.chunk_sizes[0] = 16;
  c.chunk_sizes[1] = 16;
  emel::tensor::allocator::action::run_allocate_ranges(
      emel::tensor::allocator::event::allocate_ranges{.error_out = &phase_error}, c);
  CHECK(phase_error == EMEL_ERR_BACKEND);
  CHECK(backend.free_calls == 1);
}

TEST_CASE("tensor_allocator_initialize_and_assemble_cover_error_paths") {
  backend_stub backend{};
  std::array<tensor_desc, 1> tensors = {{
    tensor_desc{
      .tensor_id = 71,
      .alloc_size = 16,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
  }};

  emel::tensor::allocator::action::context c{};
  c.no_alloc = false;
  c.backend_ctx = &backend;
  c.free_buffer = stub_free_buffer;
  c.init_tensor = stub_init_tensor;
  c.tensors = tensors.data();
  c.tensor_count = 1;
  c.chunk_count = 1;
  c.effective_sizes[0] = 16;
  c.tensor_chunk_ids[0] = 0;
  c.tensor_offsets[0] = 0;
  c.allocated_buffers[0] = std::malloc(16);
  backend.fail_init = true;
  int32_t phase_error = EMEL_OK;
  emel::tensor::allocator::action::run_initialize_tensors(
      emel::tensor::allocator::event::initialize_tensors{.error_out = &phase_error}, c);
  CHECK(phase_error == EMEL_ERR_BACKEND);
  CHECK(c.allocated_buffers[0] == nullptr);
  CHECK(backend.free_calls == 1);

  backend = {};
  std::array<tensor_desc, 1> view_tensor = {{
    tensor_desc{
      .tensor_id = 72,
      .alloc_size = 0,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = true,
      .view_src_id = 99,
      .has_external_data = false,
    },
  }};
  c = {};
  c.no_alloc = false;
  c.backend_ctx = &backend;
  c.free_buffer = stub_free_buffer;
  c.init_tensor = stub_init_tensor;
  c.init_view_tensor = stub_init_view_tensor_fail;
  c.tensors = view_tensor.data();
  c.tensor_count = 1;
  c.tensor_ids[0] = 99;
  emel::tensor::allocator::action::run_initialize_tensors(
      emel::tensor::allocator::event::initialize_tensors{.error_out = &phase_error}, c);
  CHECK(phase_error == EMEL_ERR_BACKEND);
  CHECK(backend.view_calls == 1);

  backend = {};
  void * result = reinterpret_cast<void *>(0x1);
  std::array<int32_t, 2> chunks = {{0, 0}};
  c = {};
  c.no_alloc = false;
  c.backend_ctx = &backend;
  c.free_buffer = stub_free_buffer;
  c.chunk_count = 2;
  c.chunk_sizes[0] = 16;
  c.chunk_sizes[1] = 16;
  c.total_bytes = 32;
  c.chunk_sizes_out = chunks.data();
  c.chunk_sizes_out_count = static_cast<int32_t>(chunks.size());
  c.result_buffer_out = &result;
  c.allocated_buffers[0] = std::malloc(16);
  c.allocated_buffers[1] = std::malloc(16);
  c.assemble_buffers = nullptr;
  emel::tensor::allocator::action::run_assemble(
      emel::tensor::allocator::event::assemble{.error_out = &phase_error}, c);
  CHECK(phase_error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(result == nullptr);
  CHECK(backend.free_calls == 2);

  backend = {};
  result = reinterpret_cast<void *>(0x2);
  c = {};
  c.no_alloc = false;
  c.backend_ctx = &backend;
  c.free_buffer = stub_free_buffer;
  c.chunk_count = 2;
  c.chunk_sizes[0] = 16;
  c.chunk_sizes[1] = 16;
  c.total_bytes = 32;
  c.chunk_sizes_out = chunks.data();
  c.chunk_sizes_out_count = static_cast<int32_t>(chunks.size());
  c.result_buffer_out = &result;
  c.allocated_buffers[0] = std::malloc(16);
  c.allocated_buffers[1] = std::malloc(16);
  c.assemble_buffers = stub_assemble_buffers;
  backend.fail_assemble = true;
  emel::tensor::allocator::action::run_assemble(
      emel::tensor::allocator::event::assemble{.error_out = &phase_error}, c);
  CHECK(phase_error == EMEL_ERR_BACKEND);
  CHECK(result == nullptr);
  CHECK(backend.free_calls == 2);
}

TEST_CASE("tensor_allocator_sm_wrapper_state_edge_cases") {
  std::array<tensor_desc, 1> tensors = {{
    tensor_desc{
      .tensor_id = 81,
      .alloc_size = 16,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
  }};

  emel::tensor::allocator::sm machine{};
  auto & base = static_cast<emel::tensor::allocator::sm::base_type &>(machine);
  CHECK(base.process_event(emel::tensor::allocator::event::allocate_tensors{
    .tensors = tensors.data(),
    .tensor_count = static_cast<int32_t>(tensors.size()),
    .alignment = 16,
    .max_buffer_size = 64,
    .no_alloc = true,
  }));
  CHECK_FALSE(machine.process_event(emel::tensor::allocator::event::allocate_tensors{
    .tensors = tensors.data(),
    .tensor_count = static_cast<int32_t>(tensors.size()),
    .alignment = 16,
    .max_buffer_size = 64,
    .no_alloc = true,
  }));

  emel::tensor::allocator::sm release_machine{};
  auto & release_base = static_cast<emel::tensor::allocator::sm::base_type &>(release_machine);
  CHECK(release_base.process_event(emel::tensor::allocator::event::release{}));
  CHECK_FALSE(release_machine.process_event(emel::tensor::allocator::event::release{}));
  CHECK(release_base.process_event(emel::tensor::allocator::events::release_done{}));
}

}  // namespace
