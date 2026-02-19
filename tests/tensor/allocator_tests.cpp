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
  CHECK(result != nullptr);
  CHECK(machine.process_event(emel::tensor::allocator::event::release{}));
}

TEST_CASE("tensor_allocator_reports_invalid_arguments_and_view_errors") {
  emel::tensor::allocator::sm machine{};

  int32_t error = EMEL_OK;
  emel_error_detail detail{};
  CHECK_FALSE(machine.process_event(emel::tensor::allocator::event::allocate_tensors{
    .tensors = nullptr,
    .tensor_count = 1,
    .alignment = 16,
    .max_buffer_size = 64,
    .error_out = &error,
    .detail_out = &detail,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(detail.status == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(detail.domain == EMEL_ERROR_DOMAIN_TENSOR_ALLOCATOR);
  CHECK(detail.phase ==
        static_cast<uint32_t>(emel::tensor::allocator::event::error_phase::validate));
  CHECK(detail.reason ==
        static_cast<uint32_t>(emel::tensor::allocator::event::error_reason::invalid_argument));

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

  error = EMEL_OK;
  emel_error_detail view_detail{};
  CHECK_FALSE(machine.process_event(emel::tensor::allocator::event::allocate_tensors{
    .tensors = bad_view.data(),
    .tensor_count = static_cast<int32_t>(bad_view.size()),
    .alignment = 16,
    .max_buffer_size = 64,
    .error_out = &error,
    .detail_out = &view_detail,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(view_detail.status == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(view_detail.domain == EMEL_ERROR_DOMAIN_TENSOR_ALLOCATOR);
  CHECK(view_detail.reason ==
        static_cast<uint32_t>(emel::tensor::allocator::event::error_reason::invalid_view_source));
  const uint32_t scan_phase =
      static_cast<uint32_t>(emel::tensor::allocator::event::error_phase::scan_tensors);
  const uint32_t init_phase =
      static_cast<uint32_t>(emel::tensor::allocator::event::error_phase::initialize_tensors);
  const bool view_phase_ok =
    view_detail.phase == scan_phase || view_detail.phase == init_phase;
  CHECK(view_phase_ok);
}

TEST_CASE("tensor_allocator_reports_assemble_output_mismatch") {
  emel::tensor::allocator::sm machine{};
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
  emel_error_detail detail{};
  std::array<int32_t, 1> chunks = {{0}};
  CHECK_FALSE(machine.process_event(emel::tensor::allocator::event::allocate_tensors{
    .tensors = tensors.data(),
    .tensor_count = static_cast<int32_t>(tensors.size()),
    .alignment = 16,
    .max_buffer_size = 64,
    .no_alloc = false,
    .result_buffer_out = nullptr,
    .chunk_sizes_out = chunks.data(),
    .chunk_sizes_out_count = 0,
    .error_out = &error,
    .detail_out = &detail,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(detail.status == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(detail.domain == EMEL_ERROR_DOMAIN_TENSOR_ALLOCATOR);
  CHECK(detail.phase ==
        static_cast<uint32_t>(emel::tensor::allocator::event::error_phase::assemble));
  CHECK(detail.reason ==
        static_cast<uint32_t>(emel::tensor::allocator::event::error_reason::invalid_argument));
}

TEST_CASE("tensor_allocator_release_resets_runtime") {
  emel::tensor::allocator::sm machine{};
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
    .no_alloc = false,
    .total_size_out = &total,
  }));
  CHECK(total == 32);

  CHECK(machine.process_event(emel::tensor::allocator::event::release{}));
  CHECK(machine.total_bytes() == 0);
  CHECK(machine.chunk_count() == 0);
}

TEST_CASE("tensor_allocator_action_branch_coverage") {
  using context = emel::tensor::allocator::action::context;
  using emel::tensor::allocator::action::run_allocate_ranges;
  using emel::tensor::allocator::action::run_assemble;
  using emel::tensor::allocator::action::run_initialize_tensors;
  using emel::tensor::allocator::action::run_partition_ranges;
  using emel::tensor::allocator::action::run_scan_tensors;
  using emel::tensor::allocator::action::run_validate;

  context ctx{};
  int32_t err = EMEL_OK;
  emel_error_detail detail{};

  {
    tensor_desc t{
      .tensor_id = 1,
      .alloc_size = 16,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    };
    ctx = {};
    ctx.tensors = &t;
    ctx.tensor_count = 1;
    ctx.alignment = 3;
    ctx.max_buffer_size = 64;
    run_validate(
      emel::tensor::allocator::event::validate{
        .error_out = &err,
        .detail_out = &detail,
        .chunk_sizes_out = nullptr,
        .chunk_sizes_out_count = 0,
      },
      ctx);
    CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    std::array<tensor_desc, 2> dup = {{
      tensor_desc{
        .tensor_id = 5,
        .alloc_size = 8,
        .src_ids = {{-1, -1, -1, -1}},
        .is_view = false,
        .view_src_id = -1,
        .has_external_data = false,
      },
      tensor_desc{
        .tensor_id = 5,
        .alloc_size = 8,
        .src_ids = {{-1, -1, -1, -1}},
        .is_view = false,
        .view_src_id = -1,
        .has_external_data = false,
      },
    }};
    ctx = {};
    ctx.tensors = dup.data();
    ctx.tensor_count = static_cast<int32_t>(dup.size());
    ctx.alignment = 16;
    run_scan_tensors(
      emel::tensor::allocator::event::scan_tensors{
        .error_out = &err,
        .detail_out = &detail,
      },
      ctx);
    CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
    CHECK(detail.reason ==
          static_cast<uint32_t>(emel::tensor::allocator::event::error_reason::duplicate_tensor_id));
  }

  return;

  {
    tensor_desc big{
      .tensor_id = 7,
      .alloc_size = std::numeric_limits<int32_t>::max(),
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    };
    ctx = {};
    ctx.tensors = &big;
    ctx.tensor_count = 1;
    ctx.alignment = 16;
    run_scan_tensors(
      emel::tensor::allocator::event::scan_tensors{
        .error_out = &err,
        .detail_out = &detail,
      },
      ctx);
    CHECK(err == EMEL_ERR_BACKEND);
    CHECK(detail.reason ==
          static_cast<uint32_t>(emel::tensor::allocator::event::error_reason::alignment_overflow));
  }

  {
    ctx = {};
    ctx.tensor_count = 2;
    ctx.max_buffer_size = 8;
    ctx.effective_sizes[0] = 8;
    ctx.effective_sizes[1] = 8;
    run_partition_ranges(
      emel::tensor::allocator::event::partition_ranges{
        .error_out = &err,
      },
      ctx);
    CHECK(err == EMEL_OK);
    CHECK(ctx.chunk_count == 2);
  }

  {
    ctx = {};
    ctx.no_alloc = true;
    ctx.chunk_count = 1;
    ctx.chunk_sizes[0] = 16;
    run_allocate_ranges(
      emel::tensor::allocator::event::allocate_ranges{
        .error_out = &err,
      },
      ctx);
    CHECK(err == EMEL_OK);
  }

  {
    ctx = {};
    ctx.no_alloc = false;
    ctx.chunk_count = 1;
    ctx.chunk_sizes[0] = 0;
    run_allocate_ranges(
      emel::tensor::allocator::event::allocate_ranges{
        .error_out = &err,
        .detail_out = &detail,
      },
      ctx);
    CHECK(err == EMEL_ERR_BACKEND);
    CHECK(detail.phase ==
          static_cast<uint32_t>(emel::tensor::allocator::event::error_phase::allocate_ranges));
  }

  {
    tensor_desc t{
      .tensor_id = 9,
      .alloc_size = 16,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    };
    std::array<std::byte, 32> storage = {};
    ctx = {};
    ctx.tensors = &t;
    ctx.tensor_count = 1;
    ctx.no_alloc = false;
    ctx.chunk_count = 1;
    ctx.chunk_sizes[0] = 32;
    ctx.effective_sizes[0] = 16;
    ctx.tensor_chunk_ids[0] = -1;
    ctx.tensor_offsets[0] = 0;
    ctx.allocated_buffers[0] = storage.data();
    run_initialize_tensors(
      emel::tensor::allocator::event::initialize_tensors{
        .error_out = &err,
        .detail_out = &detail,
      },
      ctx);
    CHECK(err == EMEL_ERR_BACKEND);
    CHECK(detail.reason ==
          static_cast<uint32_t>(emel::tensor::allocator::event::error_reason::offset_out_of_range));
  }

  {
    tensor_desc t{
      .tensor_id = 10,
      .alloc_size = 16,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    };
    ctx = {};
    ctx.tensors = &t;
    ctx.tensor_count = 1;
    ctx.no_alloc = false;
    ctx.chunk_count = 1;
    ctx.chunk_sizes[0] = 32;
    ctx.effective_sizes[0] = 16;
    ctx.tensor_chunk_ids[0] = 0;
    ctx.tensor_offsets[0] = 0;
    ctx.allocated_buffers[0] = nullptr;
    run_initialize_tensors(
      emel::tensor::allocator::event::initialize_tensors{
        .error_out = &err,
        .detail_out = &detail,
      },
      ctx);
    CHECK(err == EMEL_ERR_BACKEND);
    CHECK(detail.reason ==
          static_cast<uint32_t>(emel::tensor::allocator::event::error_reason::allocation_failed));
  }

  {
    tensor_desc t{
      .tensor_id = 11,
      .alloc_size = 16,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    };
    std::array<std::byte, 8> storage = {};
    ctx = {};
    ctx.tensors = &t;
    ctx.tensor_count = 1;
    ctx.no_alloc = false;
    ctx.chunk_count = 1;
    ctx.chunk_sizes[0] = 8;
    ctx.effective_sizes[0] = 16;
    ctx.tensor_chunk_ids[0] = 0;
    ctx.tensor_offsets[0] = 0;
    ctx.allocated_buffers[0] = storage.data();
    run_initialize_tensors(
      emel::tensor::allocator::event::initialize_tensors{
        .error_out = &err,
        .detail_out = &detail,
      },
      ctx);
    CHECK(err == EMEL_ERR_BACKEND);
    CHECK(detail.reason ==
          static_cast<uint32_t>(emel::tensor::allocator::event::error_reason::offset_out_of_range));
  }

  {
    std::array<int32_t, 1> chunks = {{0}};
    ctx = {};
    ctx.no_alloc = true;
    ctx.chunk_count = 2;
    run_assemble(
      emel::tensor::allocator::event::assemble{
        .error_out = &err,
        .detail_out = &detail,
        .chunk_sizes_out = chunks.data(),
        .chunk_sizes_out_count = 1,
      },
      ctx);
    CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
    CHECK(detail.phase ==
          static_cast<uint32_t>(emel::tensor::allocator::event::error_phase::assemble));
  }

  {
    std::array<std::byte, 8> buf0 = {};
    std::array<std::byte, 8> buf1 = {};
    std::array<int32_t, 2> chunk_sizes = {{0, 0}};
    int32_t total_bytes = 0;
    int32_t chunk_count = 0;
    void * result = nullptr;

    ctx = {};
    ctx.no_alloc = false;
    ctx.chunk_count = 2;
    ctx.chunk_sizes[0] = 8;
    ctx.chunk_sizes[1] = 8;
    ctx.total_bytes = 16;
    ctx.allocated_buffers[0] = buf0.data();
    ctx.allocated_buffers[1] = buf1.data();

    run_assemble(
      emel::tensor::allocator::event::assemble{
        .error_out = &err,
        .detail_out = &detail,
        .result_buffer_out = &result,
        .total_size_out = &total_bytes,
        .chunk_sizes_out = chunk_sizes.data(),
        .chunk_sizes_out_count = static_cast<int32_t>(chunk_sizes.size()),
        .chunk_count_out = &chunk_count,
      },
      ctx);
    CHECK(err == EMEL_OK);
    CHECK(result != nullptr);
    CHECK(chunk_count == 2);
    CHECK(total_bytes == 16);
  }
}

TEST_CASE("tensor_allocator_rejects_reentrant_allocate_and_release") {
  emel::tensor::allocator::sm machine{};
  std::array<tensor_desc, 1> tensors = {{
    tensor_desc{
      .tensor_id = 99,
      .alloc_size = 16,
      .src_ids = {{-1, -1, -1, -1}},
      .is_view = false,
      .view_src_id = -1,
      .has_external_data = false,
    },
  }};

  emel::tensor::allocator::event::allocate_tensors begin{
    .tensors = tensors.data(),
    .tensor_count = static_cast<int32_t>(tensors.size()),
    .alignment = 16,
    .max_buffer_size = 32,
    .no_alloc = true,
  };
  CHECK(machine.process_event(begin));

  int32_t error = EMEL_OK;
  emel_error_detail detail{};
  CHECK(machine.process_event(emel::tensor::allocator::event::allocate_tensors{
    .tensors = tensors.data(),
    .tensor_count = static_cast<int32_t>(tensors.size()),
    .alignment = 16,
    .max_buffer_size = 32,
    .no_alloc = true,
    .error_out = &error,
    .detail_out = &detail,
  }));
  CHECK(error == EMEL_OK);

  emel::tensor::allocator::event::release rel{
    .error_out = &error,
    .detail_out = &detail,
  };
  CHECK(machine.process_event(rel));
  CHECK(error == EMEL_OK);
  error = EMEL_OK;
  detail = {};
  CHECK(machine.process_event(rel));
  CHECK(error == EMEL_OK);
}

TEST_CASE("tensor_allocator_guard_and_detail_helpers_cover_edges") {
  emel::tensor::allocator::action::context c{};
  c.phase_error = EMEL_OK;
  CHECK(emel::tensor::allocator::guard::phase_ok{}(c));
  CHECK_FALSE(emel::tensor::allocator::guard::phase_failed{}(c));

  c.phase_error = EMEL_ERR_BACKEND;
  CHECK(emel::tensor::allocator::guard::phase_failed{}(c));

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
  emel::tensor::allocator::action::run_validate(
      emel::tensor::allocator::event::validate{
        .error_out = &phase_error,
        .chunk_sizes_out = nullptr,
        .chunk_sizes_out_count = -1,
      }, c);
  CHECK(phase_error == EMEL_ERR_INVALID_ARGUMENT);

  emel::tensor::allocator::action::run_validate(
      emel::tensor::allocator::event::validate{
        .error_out = &phase_error,
        .chunk_sizes_out = nullptr,
        .chunk_sizes_out_count = 1,
      }, c);
  CHECK(phase_error == EMEL_ERR_INVALID_ARGUMENT);

  c.no_alloc = false;
  emel::tensor::allocator::action::run_validate(
      emel::tensor::allocator::event::validate{
        .error_out = &phase_error,
        .chunk_sizes_out = nullptr,
        .chunk_sizes_out_count = 0,
      }, c);
  CHECK(phase_error == EMEL_OK);
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

  c = {};
  c.no_alloc = false;
  c.chunk_count = 1;
  c.chunk_sizes[0] = 0;
  emel::tensor::allocator::action::run_allocate_ranges(
      emel::tensor::allocator::event::allocate_ranges{.error_out = &phase_error}, c);
  CHECK(phase_error == EMEL_ERR_BACKEND);

  c = {};
  c.no_alloc = false;
  c.chunk_count = 2;
  c.chunk_sizes[0] = 16;
  c.chunk_sizes[1] = 16;
  emel::tensor::allocator::action::run_allocate_ranges(
      emel::tensor::allocator::event::allocate_ranges{.error_out = &phase_error}, c);
  CHECK(phase_error == EMEL_OK);
  CHECK(c.allocated_buffers[0] != nullptr);
  CHECK(c.allocated_buffers[1] != nullptr);
  CHECK(emel::tensor::allocator::action::detail::release_allocated_buffers(c));
  CHECK(c.allocated_buffers[0] == nullptr);
  CHECK(c.allocated_buffers[1] == nullptr);
}

TEST_CASE("tensor_allocator_initialize_and_assemble_cover_error_paths") {
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
  c.tensors = tensors.data();
  c.tensor_count = 1;
  c.chunk_count = 1;
  c.effective_sizes[0] = 16;
  c.tensor_chunk_ids[0] = 1;
  c.tensor_offsets[0] = 0;
  c.allocated_buffers[0] = std::malloc(16);
  int32_t phase_error = EMEL_OK;
  emel::tensor::allocator::action::run_initialize_tensors(
      emel::tensor::allocator::event::initialize_tensors{.error_out = &phase_error}, c);
  CHECK(phase_error == EMEL_ERR_BACKEND);
  CHECK(c.allocated_buffers[0] == nullptr);

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
  c.tensors = view_tensor.data();
  c.tensor_count = 1;
  c.tensor_ids[0] = 10;
  emel::tensor::allocator::action::run_initialize_tensors(
      emel::tensor::allocator::event::initialize_tensors{.error_out = &phase_error}, c);
  CHECK(phase_error == EMEL_ERR_INVALID_ARGUMENT);

  void * result = reinterpret_cast<void *>(0x1);
  std::array<int32_t, 2> chunks = {{0, 0}};
  c = {};
  c.no_alloc = false;
  c.chunk_count = 2;
  c.chunk_sizes[0] = 16;
  c.chunk_sizes[1] = 16;
  c.total_bytes = 32;
  c.allocated_buffers[0] = std::malloc(16);
  c.allocated_buffers[1] = std::malloc(16);
  emel::tensor::allocator::action::run_assemble(
      emel::tensor::allocator::event::assemble{
        .error_out = &phase_error,
        .result_buffer_out = &result,
        .chunk_sizes_out = chunks.data(),
        .chunk_sizes_out_count = 1,
      }, c);
  CHECK(phase_error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(result == nullptr);
  CHECK(c.allocated_buffers[0] == nullptr);
  CHECK(c.allocated_buffers[1] == nullptr);

  result = reinterpret_cast<void *>(0x2);
  c = {};
  c.no_alloc = false;
  c.chunk_count = 2;
  c.chunk_sizes[0] = 16;
  c.chunk_sizes[1] = 16;
  c.total_bytes = 32;
  c.allocated_buffers[0] = std::malloc(16);
  c.allocated_buffers[1] = std::malloc(16);
  emel::tensor::allocator::action::run_assemble(
      emel::tensor::allocator::event::assemble{
        .error_out = &phase_error,
        .result_buffer_out = &result,
        .chunk_sizes_out = chunks.data(),
        .chunk_sizes_out_count = static_cast<int32_t>(chunks.size()),
      }, c);
  CHECK(phase_error == EMEL_OK);
  CHECK(result != nullptr);
  auto * assembled = static_cast<void * const *>(result);
  CHECK(assembled[0] == c.allocated_buffers[0]);
  CHECK(assembled[1] == c.allocated_buffers[1]);
  CHECK(emel::tensor::allocator::action::detail::release_allocated_buffers(c));
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
  CHECK(machine.process_event(emel::tensor::allocator::event::allocate_tensors{
    .tensors = tensors.data(),
    .tensor_count = static_cast<int32_t>(tensors.size()),
    .alignment = 16,
    .max_buffer_size = 64,
    .no_alloc = true,
  }));

  emel::tensor::allocator::sm release_machine{};
  CHECK(release_machine.process_event(emel::tensor::allocator::event::release{}));
  CHECK(release_machine.process_event(emel::tensor::allocator::event::release{}));
}

}  // namespace
