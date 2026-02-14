#include <array>
#include <cstdint>
#include <limits>

#include <doctest/doctest.h>

#include "emel/buffer/chunk_allocator/actions.hpp"
#include "emel/buffer/chunk_allocator/sm.hpp"
#include "emel/emel.h"

namespace {

TEST_CASE("buffer_chunk_allocator_starts_ready_with_defaults") {
  emel::buffer::chunk_allocator::sm machine{};
  int32_t state_count = 0;
  machine.visit_current_states([&](auto) { state_count += 1; });

  CHECK(state_count == 1);
  CHECK(machine.alignment() == 16);
  CHECK(machine.chunk_count() == 0);
}

TEST_CASE("buffer_chunk_allocator_configure_and_allocate") {
  emel::buffer::chunk_allocator::sm machine{};
  int32_t error = -1;
  CHECK(machine.process_event(emel::buffer::chunk_allocator::event::configure{
    .alignment = 16,
    .max_chunk_size = 128,
    .error_out = &error,
  }));
  CHECK(error == EMEL_OK);
  CHECK(machine.max_chunk_size() == 128);

  int32_t chunk = -1;
  uint64_t offset = 99;
  uint64_t aligned = 0;
  CHECK(machine.process_event(emel::buffer::chunk_allocator::event::allocate{
    .size = 17,
    .chunk_out = &chunk,
    .offset_out = &offset,
    .aligned_size_out = &aligned,
    .error_out = &error,
  }));

  CHECK(error == EMEL_OK);
  CHECK(chunk == 0);
  CHECK(offset == 0);
  CHECK(aligned == 32);
  CHECK(machine.chunk_count() == 1);
  CHECK(machine.chunk_max_size(0) == 32);
}

TEST_CASE("buffer_chunk_allocator_allocates_new_chunk_when_needed") {
  emel::buffer::chunk_allocator::sm machine{};
  int32_t error = -1;
  CHECK(machine.process_event(emel::buffer::chunk_allocator::event::configure{
    .alignment = 16,
    .max_chunk_size = 64,
    .error_out = &error,
  }));
  CHECK(error == EMEL_OK);

  int32_t chunk = -1;
  uint64_t offset = 0;
  CHECK(machine.process_event(emel::buffer::chunk_allocator::event::allocate{
    .size = 64,
    .chunk_out = &chunk,
    .offset_out = &offset,
    .error_out = &error,
  }));
  CHECK(error == EMEL_OK);
  CHECK(chunk == 0);
  CHECK(offset == 0);

  CHECK(machine.process_event(emel::buffer::chunk_allocator::event::allocate{
    .size = 64,
    .chunk_out = &chunk,
    .offset_out = &offset,
    .error_out = &error,
  }));
  CHECK(error == EMEL_OK);
  CHECK(chunk == 1);
  CHECK(offset == 0);
  CHECK(machine.chunk_count() == 2);
}

TEST_CASE("buffer_chunk_allocator_prefers_non_last_free_block_best_fit") {
  emel::buffer::chunk_allocator::sm machine{};
  int32_t error = -1;
  CHECK(machine.process_event(emel::buffer::chunk_allocator::event::configure{
    .alignment = 16,
    .max_chunk_size = 128,
    .error_out = &error,
  }));
  CHECK(error == EMEL_OK);

  std::array<uint64_t, 3> offsets = {0, 0, 0};
  int32_t chunk = -1;
  for (int i = 0; i < 3; ++i) {
    CHECK(machine.process_event(emel::buffer::chunk_allocator::event::allocate{
      .size = 32,
      .chunk_out = &chunk,
      .offset_out = &offsets[static_cast<size_t>(i)],
      .error_out = &error,
    }));
    CHECK(error == EMEL_OK);
    CHECK(chunk == 0);
  }

  CHECK(machine.process_event(emel::buffer::chunk_allocator::event::release{
    .chunk = 0,
    .offset = offsets[1],
    .size = 32,
    .error_out = &error,
  }));
  CHECK(error == EMEL_OK);

  uint64_t reuse_offset = 0;
  CHECK(machine.process_event(emel::buffer::chunk_allocator::event::allocate{
    .size = 24,
    .chunk_out = &chunk,
    .offset_out = &reuse_offset,
    .error_out = &error,
  }));
  CHECK(error == EMEL_OK);
  CHECK(chunk == 0);
  CHECK(reuse_offset == offsets[1]);
}

TEST_CASE("buffer_chunk_allocator_release_merges_adjacent_blocks") {
  emel::buffer::chunk_allocator::sm machine{};
  int32_t error = -1;
  CHECK(machine.process_event(emel::buffer::chunk_allocator::event::configure{
    .alignment = 16,
    .max_chunk_size = 128,
    .error_out = &error,
  }));
  CHECK(error == EMEL_OK);

  std::array<uint64_t, 3> offsets = {0, 0, 0};
  int32_t chunk = -1;
  for (int i = 0; i < 3; ++i) {
    CHECK(machine.process_event(emel::buffer::chunk_allocator::event::allocate{
      .size = 32,
      .chunk_out = &chunk,
      .offset_out = &offsets[static_cast<size_t>(i)],
      .error_out = &error,
    }));
    CHECK(error == EMEL_OK);
  }

  CHECK(machine.process_event(emel::buffer::chunk_allocator::event::release{
    .chunk = 0,
    .offset = offsets[1],
    .size = 32,
    .error_out = &error,
  }));
  CHECK(error == EMEL_OK);
  CHECK(machine.process_event(emel::buffer::chunk_allocator::event::release{
    .chunk = 0,
    .offset = offsets[0],
    .size = 32,
    .error_out = &error,
  }));
  CHECK(error == EMEL_OK);
  CHECK(machine.process_event(emel::buffer::chunk_allocator::event::release{
    .chunk = 0,
    .offset = offsets[2],
    .size = 32,
    .error_out = &error,
  }));
  CHECK(error == EMEL_OK);

  uint64_t merged_offset = 77;
  CHECK(machine.process_event(emel::buffer::chunk_allocator::event::allocate{
    .size = 96,
    .chunk_out = &chunk,
    .offset_out = &merged_offset,
    .error_out = &error,
  }));
  CHECK(error == EMEL_OK);
  CHECK(chunk == 0);
  CHECK(merged_offset == 0);
}

TEST_CASE("buffer_chunk_allocator_reports_invalid_arguments") {
  emel::buffer::chunk_allocator::sm machine{};
  int32_t error = EMEL_OK;

  int32_t chunk = -1;
  uint64_t offset = 0;
  CHECK_FALSE(machine.process_event(emel::buffer::chunk_allocator::event::allocate{
    .size = 0,
    .chunk_out = &chunk,
    .offset_out = &offset,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);

  CHECK_FALSE(machine.process_event(emel::buffer::chunk_allocator::event::allocate{
    .size = 16,
    .chunk_out = nullptr,
    .offset_out = &offset,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);

  CHECK_FALSE(machine.process_event(emel::buffer::chunk_allocator::event::release{
    .chunk = 8,
    .offset = 0,
    .size = 16,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);

  CHECK_FALSE(machine.process_event(emel::buffer::chunk_allocator::event::configure{
    .alignment = 0,
    .max_chunk_size = 64,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("buffer_chunk_allocator_reset_clears_chunks") {
  emel::buffer::chunk_allocator::sm machine{};
  int32_t error = -1;
  int32_t chunk = -1;
  uint64_t offset = 0;

  CHECK(machine.process_event(emel::buffer::chunk_allocator::event::allocate{
    .size = 32,
    .chunk_out = &chunk,
    .offset_out = &offset,
    .error_out = &error,
  }));
  CHECK(error == EMEL_OK);
  CHECK(machine.chunk_count() == 1);

  CHECK(machine.process_event(emel::buffer::chunk_allocator::event::reset{
    .error_out = &error,
  }));
  CHECK(error == EMEL_OK);
  CHECK(machine.chunk_count() == 0);

  CHECK(machine.process_event(emel::buffer::chunk_allocator::event::allocate{
    .size = 32,
    .chunk_out = &chunk,
    .offset_out = &offset,
    .error_out = &error,
  }));
  CHECK(error == EMEL_OK);
  CHECK(chunk == 0);
  CHECK(offset == 0);
}

TEST_CASE("buffer_chunk_allocator_action_helpers_cover_edge_branches") {
  namespace action = emel::buffer::chunk_allocator::action;

  CHECK(action::detail::normalize_error(EMEL_ERR_INVALID_ARGUMENT, EMEL_ERR_BACKEND) ==
        EMEL_ERR_INVALID_ARGUMENT);
  CHECK(action::detail::normalize_error(EMEL_OK, EMEL_ERR_INVALID_ARGUMENT) ==
        EMEL_ERR_INVALID_ARGUMENT);
  CHECK(action::detail::normalize_error(EMEL_OK, EMEL_OK) == EMEL_ERR_BACKEND);

  uint64_t out = 0;
  CHECK(action::detail::add_overflow(std::numeric_limits<uint64_t>::max(), 1, out));
  CHECK_FALSE(action::detail::align_up(1, 0, out));
  CHECK(action::detail::align_up(16, 16, out));
  CHECK(out == 16);
  CHECK_FALSE(action::detail::align_up(std::numeric_limits<uint64_t>::max(), 16, out));

  action::chunk_data chunk{};
  action::detail::remove_block(chunk, -1);
  CHECK(action::detail::insert_block(chunk, 0, 0));
  chunk.free_block_count = action::k_max_free_blocks;
  CHECK_FALSE(action::detail::insert_block(chunk, 8, 16));

  action::chunk_data overflow_chunk{};
  action::free_block overflow_block{
    .offset = std::numeric_limits<uint64_t>::max(),
    .size = 1,
  };
  CHECK(
      action::detail::reuse_factor(overflow_chunk, overflow_block, 1) ==
      std::numeric_limits<int64_t>::min());

  action::chunk_data slack_chunk{};
  slack_chunk.max_size = std::numeric_limits<uint64_t>::max();
  action::free_block tiny_block{
    .offset = 0,
    .size = 0,
  };
  CHECK(
      action::detail::reuse_factor(slack_chunk, tiny_block, 0) ==
      std::numeric_limits<int64_t>::max());

  action::chunk_data deficit_chunk{};
  deficit_chunk.max_size = 0;
  action::free_block huge_block{
    .offset = static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 16ULL,
    .size = 0,
  };
  CHECK(
      action::detail::reuse_factor(deficit_chunk, huge_block, 1) ==
      std::numeric_limits<int64_t>::min());
}

TEST_CASE("buffer_chunk_allocator_create_chunk_and_allocate_from_selected_error_paths") {
  namespace action = emel::buffer::chunk_allocator::action;

  action::context c{};
  int32_t idx = -1;

  c.chunk_count = action::k_max_chunks;
  CHECK_FALSE(action::detail::create_chunk(c, 16, idx));

  c.chunk_count = action::k_max_chunks - 1;
  c.max_chunk_size = 64;
  CHECK(action::detail::create_chunk(c, 16, idx));
  CHECK(idx == action::k_max_chunks - 1);
  CHECK(c.chunks[idx].free_blocks[0].size == action::k_max_chunk_limit);

  c = {};
  c.chunk_count = 1;
  c.aligned_request_size = 16;
  c.selected_chunk = -1;
  CHECK_FALSE(action::detail::allocate_from_selected(c));

  c.selected_chunk = 0;
  c.selected_block = 0;
  c.chunks[0].free_block_count = 0;
  CHECK_FALSE(action::detail::allocate_from_selected(c));

  c.chunks[0].free_block_count = 1;
  c.chunks[0].free_blocks[0] = action::free_block{
    .offset = 0,
    .size = 8,
  };
  CHECK_FALSE(action::detail::allocate_from_selected(c));

  c.chunks[0].free_blocks[0] = action::free_block{
    .offset = std::numeric_limits<uint64_t>::max(),
    .size = 32,
  };
  CHECK_FALSE(action::detail::allocate_from_selected(c));
}

TEST_CASE("buffer_chunk_allocator_release_validation_and_merge_error_paths") {
  namespace action = emel::buffer::chunk_allocator::action;

  action::context c{};
  c.chunk_count = 1;
  c.alignment = 0;
  c.request_chunk = 0;
  c.request_offset = 0;
  c.request_size = 16;
  int32_t err = EMEL_OK;
  action::run_validate_release(emel::buffer::chunk_allocator::event::validate_release{.error_out = &err}, c);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  c.alignment = 16;
  c.request_chunk = 1;
  action::run_validate_release(emel::buffer::chunk_allocator::event::validate_release{.error_out = &err}, c);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  c.request_chunk = 0;
  c.request_offset = std::numeric_limits<uint64_t>::max();
  c.request_size = 16;
  action::run_validate_release(emel::buffer::chunk_allocator::event::validate_release{.error_out = &err}, c);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  c.request_offset = 0;
  c.request_size = 16;
  c.chunks[0].max_size = 8;
  action::run_validate_release(emel::buffer::chunk_allocator::event::validate_release{.error_out = &err}, c);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  c = {};
  c.request_chunk = 2;
  c.chunk_count = 1;
  c.request_offset = 0;
  c.aligned_request_size = 16;
  CHECK_FALSE(action::detail::free_bytes(c));

  c.request_chunk = 0;
  c.request_offset = std::numeric_limits<uint64_t>::max();
  CHECK_FALSE(action::detail::free_bytes(c));

  c.request_offset = 16;
  c.aligned_request_size = 32;
  c.chunks[0].free_block_count = 2;
  c.chunks[0].free_blocks[0] = action::free_block{
    .offset = 0,
    .size = 16,
  };
  c.chunks[0].free_blocks[1] = action::free_block{
    .offset = 48,
    .size = 16,
  };
  CHECK(action::detail::free_bytes(c));
  CHECK(c.chunks[0].free_block_count == 1);
  CHECK(c.chunks[0].free_blocks[0].offset == 0);
  CHECK(c.chunks[0].free_blocks[0].size == 64);
}

TEST_CASE("buffer_chunk_allocator_action_error_handlers_cover_null_outputs") {
  namespace action = emel::buffer::chunk_allocator::action;
  action::context c{};
  int32_t err = EMEL_OK;
  c.request_error_out = &err;

  action::on_configure_error(emel::buffer::chunk_allocator::events::configure_error{.err = EMEL_OK}, c);
  CHECK(err == EMEL_ERR_BACKEND);
  action::on_allocate_error(emel::buffer::chunk_allocator::events::allocate_error{.err = EMEL_OK}, c);
  CHECK(err == EMEL_ERR_BACKEND);
  action::on_release_error(emel::buffer::chunk_allocator::events::release_error{.err = EMEL_OK}, c);
  CHECK(err == EMEL_ERR_BACKEND);
  action::on_reset_error(emel::buffer::chunk_allocator::events::reset_error{.err = EMEL_OK}, c);
  CHECK(err == EMEL_ERR_BACKEND);

  c.request_error_out = nullptr;
  const auto step_before = c.step;
  action::on_configure_error(
      emel::buffer::chunk_allocator::events::configure_error{.err = EMEL_ERR_INVALID_ARGUMENT}, c);
  action::on_allocate_error(
      emel::buffer::chunk_allocator::events::allocate_error{.err = EMEL_ERR_INVALID_ARGUMENT}, c);
  action::on_release_error(
      emel::buffer::chunk_allocator::events::release_error{.err = EMEL_ERR_INVALID_ARGUMENT}, c);
  action::on_reset_error(
      emel::buffer::chunk_allocator::events::reset_error{.err = EMEL_ERR_INVALID_ARGUMENT}, c);
  CHECK(c.step == step_before + 4);
}

TEST_CASE("buffer_chunk_allocator_sm_error_paths_update_last_error") {
  emel::buffer::chunk_allocator::sm machine{};
  int32_t error = EMEL_OK;

  CHECK_FALSE(machine.process_event(emel::buffer::chunk_allocator::event::configure{
    .alignment = 0,
    .max_chunk_size = 64,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(machine.last_error() == EMEL_ERR_INVALID_ARGUMENT);

  int32_t chunk = -1;
  uint64_t offset = 0;
  CHECK_FALSE(machine.process_event(emel::buffer::chunk_allocator::event::allocate{
    .size = 0,
    .chunk_out = &chunk,
    .offset_out = &offset,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(machine.last_error() == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("buffer_chunk_allocator_additional_action_branch_coverage") {
  namespace action = emel::buffer::chunk_allocator::action;

  action::chunk_data chunk{};
  CHECK(action::detail::insert_block(chunk, 32, 16));
  CHECK(action::detail::insert_block(chunk, 16, 8));
  CHECK(action::detail::insert_block(chunk, 64, 8));
  CHECK(chunk.free_block_count == 3);
  CHECK(chunk.free_blocks[0].offset == 16);
  CHECK(chunk.free_blocks[1].offset == 32);
  CHECK(chunk.free_blocks[2].offset == 64);

  action::context c{};
  c.chunk_count = 1;
  c.aligned_request_size = 64;
  c.chunks[0].free_block_count = 1;
  c.chunks[0].free_blocks[0] = action::free_block{
    .offset = 0,
    .size = 16,
  };
  action::detail::select_best_last(c);
  CHECK(c.selected_chunk == -1);

  c = {};
  c.request_size = 32;
  int32_t chunk_out = -1;
  uint64_t offset_out = 0;
  c.request_chunk_out = &chunk_out;
  c.request_offset_out = &offset_out;
  c.alignment = 0;
  c.max_chunk_size = 64;
  int32_t err = EMEL_OK;
  action::run_validate_allocate(
      emel::buffer::chunk_allocator::event::validate_allocate{.error_out = &err}, c);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  c.alignment = 16;
  c.request_size = std::numeric_limits<uint64_t>::max();
  action::run_validate_allocate(
      emel::buffer::chunk_allocator::event::validate_allocate{.error_out = &err}, c);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  c = {};
  c.chunk_count = action::k_max_chunks;
  c.selected_chunk = -1;
  c.aligned_request_size = 16;
  action::run_ensure_chunk(emel::buffer::chunk_allocator::event::ensure_chunk{.error_out = &err}, c);
  CHECK(err == EMEL_ERR_BACKEND);

  c = {};
  c.selected_chunk = -1;
  c.selected_block = -1;
  action::run_commit_allocate(
      emel::buffer::chunk_allocator::event::commit_allocate{.error_out = &err}, c);
  CHECK(err == EMEL_ERR_BACKEND);

  c = {};
  c.request_chunk = 3;
  c.chunk_count = 1;
  c.aligned_request_size = 16;
  c.request_offset = 0;
  action::run_merge_release(
      emel::buffer::chunk_allocator::event::merge_release{.error_out = &err}, c);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("buffer_chunk_allocator_last_chunk_continues_allocating") {
  emel::buffer::chunk_allocator::sm machine{};
  int32_t error = EMEL_OK;
  CHECK(machine.process_event(emel::buffer::chunk_allocator::event::configure{
    .alignment = 16,
    .max_chunk_size = 16,
    .error_out = &error,
  }));
  CHECK(error == EMEL_OK);

  int32_t chunk = -1;
  uint64_t offset = 0;
  for (int i = 0; i < emel::buffer::chunk_allocator::action::k_max_chunks; ++i) {
    CHECK(machine.process_event(emel::buffer::chunk_allocator::event::allocate{
      .size = 16,
      .chunk_out = &chunk,
      .offset_out = &offset,
      .error_out = &error,
    }));
    CHECK(error == EMEL_OK);
  }

  CHECK(machine.process_event(emel::buffer::chunk_allocator::event::allocate{
    .size = 16,
    .chunk_out = &chunk,
    .offset_out = &offset,
    .error_out = &error,
  }));
  CHECK(error == EMEL_OK);
  CHECK(chunk == emel::buffer::chunk_allocator::action::k_max_chunks - 1);
  CHECK(offset == 16);
}

TEST_CASE("buffer_chunk_allocator_chunk_max_size_returns_zero_for_invalid_chunk") {
  emel::buffer::chunk_allocator::sm machine{};
  CHECK(machine.chunk_max_size(-1) == 0);
  CHECK(machine.chunk_max_size(0) == 0);

  int32_t error = EMEL_OK;
  int32_t chunk = -1;
  uint64_t offset = 0;
  CHECK(machine.process_event(emel::buffer::chunk_allocator::event::allocate{
    .size = 16,
    .chunk_out = &chunk,
    .offset_out = &offset,
    .error_out = &error,
  }));
  CHECK(machine.chunk_max_size(1) == 0);
}

}  // namespace
