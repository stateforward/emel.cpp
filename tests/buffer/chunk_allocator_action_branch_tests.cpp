#include <limits>
#include <doctest/doctest.h>

#include "emel/buffer/chunk_allocator/actions.hpp"
#include "emel/buffer/chunk_allocator/events.hpp"
#include "emel/buffer/chunk_allocator/guards.hpp"
#include "emel/emel.h"

TEST_CASE("chunk_allocator_detail_helpers_cover_branches") {
  uint64_t out = 0;
  CHECK_FALSE(emel::buffer::chunk_allocator::action::detail::align_up(1, 0, out));
  CHECK(emel::buffer::chunk_allocator::action::detail::align_up(8, 4, out));
  CHECK(out == 8);

  uint64_t sum = 0;
  CHECK(emel::buffer::chunk_allocator::action::detail::add_overflow(
    std::numeric_limits<uint64_t>::max(), 1, sum));

  CHECK(
    emel::buffer::chunk_allocator::action::detail::clamp_chunk_size_limit(
      std::numeric_limits<uint64_t>::max()) ==
    emel::buffer::chunk_allocator::action::k_max_chunk_limit);

  emel::buffer::chunk_allocator::action::chunk_data chunk{};
  chunk.free_block_count = emel::buffer::chunk_allocator::action::k_max_free_blocks;
  CHECK_FALSE(emel::buffer::chunk_allocator::action::detail::insert_block(chunk, 0, 1));
  CHECK(emel::buffer::chunk_allocator::action::detail::insert_block(chunk, 0, 0));
}

TEST_CASE("chunk_allocator_configure_actions_validate_alignment") {
  emel::buffer::chunk_allocator::action::context ctx{};
  int32_t err = EMEL_OK;

  emel::buffer::chunk_allocator::event::configure request{
    .alignment = 0,
    .max_chunk_size = 1,
    .error_out = &err,
  };
  emel::buffer::chunk_allocator::action::begin_configure(request, ctx);
  CHECK(err == EMEL_OK);

  err = EMEL_OK;
  emel::buffer::chunk_allocator::event::validate_configure validate{
    .error_out = &err,
    .request = &request,
  };
  CHECK(emel::buffer::chunk_allocator::guard::invalid_configure{}(validate, ctx));
  CHECK(err == EMEL_OK);

  request.alignment = 16;
  request.max_chunk_size = 64;
  emel::buffer::chunk_allocator::action::begin_configure(request, ctx);
  err = EMEL_OK;
  emel::buffer::chunk_allocator::event::validate_configure ok_validate{
    .error_out = &err,
    .request = &request,
  };
  CHECK(emel::buffer::chunk_allocator::guard::valid_configure{}(ok_validate, ctx));
  emel::buffer::chunk_allocator::action::run_validate_configure(ok_validate, ctx);
  CHECK(err == EMEL_OK);
}

TEST_CASE("chunk_allocator_allocate_actions_handle_invalid_request") {
  emel::buffer::chunk_allocator::action::context ctx{};
  int32_t err = EMEL_OK;
  int32_t chunk = -1;
  uint64_t offset = 0;

  emel::buffer::chunk_allocator::event::allocate request{
    .size = 0,
    .chunk_out = &chunk,
    .offset_out = &offset,
    .error_out = &err,
  };
  emel::buffer::chunk_allocator::action::begin_allocate(request, ctx);
  CHECK(err == EMEL_OK);

  err = EMEL_OK;
  emel::buffer::chunk_allocator::event::validate_allocate validate{
    .error_out = &err,
    .request = &request,
  };
  CHECK(emel::buffer::chunk_allocator::guard::invalid_allocate_request{}(validate, ctx));
  CHECK(err == EMEL_OK);

  request.size = 32;
  emel::buffer::chunk_allocator::action::begin_allocate(request, ctx);
  err = EMEL_OK;
  emel::buffer::chunk_allocator::event::validate_allocate ok_validate{
    .error_out = &err,
    .request = &request,
  };
  CHECK(emel::buffer::chunk_allocator::guard::valid_allocate_request{}(ok_validate, ctx));
  emel::buffer::chunk_allocator::action::run_validate_allocate(ok_validate, ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.aligned_request_size != 0);
}

TEST_CASE("chunk_allocator_guards_reject_null_and_invalid_release_requests") {
  emel::buffer::chunk_allocator::action::context ctx{};

  emel::buffer::chunk_allocator::event::validate_allocate allocate_validate{
    .request = nullptr,
  };
  CHECK_FALSE(emel::buffer::chunk_allocator::guard::valid_allocate_request{}(
    allocate_validate, ctx));

  emel::buffer::chunk_allocator::event::validate_release release_validate{
    .request = nullptr,
  };
  CHECK_FALSE(emel::buffer::chunk_allocator::guard::valid_release_request{}(
    release_validate, ctx));

  emel::buffer::chunk_allocator::event::release request{
    .chunk = 0,
    .offset = 0,
    .size = 16,
    .alignment = 0,
  };
  release_validate.request = &request;
  ctx.chunk_count = 1;
  ctx.request_chunk = 0;
  ctx.request_offset = 0;
  ctx.request_size = 16;
  ctx.request_alignment = 0;
  ctx.chunks[0].max_size = 32;
  CHECK_FALSE(emel::buffer::chunk_allocator::guard::valid_release_request{}(
    release_validate, ctx));

  ctx.request_alignment = 16;
  ctx.request_size = std::numeric_limits<uint64_t>::max();
  CHECK_FALSE(emel::buffer::chunk_allocator::guard::valid_release_request{}(
    release_validate, ctx));
}

TEST_CASE("chunk_allocator_run_ensure_chunk_reports_backend") {
  emel::buffer::chunk_allocator::action::context ctx{};
  int32_t err = EMEL_OK;

  ctx.selected_chunk = -1;
  ctx.aligned_request_size = 64;
  ctx.chunk_count = emel::buffer::chunk_allocator::action::k_max_chunks;
  emel::buffer::chunk_allocator::action::run_ensure_chunk(
    emel::buffer::chunk_allocator::event::ensure_chunk{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("chunk_allocator_run_commit_allocate_reports_backend") {
  emel::buffer::chunk_allocator::action::context ctx{};
  int32_t err = EMEL_OK;

  ctx.selected_chunk = -1;
  emel::buffer::chunk_allocator::action::run_commit_allocate(
    emel::buffer::chunk_allocator::event::commit_allocate{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("chunk_allocator_release_actions_validate_request") {
  emel::buffer::chunk_allocator::action::context ctx{};
  int32_t err = EMEL_OK;

  emel::buffer::chunk_allocator::event::release request{
    .chunk = -1,
    .size = 0,
    .error_out = &err,
  };
  emel::buffer::chunk_allocator::action::begin_release(request, ctx);
  CHECK(err == EMEL_OK);

  err = EMEL_OK;
  emel::buffer::chunk_allocator::event::validate_release validate{
    .error_out = &err,
    .request = &request,
  };
  CHECK(emel::buffer::chunk_allocator::guard::invalid_release_request{}(validate, ctx));
  CHECK(err == EMEL_OK);

  request.chunk = 0;
  request.size = 16;
  ctx.chunk_count = 1;
  ctx.chunks[0].max_size = 32;
  emel::buffer::chunk_allocator::action::begin_release(request, ctx);
  err = EMEL_OK;
  emel::buffer::chunk_allocator::event::validate_release ok_validate{
    .error_out = &err,
    .request = &request,
  };
  CHECK(emel::buffer::chunk_allocator::guard::valid_release_request{}(ok_validate, ctx));
  emel::buffer::chunk_allocator::action::run_validate_release(ok_validate, ctx);
  CHECK(err == EMEL_OK);
}

TEST_CASE("chunk_allocator_run_merge_release_reports_backend") {
  emel::buffer::chunk_allocator::action::context ctx{};
  int32_t err = EMEL_OK;

  ctx.request_chunk = -1;
  emel::buffer::chunk_allocator::action::run_merge_release(
    emel::buffer::chunk_allocator::event::merge_release{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("chunk_allocator_action_on_unexpected_reports_invalid_argument") {
  emel::buffer::chunk_allocator::action::context ctx{};
  int32_t err = EMEL_OK;

  emel::buffer::chunk_allocator::events::allocate_error ev{
    .err = EMEL_ERR_BACKEND,
    .error_out = &err,
  };

  emel::buffer::chunk_allocator::action::on_unexpected(ev, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(ctx.step == 1);
}

TEST_CASE("chunk_allocator_detail_allocate_from_selected_paths") {
  emel::buffer::chunk_allocator::action::context ctx{};
  ctx.chunk_count = 1;
  ctx.aligned_request_size = 8;
  ctx.selected_chunk = 0;
  ctx.selected_block = 0;
  ctx.chunks[0].free_block_count = 1;
  ctx.chunks[0].free_blocks[0] = emel::buffer::chunk_allocator::action::free_block{
    .offset = 0,
    .size = 8,
  };

  CHECK(emel::buffer::chunk_allocator::action::detail::allocate_from_selected(ctx));
  CHECK(ctx.result_chunk == 0);
  CHECK(ctx.result_offset == 0);

  ctx.selected_chunk = -1;
  CHECK_FALSE(emel::buffer::chunk_allocator::action::detail::allocate_from_selected(ctx));
}

TEST_CASE("chunk_allocator_detail_create_chunk_paths") {
  emel::buffer::chunk_allocator::action::context ctx{};
  int32_t chunk = -1;

  ctx.chunk_count = emel::buffer::chunk_allocator::action::k_max_chunks;
  CHECK_FALSE(emel::buffer::chunk_allocator::action::detail::create_chunk(ctx, 64, chunk));

  ctx.chunk_count = 0;
  CHECK(emel::buffer::chunk_allocator::action::detail::create_chunk(ctx, 64, chunk));
  CHECK(chunk == 0);
  CHECK(ctx.chunk_count == 1);
}

TEST_CASE("chunk_allocator_detail_free_bytes_paths") {
  emel::buffer::chunk_allocator::action::context ctx{};
  ctx.chunk_count = 1;
  ctx.request_chunk = 0;
  ctx.request_offset = 0;
  ctx.aligned_request_size = 8;
  ctx.chunks[0].max_size = 16;

  CHECK(emel::buffer::chunk_allocator::action::detail::free_bytes(ctx));

  ctx.request_chunk = -1;
  CHECK_FALSE(emel::buffer::chunk_allocator::action::detail::free_bytes(ctx));
}

TEST_CASE("chunk_allocator_detail_allocate_from_selected_splits_block") {
  emel::buffer::chunk_allocator::action::context ctx{};
  ctx.chunk_count = 1;
  ctx.aligned_request_size = 8;
  ctx.selected_chunk = 0;
  ctx.selected_block = 0;
  ctx.chunks[0].free_block_count = 1;
  ctx.chunks[0].free_blocks[0] = emel::buffer::chunk_allocator::action::free_block{
    .offset = 0,
    .size = 16,
  };

  CHECK(emel::buffer::chunk_allocator::action::detail::allocate_from_selected(ctx));
  CHECK(ctx.result_chunk == 0);
  CHECK(ctx.result_offset == 0);
  CHECK(ctx.chunks[0].free_block_count == 1);
  CHECK(ctx.chunks[0].free_blocks[0].offset == 8);
  CHECK(ctx.chunks[0].free_blocks[0].size == 8);
}

TEST_CASE("chunk_allocator_detail_insert_block_inserts_between_blocks") {
  emel::buffer::chunk_allocator::action::chunk_data chunk{};
  chunk.free_block_count = 2;
  chunk.free_blocks[0] = emel::buffer::chunk_allocator::action::free_block{
    .offset = 0,
    .size = 8,
  };
  chunk.free_blocks[1] = emel::buffer::chunk_allocator::action::free_block{
    .offset = 16,
    .size = 8,
  };

  CHECK(emel::buffer::chunk_allocator::action::detail::insert_block(chunk, 8, 8));
  CHECK(chunk.free_block_count == 3);
  CHECK(chunk.free_blocks[0].offset == 0);
  CHECK(chunk.free_blocks[1].offset == 8);
  CHECK(chunk.free_blocks[2].offset == 16);
}
