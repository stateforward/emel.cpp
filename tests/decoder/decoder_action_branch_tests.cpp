#include <array>
#include <cstdint>

#include <doctest/doctest.h>

#include "emel/decoder/actions.hpp"
#include "emel/decoder/context.hpp"
#include "emel/decoder/events.hpp"
#include "emel/decoder/guards.hpp"
#include "emel/emel.h"
#include "emel/memory/coordinator/events.hpp"

namespace {

using decoder_context = emel::decoder::action::context;
using namespace emel::decoder;

using execute_event = emel::graph::processor::event::execute;

bool compute_validate_ok(const execute_event &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool compute_prepare_reuse(const execute_event &, bool * reused_out, int32_t * err_out) {
  if (reused_out != nullptr) {
    *reused_out = true;
  }
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool compute_alloc_ok(const execute_event &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool compute_bind_ok(const execute_event &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool compute_run_ok(const execute_event &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool compute_extract_expected(const execute_event & ev, int32_t * outputs_out, int32_t * err_out) {
  if (outputs_out != nullptr) {
    *outputs_out = ev.expected_outputs;
  }
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

struct owner_capture {
  bool saw_done = false;
  bool saw_error = false;
  int32_t outputs = 0;
  int32_t err = EMEL_OK;
};

bool capture_owner_event(void * owner_sm, const events::owner_event & ev) {
  auto * capture = static_cast<owner_capture *>(owner_sm);
  if (capture == nullptr) {
    return false;
  }
  if (ev.type == events::owner_event::kind::done) {
    capture->saw_done = true;
    capture->outputs = ev.done.outputs;
    return true;
  }
  capture->saw_error = true;
  capture->err = ev.error.err;
  return true;
}

}  // namespace

TEST_CASE("decoder_action_update_memory_reserves_once_and_sets_flag") {
  decoder_context ctx{};
  int32_t err = EMEL_OK;
  ctx.memory_coordinator->set_kind(emel::memory::coordinator::coordinator_kind::hybrid);

  action::run_update_memory(event::update_memory{
                              .error_out = &err,
                            },
                            ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.memory_reserved);

  err = EMEL_OK;
  action::run_update_memory(event::update_memory{
                              .error_out = &err,
                            },
                            ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.memory_reserved);
}

TEST_CASE("decoder_action_prepare_memory_batch_uses_lifecycle_events") {
  decoder_context ctx{};
  int32_t err = EMEL_OK;
  bool retryable = true;
  ctx.memory_coordinator->set_kind(emel::memory::coordinator::coordinator_kind::hybrid);

  std::array<int32_t, 3> seq_primary_ids = {{0, 0, 1}};
  ctx.ubatches_total = 2;
  ctx.n_tokens = 3;
  ctx.token_indices_count = 3;
  ctx.ubatch_seq_ids[0] = 0;
  ctx.ubatch_sizes[0] = 2;
  ctx.ubatch_seq_ids[1] = 1;
  ctx.ubatch_sizes[1] = 1;
  ctx.ubatch_token_indices[0] = 0;
  ctx.ubatch_token_indices[1] = 1;
  ctx.ubatch_token_indices[2] = 2;
  ctx.ubatch_token_offsets[0] = 0;
  ctx.ubatch_token_offsets[1] = 2;
  ctx.ubatch_token_offsets[2] = 3;
  ctx.seq_primary_ids = seq_primary_ids.data();
  ctx.seq_primary_ids_count = static_cast<int32_t>(seq_primary_ids.size());

  action::run_update_memory(event::update_memory{
                              .error_out = &err,
                            },
                            ctx);
  REQUIRE(err == EMEL_OK);

  action::run_prepare_memory_batch(event::prepare_memory_batch{
                                     .error_out = &err,
                                     .retryable_out = &retryable,
                                   },
                                   ctx);
  CHECK(err == EMEL_OK);
  CHECK_FALSE(retryable);

  const auto view = ctx.memory_coordinator->view();
  CHECK(view.is_sequence_active(0));
  CHECK(view.is_sequence_active(1));
  CHECK(view.sequence_length(0) == 2);
  CHECK(view.sequence_length(1) == 1);
}

TEST_CASE("decoder_action_prepare_memory_batch_propagates_lifecycle_errors") {
  decoder_context ctx{};
  int32_t err = EMEL_OK;
  bool retryable = true;
  ctx.memory_coordinator->set_kind(emel::memory::coordinator::coordinator_kind::hybrid);

  std::array<int32_t, 2> seq_primary_ids = {{0, 0}};

  REQUIRE(ctx.memory_coordinator->process_event(emel::memory::coordinator::event::reserve{
    .max_sequences = 4,
    .max_blocks = 1,
    .block_tokens = 1,
    .error_out = &err,
  }));
  ctx.memory_reserved = true;
  ctx.ubatches_total = 1;
  ctx.n_tokens = 2;
  ctx.token_indices_count = 2;
  ctx.ubatch_seq_ids[0] = 0;
  ctx.ubatch_sizes[0] = 2;
  ctx.ubatch_token_indices[0] = 0;
  ctx.ubatch_token_indices[1] = 1;
  ctx.ubatch_token_offsets[0] = 0;
  ctx.ubatch_token_offsets[1] = 2;
  ctx.seq_primary_ids = seq_primary_ids.data();
  ctx.seq_primary_ids_count = static_cast<int32_t>(seq_primary_ids.size());

  action::run_prepare_memory_batch(event::prepare_memory_batch{
                                     .error_out = &err,
                                     .retryable_out = &retryable,
                                   },
                                   ctx);

  CHECK(err == EMEL_ERR_BACKEND);
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);
  CHECK_FALSE(retryable);
}

TEST_CASE("decoder_action_rollback_ubatch_uses_lifecycle_rollback_slots") {
  decoder_context ctx{};
  int32_t err = EMEL_OK;
  bool retryable = false;
  ctx.memory_coordinator->set_kind(emel::memory::coordinator::coordinator_kind::hybrid);

  std::array<int32_t, 3> seq_primary_ids = {{0, 0, 0}};

  ctx.ubatches_total = 1;
  ctx.ubatches_processed = 0;
  ctx.n_tokens = 3;
  ctx.token_indices_count = 3;
  ctx.ubatch_seq_ids[0] = 0;
  ctx.ubatch_sizes[0] = 3;
  ctx.ubatch_token_indices[0] = 0;
  ctx.ubatch_token_indices[1] = 1;
  ctx.ubatch_token_indices[2] = 2;
  ctx.ubatch_token_offsets[0] = 0;
  ctx.ubatch_token_offsets[1] = 3;
  ctx.seq_primary_ids = seq_primary_ids.data();
  ctx.seq_primary_ids_count = static_cast<int32_t>(seq_primary_ids.size());

  action::run_update_memory(event::update_memory{
                              .error_out = &err,
                            },
                            ctx);
  REQUIRE(err == EMEL_OK);

  action::run_prepare_memory_batch(event::prepare_memory_batch{
                                     .error_out = &err,
                                     .retryable_out = &retryable,
                                   },
                                   ctx);
  REQUIRE(err == EMEL_OK);
  REQUIRE(ctx.memory_coordinator->view().sequence_length(0) == 3);

  action::run_rollback_ubatch(event::rollback_ubatch{
                                .error_out = &err,
                                .rollback_needed = true,
                              },
                              ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.memory_coordinator->view().sequence_length(0) == 0);
}

TEST_CASE("decoder_action_prepare_memory_batch_allocates_mixed_ubatch_per_sequence") {
  decoder_context ctx{};
  int32_t err = EMEL_OK;
  bool retryable = false;
  ctx.memory_coordinator->set_kind(emel::memory::coordinator::coordinator_kind::hybrid);

  std::array<int32_t, 2> seq_primary_ids = {{0, 1}};
  ctx.n_tokens = 2;
  ctx.ubatches_total = 1;
  ctx.token_indices_count = 2;
  ctx.ubatch_sizes[0] = 2;
  ctx.ubatch_seq_ids[0] = 0;
  ctx.ubatch_token_indices[0] = 0;
  ctx.ubatch_token_indices[1] = 1;
  ctx.ubatch_token_offsets[0] = 0;
  ctx.ubatch_token_offsets[1] = 2;
  ctx.seq_primary_ids = seq_primary_ids.data();
  ctx.seq_primary_ids_count = static_cast<int32_t>(seq_primary_ids.size());

  action::run_update_memory(event::update_memory{
                              .error_out = &err,
                            },
                            ctx);
  REQUIRE(err == EMEL_OK);

  action::run_prepare_memory_batch(event::prepare_memory_batch{
                                     .error_out = &err,
                                     .retryable_out = &retryable,
                                   },
                                   ctx);
  REQUIRE(err == EMEL_OK);
  CHECK_FALSE(retryable);

  const auto view = ctx.memory_coordinator->view();
  CHECK(view.is_sequence_active(0));
  CHECK(view.is_sequence_active(1));
  CHECK(view.sequence_length(0) == 1);
  CHECK(view.sequence_length(1) == 1);
}

TEST_CASE("decoder_action_rollback_ubatch_reverts_failed_and_remaining_preallocations") {
  decoder_context ctx{};
  int32_t err = EMEL_OK;
  bool retryable = false;
  ctx.memory_coordinator->set_kind(emel::memory::coordinator::coordinator_kind::hybrid);

  std::array<int32_t, 4> seq_primary_ids = {{0, 0, 1, 1}};
  ctx.n_tokens = 4;
  ctx.ubatches_total = 3;
  ctx.token_indices_count = 4;
  ctx.ubatch_sizes[0] = 1;
  ctx.ubatch_sizes[1] = 2;
  ctx.ubatch_sizes[2] = 1;
  ctx.ubatch_seq_ids[0] = 0;
  ctx.ubatch_seq_ids[1] = 0;
  ctx.ubatch_seq_ids[2] = 1;
  ctx.ubatch_token_indices[0] = 0;
  ctx.ubatch_token_indices[1] = 1;
  ctx.ubatch_token_indices[2] = 2;
  ctx.ubatch_token_indices[3] = 3;
  ctx.ubatch_token_offsets[0] = 0;
  ctx.ubatch_token_offsets[1] = 1;
  ctx.ubatch_token_offsets[2] = 3;
  ctx.ubatch_token_offsets[3] = 4;
  ctx.seq_primary_ids = seq_primary_ids.data();
  ctx.seq_primary_ids_count = static_cast<int32_t>(seq_primary_ids.size());

  action::run_update_memory(event::update_memory{
                              .error_out = &err,
                            },
                            ctx);
  REQUIRE(err == EMEL_OK);
  action::run_prepare_memory_batch(event::prepare_memory_batch{
                                     .error_out = &err,
                                     .retryable_out = &retryable,
                                   },
                                   ctx);
  REQUIRE(err == EMEL_OK);

  const auto before = ctx.memory_coordinator->view();
  REQUIRE(before.sequence_length(0) == 2);
  REQUIRE(before.sequence_length(1) == 2);

  // ubatch[0] was processed successfully; ubatch[1] failed and ubatch[2] was never run.
  ctx.ubatches_processed = 1;
  action::run_rollback_ubatch(event::rollback_ubatch{
                                .error_out = &err,
                                .rollback_needed = true,
                              },
                              ctx);
  REQUIRE(err == EMEL_OK);

  const auto after = ctx.memory_coordinator->view();
  CHECK(after.sequence_length(0) == 1);
  CHECK(after.sequence_length(1) == 0);
}

TEST_CASE("decoder_action_process_ubatch_projects_payloads_and_updates_progress") {
  decoder_context ctx{};
  int32_t err = EMEL_OK;
  bool retryable = false;
  bool rollback_needed = false;
  ctx.memory_coordinator->set_kind(emel::memory::coordinator::coordinator_kind::hybrid);

  std::array<int32_t, 1> positions = {{11}};
  std::array<int32_t, 1> positions_y = {{22}};
  std::array<int32_t, 1> positions_z = {{33}};
  std::array<int32_t, 3> packed_positions = {
      positions[0], positions_y[0], positions_z[0]};
  std::array<uint64_t, 1> seq_masks = {{1u}};
  std::array<int32_t, 1> seq_primary_ids = {{0}};

  ctx.n_tokens = 1;
  ctx.seq_masks = seq_masks.data();
  ctx.seq_masks_count = 1;
  ctx.seq_mask_words = 1;
  ctx.seq_primary_ids = seq_primary_ids.data();
  ctx.seq_primary_ids_count = 1;
  ctx.positions = packed_positions.data();
  ctx.positions_count = 3;
  ctx.ubatches_total = 1;
  ctx.ubatches_processed = 0;
  ctx.ubatch_sizes[0] = 1;
  ctx.ubatch_outputs[0] = 1;
  ctx.ubatch_seq_ids[0] = 0;
  ctx.ubatch_token_indices[0] = 0;
  ctx.ubatch_token_offsets[0] = 0;
  ctx.ubatch_token_offsets[1] = 1;
  ctx.token_indices_count = 1;

  ctx.compute_validate = &compute_validate_ok;
  ctx.compute_prepare_graph = &compute_prepare_reuse;
  ctx.compute_alloc_graph = &compute_alloc_ok;
  ctx.compute_bind_inputs = &compute_bind_ok;
  ctx.compute_run_backend = &compute_run_ok;
  ctx.compute_extract_outputs = &compute_extract_expected;

  action::run_update_memory(event::update_memory{
                              .error_out = &err,
                            },
                            ctx);
  REQUIRE(err == EMEL_OK);

  action::run_prepare_memory_batch(event::prepare_memory_batch{
                                     .error_out = &err,
                                     .retryable_out = &retryable,
                                   },
                                   ctx);
  REQUIRE(err == EMEL_OK);

  action::run_process_ubatch(event::process_ubatch{
                               .error_out = &err,
                               .rollback_needed_out = &rollback_needed,
                             },
                             ctx);

  CHECK(err == EMEL_OK);
  CHECK_FALSE(rollback_needed);
  CHECK(ctx.outputs_processed == 1);
  CHECK(ctx.ubatches_processed == 1);
}

TEST_CASE("decoder_action_process_and_rollback_error_paths") {
  int32_t err = EMEL_OK;
  bool rollback_needed = false;

  {
    decoder_context ctx{};
    ctx.ubatch_executor.reset();
    action::run_process_ubatch(event::process_ubatch{
                                 .error_out = &err,
                                 .rollback_needed_out = &rollback_needed,
                               },
                               ctx);
    CHECK(err == EMEL_ERR_BACKEND);
    CHECK(rollback_needed);
  }

  {
    decoder_context ctx{};
    ctx.memory_coordinator.reset();
    rollback_needed = false;
    action::run_process_ubatch(event::process_ubatch{
                                 .error_out = &err,
                                 .rollback_needed_out = &rollback_needed,
                               },
                               ctx);
    CHECK(err == EMEL_ERR_BACKEND);
    CHECK_FALSE(rollback_needed);
  }

  {
    decoder_context ctx{};
    ctx.ubatches_total = 1;
    ctx.ubatches_processed = 1;
    rollback_needed = false;
    action::run_process_ubatch(event::process_ubatch{
                                 .error_out = &err,
                                 .rollback_needed_out = &rollback_needed,
                               },
                               ctx);
    CHECK(err == EMEL_ERR_BACKEND);
  }

  {
    decoder_context ctx{};
    std::array<int32_t, 1> seq_primary_ids = {{0}};
    ctx.memory_coordinator->set_kind(emel::memory::coordinator::coordinator_kind::hybrid);
    ctx.n_tokens = 1;
    ctx.ubatches_total = 1;
    ctx.ubatches_processed = 1;
    ctx.token_indices_count = 1;
    ctx.ubatch_seq_ids[0] = 0;
    ctx.ubatch_sizes[0] = 1;
    ctx.ubatch_token_indices[0] = 0;
    ctx.ubatch_token_offsets[0] = 0;
    ctx.ubatch_token_offsets[1] = 1;
    ctx.seq_primary_ids = seq_primary_ids.data();
    ctx.seq_primary_ids_count = static_cast<int32_t>(seq_primary_ids.size());

    bool retryable = false;
    action::run_update_memory(event::update_memory{
                                .error_out = &err,
                              },
                              ctx);
    REQUIRE(err == EMEL_OK);
    action::run_prepare_memory_batch(event::prepare_memory_batch{
                                       .error_out = &err,
                                       .retryable_out = &retryable,
                                     },
                                     ctx);
    REQUIRE(err == EMEL_OK);

    ctx.outputs_processed = 2;
    ctx.outputs_total = 1;

    action::run_rollback_ubatch(event::rollback_ubatch{
                                  .error_out = &err,
                                  .rollback_needed = true,
                                },
                                ctx);
    CHECK(err == EMEL_ERR_BACKEND);
  }
}

TEST_CASE("decoder_action_helper_branches_and_dispatch_paths") {
  decoder_context ctx{};
  int32_t err = EMEL_OK;

  ctx.outputs_capacity = 1;
  ctx.outputs_total = 2;
  action::run_reserve_output(event::reserve_output{
                               .error_out = &err,
                             },
                             ctx);
  CHECK(err == EMEL_ERR_BACKEND);

  action::reject_invalid_reserve_output(event::reserve_output{
                                          .error_out = &err,
                                        },
                                        ctx);
  CHECK(err == EMEL_ERR_BACKEND);

  ctx.outputs_processed = 0;
  ctx.outputs_total = 1;
  action::run_finalize_outputs(event::finalize_outputs{
                                 .error_out = &err,
                               },
                               ctx);
  CHECK(err == EMEL_ERR_BACKEND);

  action::on_invalid_ubatch_size(event::process_ubatch{
                                   .error_out = &err,
                                 },
                                 ctx);
  CHECK(err == EMEL_ERR_BACKEND);

  action::run_optimize_memory(event::decode{}, ctx);
  action::run_update_memory(event::decode{}, ctx);
  action::run_prepare_memory_batch(event::decode{}, ctx);
  action::run_rollback_ubatch(event::decode{}, ctx);
  action::run_finalize_outputs(event::decode{}, ctx);

  owner_capture capture{};
  action::dispatch_decoding_done_to_owner(events::decoding_done{
                                            .outputs = 7,
                                            .error_out = &err,
                                            .owner_sm = &capture,
                                            .dispatch_event = &capture_owner_event,
                                          },
                                          ctx);
  CHECK(capture.saw_done);
  CHECK(capture.outputs == 7);

  action::dispatch_decoding_error_to_owner(events::decoding_error{
                                             .err = EMEL_ERR_BACKEND,
                                             .error_out = &err,
                                             .owner_sm = &capture,
                                             .dispatch_event = &capture_owner_event,
                                           },
                                           ctx);
  CHECK(capture.saw_error);
  CHECK(capture.err == EMEL_ERR_BACKEND);

  ctx.phase_error = EMEL_ERR_BACKEND;
  ctx.last_error = EMEL_OK;
  action::capture_rollback_error(ctx);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);
  action::capture_ubatch_error(ctx);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);
  action::ensure_last_error(ctx);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);
  action::mark_done(ctx);
  CHECK(ctx.last_error == EMEL_OK);

  action::on_unexpected(event::decode{
                          .error_out = &err,
                        },
                        ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("decoder_action_detail_helpers_and_guard_validation_edges") {
  CHECK(action::detail::normalize_error(true, EMEL_OK) == EMEL_OK);
  CHECK(action::detail::normalize_error(false, EMEL_OK) == EMEL_ERR_BACKEND);
  CHECK(action::detail::normalize_error(false, EMEL_ERR_INVALID_ARGUMENT) ==
        EMEL_ERR_INVALID_ARGUMENT);

  CHECK(action::detail::normalize_ubatch_error(true, EMEL_OK) == EMEL_OK);
  CHECK(action::detail::normalize_ubatch_error(false, EMEL_OK) == EMEL_ERR_BACKEND);
  CHECK(action::detail::normalize_ubatch_error(false, EMEL_ERR_INVALID_ARGUMENT) ==
        EMEL_ERR_BACKEND);

  CHECK(action::detail::primary_seq_from_mask(nullptr, 1, 0) == 0);
  std::array<uint64_t, 2> masks = {{0, 0}};
  CHECK(action::detail::primary_seq_from_mask(masks.data(), 0, 0) == 0);
  CHECK(action::detail::primary_seq_from_mask(masks.data(), 1, 0) == 0);
  masks[1] = (1ULL << 5);
  CHECK(action::detail::primary_seq_from_mask(masks.data(), 1, 1) == 5);

  std::array<int32_t, 2> token_ids = {{7, 8}};
  std::array<int8_t, 2> output_mask = {{1, 0}};
  std::array<uint64_t, 2> seq_masks = {{1, 1}};
  std::array<int32_t, 2> seq_primary_ids = {{0, 0}};
  std::array<int32_t, 3> positions_short = {{0, 1, 2}};
  std::array<int32_t, 2> positions_too_small = {{0, 1}};

  decoder_context base{};
  base.token_ids = token_ids.data();
  base.n_tokens = 2;
  base.n_ubatch = 1;
  base.outputs_capacity = 2;
  CHECK(guard::valid_token_inputs(base));

  decoder_context invalid_n_ubatch{};
  invalid_n_ubatch.token_ids = token_ids.data();
  invalid_n_ubatch.n_tokens = 2;
  invalid_n_ubatch.n_ubatch = 1;
  invalid_n_ubatch.outputs_capacity = 2;
  invalid_n_ubatch.n_ubatch = -1;
  CHECK_FALSE(guard::valid_token_inputs(invalid_n_ubatch));

  decoder_context invalid_outputs_capacity{};
  invalid_outputs_capacity.token_ids = token_ids.data();
  invalid_outputs_capacity.n_tokens = 2;
  invalid_outputs_capacity.n_ubatch = 1;
  invalid_outputs_capacity.outputs_capacity = 2;
  invalid_outputs_capacity.outputs_capacity = -1;
  CHECK_FALSE(guard::valid_token_inputs(invalid_outputs_capacity));

  decoder_context invalid_output_mask{};
  invalid_output_mask.token_ids = token_ids.data();
  invalid_output_mask.n_tokens = 2;
  invalid_output_mask.n_ubatch = 1;
  invalid_output_mask.outputs_capacity = 2;
  invalid_output_mask.output_mask = output_mask.data();
  invalid_output_mask.output_mask_count = 1;
  CHECK_FALSE(guard::valid_token_inputs(invalid_output_mask));

  decoder_context invalid_mask_words{};
  invalid_mask_words.token_ids = token_ids.data();
  invalid_mask_words.n_tokens = 2;
  invalid_mask_words.n_ubatch = 1;
  invalid_mask_words.outputs_capacity = 2;
  invalid_mask_words.seq_masks = seq_masks.data();
  invalid_mask_words.seq_mask_words = 0;
  invalid_mask_words.seq_masks_count = 2;
  CHECK_FALSE(guard::valid_token_inputs(invalid_mask_words));

  decoder_context invalid_mask_count{};
  invalid_mask_count.token_ids = token_ids.data();
  invalid_mask_count.n_tokens = 2;
  invalid_mask_count.n_ubatch = 1;
  invalid_mask_count.outputs_capacity = 2;
  invalid_mask_count.seq_masks = seq_masks.data();
  invalid_mask_count.seq_mask_words = 1;
  invalid_mask_count.seq_masks_count = 1;
  CHECK_FALSE(guard::valid_token_inputs(invalid_mask_count));

  decoder_context invalid_primary_count{};
  invalid_primary_count.token_ids = token_ids.data();
  invalid_primary_count.n_tokens = 2;
  invalid_primary_count.n_ubatch = 1;
  invalid_primary_count.outputs_capacity = 2;
  invalid_primary_count.seq_primary_ids = seq_primary_ids.data();
  invalid_primary_count.seq_primary_ids_count = 1;
  CHECK_FALSE(guard::valid_token_inputs(invalid_primary_count));

  decoder_context invalid_positions_count{};
  invalid_positions_count.token_ids = token_ids.data();
  invalid_positions_count.n_tokens = 2;
  invalid_positions_count.n_ubatch = 1;
  invalid_positions_count.outputs_capacity = 2;
  invalid_positions_count.positions = positions_too_small.data();
  invalid_positions_count.positions_count = 1;
  CHECK_FALSE(guard::valid_token_inputs(invalid_positions_count));

  decoder_context invalid_positions_stride{};
  invalid_positions_stride.token_ids = token_ids.data();
  invalid_positions_stride.n_tokens = 2;
  invalid_positions_stride.n_ubatch = 1;
  invalid_positions_stride.outputs_capacity = 2;
  invalid_positions_stride.positions = positions_short.data();
  invalid_positions_stride.positions_count = 3;
  CHECK_FALSE(guard::valid_token_inputs(invalid_positions_stride));
}

TEST_CASE("decoder_action_null_pointer_and_template_overload_paths") {
  decoder_context ctx{};
  bool rollback_needed = true;

  action::run_validate(event::validate{}, ctx);
  action::run_batch_tokens(event::batch_tokens{}, ctx);
  action::run_initialize_batch(event::initialize_batch{}, ctx);
  action::run_update_memory(event::update_memory{}, ctx);
  action::run_optimize_memory(event::optimize_memory{}, ctx);
  action::run_reserve_output(event::reserve_output{}, ctx);
  action::reject_invalid_reserve_output(event::reserve_output{}, ctx);
  action::run_process_ubatch(event::process_ubatch{}, ctx);
  action::run_rollback_ubatch(event::rollback_ubatch{
                                .rollback_needed = true,
                              },
                              ctx);
  action::run_finalize_outputs(event::finalize_outputs{}, ctx);

  int32_t err = EMEL_OK;
  action::on_invalid_ubatch_size(event::process_ubatch{
                                   .error_out = &err,
                                   .rollback_needed_out = &rollback_needed,
                                 },
                                 ctx);
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK_FALSE(rollback_needed);

  action::reject_invalid_reserve_output(event::decode{}, ctx);
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);

  action::on_invalid_ubatch_size(event::decode{}, ctx);
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);
  CHECK(ctx.ubatch_error == EMEL_ERR_BACKEND);
  CHECK_FALSE(ctx.rollback_needed);
}

TEST_CASE("decoder_action_batch_initialize_and_memory_failure_edges") {
  int32_t err = EMEL_OK;

  {
    decoder_context ctx{};
    ctx.token_batcher.reset();
    action::run_batch_tokens(event::batch_tokens{
                                 .error_out = &err,
                               },
                               ctx);
    CHECK(err == EMEL_ERR_BACKEND);
  }

  {
    decoder_context ctx{};
    ctx.n_tokens = 1;
    ctx.token_ids = nullptr;
    action::run_batch_tokens(event::batch_tokens{
                                 .error_out = &err,
                               },
                               ctx);
    const bool expected_error = err == EMEL_ERR_INVALID_ARGUMENT || err == EMEL_ERR_BACKEND;
    CHECK(expected_error);
  }

  {
    decoder_context ctx{};
    ctx.token_ids = nullptr;
    ctx.n_tokens = 1;
    ctx.n_ubatch = 1;
    action::run_initialize_batch(event::initialize_batch{
                                   .error_out = &err,
                                 },
                                 ctx);
    CHECK(err == EMEL_ERR_BACKEND);
  }

  {
    decoder_context ctx{};
    std::array<int32_t, 1> token_ids = {{11}};
    std::array<uint64_t, 1> seq_masks = {{1}};

    ctx.token_ids = token_ids.data();
    ctx.n_tokens = 1;
    ctx.n_ubatch = 0;
    ctx.output_all = true;
    ctx.seq_masks = seq_masks.data();
    ctx.seq_masks_count = 1;
    ctx.seq_mask_words = 1;
    ctx.seq_primary_ids = nullptr;
    ctx.seq_primary_ids_count = 0;

    action::run_initialize_batch(event::initialize_batch{
                                   .error_out = &err,
                                 },
                                 ctx);
    CHECK(err == EMEL_OK);
    CHECK(ctx.ubatch_seq_ids[0] == 0);
    CHECK(ctx.n_ubatch == ctx.n_tokens);
  }

  {
    decoder_context ctx{};
    std::array<int32_t, 2> token_ids = {{1, 2}};
    ctx.token_ids = token_ids.data();
    ctx.n_tokens = 2;
    ctx.n_ubatch = 1;
    ctx.output_all = false;
    ctx.output_mask = nullptr;
    ctx.output_mask_count = 0;
    ctx.seq_masks = nullptr;
    ctx.seq_primary_ids = nullptr;
    action::run_initialize_batch(event::initialize_batch{
                                   .error_out = &err,
                                 },
                                 ctx);
    CHECK(err == EMEL_OK);
    CHECK(ctx.outputs_total == 1);
  }

  {
    decoder_context ctx{};
    ctx.memory_coordinator.reset();
    action::run_update_memory(event::update_memory{
                                .error_out = &err,
                              },
                              ctx);
    CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    decoder_context ctx{};
    int32_t coordinator_err = EMEL_OK;
    CHECK_FALSE(ctx.memory_coordinator->process_event(emel::memory::coordinator::event::allocate_slots{
      .seq_id = 0,
      .token_count = 1,
      .error_out = &coordinator_err,
    }));
    action::run_update_memory(event::update_memory{
                                .error_out = &err,
                              },
                              ctx);
    const bool reserve_result_is_stable = err == EMEL_OK || err == EMEL_ERR_BACKEND;
    CHECK(reserve_result_is_stable);
  }

  {
    decoder_context ctx{};
    bool retryable = true;
    ctx.ubatches_total = 1;
    ctx.ubatch_seq_ids[0] = -1;
    ctx.ubatch_sizes[0] = 1;
    action::run_prepare_memory_batch(event::prepare_memory_batch{
                                       .error_out = &err,
                                       .retryable_out = &retryable,
                                     },
                                     ctx);
    const bool expected_error = err == EMEL_ERR_INVALID_ARGUMENT || err == EMEL_ERR_BACKEND;
    CHECK(expected_error);
    CHECK_FALSE(retryable);
  }

  {
    decoder_context ctx{};
    ctx.memory_coordinator.reset();
    bool retryable = true;
    action::run_prepare_memory_batch(event::prepare_memory_batch{
                                       .error_out = &err,
                                       .retryable_out = &retryable,
                                     },
                                     ctx);
    CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    decoder_context ctx{};
    action::run_prepare_memory_batch(event::prepare_memory_batch{}, ctx);
    CHECK(ctx.phase_error == EMEL_OK);
  }

  {
    decoder_context ctx{};
    ctx.memory_coordinator->set_kind(emel::memory::coordinator::coordinator_kind::hybrid);
    action::run_update_memory(event::update_memory{
                                .error_out = &err,
                              },
                              ctx);
    REQUIRE(err == EMEL_OK);
    ctx.ubatches_total = 1;
    ctx.ubatches_processed = 5;
    action::run_rollback_ubatch(event::rollback_ubatch{
                                  .error_out = &err,
                                  .rollback_needed = true,
                                },
                                ctx);
    CHECK(err == EMEL_ERR_BACKEND);
  }

  {
    decoder_context ctx{};
    ctx.memory_coordinator->set_kind(emel::memory::coordinator::coordinator_kind::hybrid);
    action::run_update_memory(event::update_memory{
                                .error_out = &err,
                              },
                              ctx);
    REQUIRE(err == EMEL_OK);
    ctx.ubatches_total = 1;
    ctx.ubatches_processed = 1;
    ctx.ubatch_seq_ids[0] = -1;
    ctx.ubatch_sizes[0] = 1;
    action::run_rollback_ubatch(event::rollback_ubatch{
                                  .error_out = &err,
                                  .rollback_needed = true,
                                },
                                ctx);
    const bool expected_error = err == EMEL_ERR_INVALID_ARGUMENT || err == EMEL_ERR_BACKEND;
    CHECK(expected_error);
  }

  {
    decoder_context ctx{};
    ctx.memory_coordinator.reset();
    ctx.ubatches_total = 1;
    ctx.ubatches_processed = 1;
    action::run_rollback_ubatch(event::rollback_ubatch{
                                  .error_out = &err,
                                  .rollback_needed = true,
                                },
                                ctx);
    CHECK(err == EMEL_ERR_BACKEND);
  }

  {
    decoder_context ctx{};
    action::ensure_last_error(ctx);
    CHECK(ctx.last_error == EMEL_ERR_BACKEND);
  }
}
