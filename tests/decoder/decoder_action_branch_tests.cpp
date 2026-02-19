#include <array>
#include <doctest/doctest.h>

#include "emel/decoder/actions.hpp"
#include "emel/decoder/events.hpp"
#include "emel/decoder/guards.hpp"
#include "emel/emel.h"
#include "emel/kv/cache/actions.hpp"

namespace {

struct owner_probe {
  int32_t done_calls = 0;
  int32_t error_calls = 0;
};

bool on_decode_event(void * owner_sm, const emel::decoder::events::owner_event & ev) {
  auto * owner = static_cast<owner_probe *>(owner_sm);
  if (owner == nullptr) {
    return false;
  }
  if (ev.type == emel::decoder::events::owner_event::kind::done) {
    owner->done_calls += 1;
  } else {
    owner->error_calls += 1;
  }
  return true;
}

using compute_execute_t = emel::decoder::compute_executor::event::execute;

bool compute_validate(const compute_execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool compute_prepare_graph(const compute_execute_t &, bool * reused_out, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  if (reused_out != nullptr) {
    *reused_out = true;
  }
  return true;
}

bool compute_alloc_graph(const compute_execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool compute_bind_inputs(const compute_execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool compute_run_backend(const compute_execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool compute_extract_outputs(
    const compute_execute_t & request, int32_t * outputs_out, int32_t * err_out) {
  if (outputs_out != nullptr) {
    *outputs_out = request.ubatch_size;
  }
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

}  // namespace

TEST_CASE("decoder_actions_return_early_when_error_out_is_null") {
  emel::decoder::action::context ctx{};

  emel::decoder::action::run_validate(emel::decoder::event::validate{}, ctx);
  emel::decoder::action::reject_invalid_validate(emel::decoder::event::validate{}, ctx);
  emel::decoder::action::run_initialize_batch(emel::decoder::event::initialize_batch{}, ctx);
  emel::decoder::action::run_update_memory(emel::decoder::event::update_memory{}, ctx);
  emel::decoder::action::run_prepare_memory_batch(emel::decoder::event::prepare_memory_batch{}, ctx);
  emel::decoder::action::run_optimize_memory(emel::decoder::event::optimize_memory{}, ctx);
  emel::decoder::action::run_reserve_output(emel::decoder::event::reserve_output{}, ctx);
  emel::decoder::action::reject_invalid_reserve_output(emel::decoder::event::reserve_output{}, ctx);
  emel::decoder::action::run_process_ubatch(emel::decoder::event::process_ubatch{}, ctx);
  emel::decoder::action::run_rollback_ubatch(emel::decoder::event::rollback_ubatch{}, ctx);
  emel::decoder::action::run_finalize_outputs(emel::decoder::event::finalize_outputs{}, ctx);
}

TEST_CASE("decoder_action_detail_normalizes_errors") {
  using emel::decoder::action::detail::normalize_error;
  using emel::decoder::action::detail::normalize_ubatch_error;

  CHECK(normalize_error(false, EMEL_OK) == EMEL_ERR_BACKEND);
  CHECK(normalize_ubatch_error(true, EMEL_OK) == EMEL_OK);
  CHECK(normalize_ubatch_error(false, EMEL_ERR_BACKEND) == EMEL_ERR_BACKEND);
}

TEST_CASE("decoder_action_detail_selects_primary_seq_from_masks") {
  using emel::decoder::action::detail::primary_seq_from_mask;

  CHECK(primary_seq_from_mask(nullptr, 1, 0) == 0);
  CHECK(primary_seq_from_mask(nullptr, 0, 0) == 0);

  std::array<uint64_t, 4> masks = {{0U, 0U, 0U, (uint64_t{1} << 5)}};
  CHECK(primary_seq_from_mask(masks.data(), 2, 0) == 0);
  CHECK(primary_seq_from_mask(masks.data(), 2, 1) == 69);
}

TEST_CASE("decoder_run_validate_checks_token_inputs") {
  emel::decoder::action::context ctx{};
  int32_t err = EMEL_OK;

  ctx.token_ids = nullptr;
  ctx.n_tokens = 0;
  CHECK(emel::decoder::guard::invalid_token_inputs(ctx));
  emel::decoder::action::reject_invalid_validate(
    emel::decoder::event::validate{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  std::array<int32_t, 1> tokens = {{1}};
  std::array<int8_t, 1> output_mask = {{1}};
  std::array<int32_t, 1> seq_ids = {{0}};
  std::array<int32_t, 1> positions = {{0}};

  ctx.token_ids = tokens.data();
  ctx.n_tokens = 1;
  CHECK(emel::decoder::guard::valid_token_inputs(ctx));
  err = EMEL_OK;
  emel::decoder::action::run_validate(emel::decoder::event::validate{.error_out = &err}, ctx);
  CHECK(err == EMEL_OK);

  ctx.output_mask = output_mask.data();
  ctx.output_mask_count = 0;
  CHECK(emel::decoder::guard::invalid_token_inputs(ctx));
  ctx.output_mask_count = 1;
  CHECK(emel::decoder::guard::valid_token_inputs(ctx));

  ctx.n_ubatch = -1;
  CHECK(emel::decoder::guard::invalid_token_inputs(ctx));
  ctx.n_ubatch = 0;

  ctx.seq_mask_words = 0;
  CHECK(emel::decoder::guard::invalid_token_inputs(ctx));
  ctx.seq_mask_words = emel::batch::splitter::action::SEQ_WORDS + 1;
  CHECK(emel::decoder::guard::invalid_token_inputs(ctx));
  ctx.seq_mask_words = 1;

  std::array<uint64_t, 1> seq_masks = {{1U}};
  ctx.seq_masks = seq_masks.data();
  ctx.seq_masks_count = 0;
  CHECK(emel::decoder::guard::invalid_token_inputs(ctx));
  ctx.seq_masks_count = 1;
  CHECK(emel::decoder::guard::valid_token_inputs(ctx));

  ctx.seq_primary_ids = seq_ids.data();
  ctx.seq_primary_ids_count = 0;
  CHECK(emel::decoder::guard::invalid_token_inputs(ctx));
  ctx.seq_primary_ids_count = 1;
  CHECK(emel::decoder::guard::valid_token_inputs(ctx));

  ctx.positions = positions.data();
  ctx.positions_count = 0;
  CHECK(emel::decoder::guard::invalid_token_inputs(ctx));
  ctx.positions_count = 2;
  CHECK(emel::decoder::guard::invalid_token_inputs(ctx));
  ctx.positions_count = 1;
  CHECK(emel::decoder::guard::valid_token_inputs(ctx));

  ctx.outputs_capacity = -1;
  CHECK(emel::decoder::guard::invalid_token_inputs(ctx));
  ctx.outputs_capacity = 0;
  CHECK(emel::decoder::guard::valid_token_inputs(ctx));
}

TEST_CASE("decoder_run_initialize_batch_supports_output_all_with_seq_masks") {
  emel::decoder::action::context ctx{};
  std::array<int32_t, 3> tokens = {{1, 2, 3}};
  std::array<uint64_t, 3> masks = {{uint64_t{1} << 5, uint64_t{1} << 5, uint64_t{1} << 5}};
  int32_t err = EMEL_OK;

  ctx.token_ids = tokens.data();
  ctx.n_tokens = static_cast<int32_t>(tokens.size());
  ctx.n_ubatch = 2;
  ctx.output_all = true;
  ctx.seq_masks = masks.data();
  ctx.seq_masks_count = static_cast<int32_t>(masks.size());
  ctx.seq_mask_words = 1;

  emel::decoder::action::run_initialize_batch(
    emel::decoder::event::initialize_batch{.error_out = &err}, ctx);

  CHECK(err == EMEL_OK);
  CHECK(ctx.outputs_total == 3);
  CHECK(ctx.ubatch_outputs[0] == 2);
  CHECK(ctx.ubatch_outputs[1] == 1);
  CHECK(ctx.ubatch_seq_ids[0] == 5);
}

TEST_CASE("decoder_run_initialize_batch_uses_output_mask_and_primary_ids") {
  emel::decoder::action::context ctx{};
  std::array<int32_t, 4> tokens = {{1, 2, 3, 4}};
  std::array<int8_t, 4> output_mask = {{1, 0, 1, 0}};
  std::array<int32_t, 4> primary_ids = {{2, 2, 2, 2}};
  int32_t err = EMEL_OK;

  ctx.token_ids = tokens.data();
  ctx.n_tokens = static_cast<int32_t>(tokens.size());
  ctx.n_ubatch = 2;
  ctx.output_all = false;
  ctx.output_mask = output_mask.data();
  ctx.output_mask_count = static_cast<int32_t>(output_mask.size());
  ctx.seq_primary_ids = primary_ids.data();
  ctx.seq_primary_ids_count = static_cast<int32_t>(primary_ids.size());
  ctx.seq_mask_words = 1;

  emel::decoder::action::run_initialize_batch(
    emel::decoder::event::initialize_batch{.error_out = &err}, ctx);

  CHECK(err == EMEL_OK);
  CHECK(ctx.outputs_total == 2);
  CHECK(ctx.ubatch_seq_ids[0] == 2);
}

TEST_CASE("decoder_run_initialize_batch_defaults_to_last_token_output") {
  emel::decoder::action::context ctx{};
  std::array<int32_t, 3> tokens = {{1, 2, 3}};
  int32_t err = EMEL_OK;

  ctx.token_ids = tokens.data();
  ctx.n_tokens = static_cast<int32_t>(tokens.size());
  ctx.n_ubatch = 2;
  ctx.output_all = false;
  ctx.output_mask = nullptr;
  ctx.seq_mask_words = 1;

  emel::decoder::action::run_initialize_batch(
    emel::decoder::event::initialize_batch{.error_out = &err}, ctx);

  CHECK(err == EMEL_OK);
  CHECK(ctx.outputs_total == 1);
}

TEST_CASE("decoder_run_reserve_output_reports_negative_total") {
  emel::decoder::action::context ctx{};
  int32_t err = EMEL_OK;

  ctx.outputs_total = -1;
  emel::decoder::action::reject_invalid_reserve_output(
    emel::decoder::event::reserve_output{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("decoder_run_reserve_output_enforces_capacity") {
  emel::decoder::action::context ctx{};
  int32_t err = EMEL_OK;

  ctx.outputs_total = 5;
  ctx.outputs_capacity = 4;
  emel::decoder::action::run_reserve_output(
    emel::decoder::event::reserve_output{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("decoder_run_process_ubatch_reports_bounds_error") {
  emel::decoder::action::context ctx{};
  int32_t err = EMEL_OK;
  bool rollback_needed = false;

  ctx.ubatches_total = 1;
  ctx.ubatches_processed = 1;
  emel::decoder::action::on_invalid_ubatch_size(
    emel::decoder::event::process_ubatch{
      .error_out = &err,
      .rollback_needed_out = &rollback_needed,
    },
    ctx);

  CHECK(err == EMEL_ERR_BACKEND);
  CHECK_FALSE(rollback_needed);
}

TEST_CASE("decoder_run_rollback_ubatch_skips_when_not_needed") {
  emel::decoder::action::context ctx{};
  int32_t err = EMEL_OK;

  emel::decoder::action::run_rollback_ubatch(
    emel::decoder::event::rollback_ubatch{
      .error_out = &err,
      .rollback_needed = false,
    },
    ctx);
  CHECK(err == EMEL_OK);
}

TEST_CASE("decoder_run_finalize_outputs_reports_mismatch") {
  emel::decoder::action::context ctx{};
  int32_t err = EMEL_OK;

  ctx.outputs_processed = 0;
  ctx.outputs_total = 1;
  emel::decoder::action::run_finalize_outputs(
    emel::decoder::event::finalize_outputs{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("decoder_dispatch_actions_honor_owner_callback") {
  owner_probe owner{};
  int32_t err = EMEL_OK;
  emel::decoder::action::context ctx{};

  emel::decoder::action::dispatch_decoding_done_to_owner(
    emel::decoder::events::decoding_done{
      .outputs = 1,
      .error_out = &err,
      .owner_sm = &owner,
      .dispatch_event = &on_decode_event,
    },
    ctx);
  CHECK(err == EMEL_OK);
  CHECK(owner.done_calls == 1);

  emel::decoder::action::dispatch_decoding_error_to_owner(
    emel::decoder::events::decoding_error{
      .err = EMEL_ERR_BACKEND,
      .error_out = &err,
      .owner_sm = &owner,
      .dispatch_event = &on_decode_event,
    },
    ctx);
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK(owner.error_calls == 1);
}

TEST_CASE("decoder_dispatch_actions_handle_null_callback") {
  int32_t err = EMEL_ERR_BACKEND;
  emel::decoder::action::context ctx{};

  emel::decoder::action::dispatch_decoding_done_to_owner(
    emel::decoder::events::decoding_done{
      .outputs = 1,
      .error_out = &err,
      .owner_sm = nullptr,
      .dispatch_event = nullptr,
    },
    ctx);
  CHECK(err == EMEL_OK);

  err = EMEL_OK;
  emel::decoder::action::dispatch_decoding_error_to_owner(
    emel::decoder::events::decoding_error{
      .err = EMEL_OK,
      .error_out = &err,
      .owner_sm = nullptr,
      .dispatch_event = nullptr,
    },
    ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("decoder_run_initialize_batch_sets_default_ubatch") {
  emel::decoder::action::context ctx{};
  std::array<int32_t, 1> tokens = {{1}};
  int32_t err = EMEL_OK;

  ctx.token_ids = tokens.data();
  ctx.n_tokens = 1;
  ctx.n_ubatch = 0;

  emel::decoder::action::run_initialize_batch(
    emel::decoder::event::initialize_batch{.error_out = &err}, ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.n_ubatch == ctx.n_tokens);
}

TEST_CASE("decoder_run_initialize_batch_reports_split_failure") {
  emel::decoder::action::context ctx{};
  int32_t err = EMEL_OK;
  ctx.token_ids = nullptr;
  ctx.n_tokens = 0;
  ctx.n_ubatch = 0;

  emel::decoder::action::run_initialize_batch(
    emel::decoder::event::initialize_batch{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("decoder_prepare_failure_classification_paths") {
  using emel::decoder::action::classify_prepare_failure_from_memory_status;
  using emel::decoder::action::prepare_failure_kind;
  using emel::memory::coordinator::event::memory_status;

  CHECK(classify_prepare_failure_from_memory_status(memory_status::success) ==
        prepare_failure_kind::none);
  CHECK(classify_prepare_failure_from_memory_status(memory_status::failed_prepare) ==
        prepare_failure_kind::retryable);
  CHECK(classify_prepare_failure_from_memory_status(memory_status::failed_compute) ==
        prepare_failure_kind::permanent);
  CHECK(classify_prepare_failure_from_memory_status(memory_status::no_update) ==
        prepare_failure_kind::permanent);
}

TEST_CASE("decoder_update_status_error_paths") {
  using emel::decoder::action::update_status_is_error;
  using emel::memory::coordinator::event::memory_status;

  CHECK_FALSE(update_status_is_error(memory_status::success));
  CHECK_FALSE(update_status_is_error(memory_status::no_update));
  CHECK(update_status_is_error(memory_status::failed_prepare));
  CHECK(update_status_is_error(memory_status::failed_compute));
}

TEST_CASE("decoder_phase_helpers_cover_optimize_and_reserve_paths") {
  emel::decoder::action::context ctx{};
  int32_t err = EMEL_OK;

  emel::decoder::action::run_optimize_memory(
    emel::decoder::event::optimize_memory{.error_out = &err}, ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.phase_error == EMEL_OK);

  err = EMEL_OK;
  emel::decoder::action::run_reserve_output(
    emel::decoder::event::reserve_output{.error_out = &err}, ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.phase_error == EMEL_OK);

  err = EMEL_OK;
  emel::decoder::action::reject_invalid_reserve_output(
    emel::decoder::event::reserve_output{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);
}

TEST_CASE("decoder_on_invalid_ubatch_size_sets_error") {
  emel::decoder::action::context ctx{};

  emel::decoder::action::on_invalid_ubatch_size(emel::decoder::event::decode{}, ctx);
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);
  CHECK(ctx.ubatch_error == EMEL_ERR_BACKEND);
}

TEST_CASE("decoder_finalize_outputs_phase_updates_status") {
  emel::decoder::action::context ctx{};

  ctx.outputs_processed = 1;
  ctx.outputs_total = 1;
  emel::decoder::action::run_finalize_outputs(emel::decoder::event::decode{}, ctx);
  CHECK(ctx.phase_error == EMEL_OK);

  ctx.outputs_total = 2;
  emel::decoder::action::run_finalize_outputs(emel::decoder::event::decode{}, ctx);
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);
}

TEST_CASE("decoder_action_markers_cover_error_paths") {
  emel::decoder::action::context ctx{};
  int32_t err = EMEL_OK;

  ctx.last_error = EMEL_ERR_BACKEND;
  emel::decoder::action::mark_done(ctx);
  CHECK(ctx.last_error == EMEL_OK);

  ctx.phase_error = EMEL_OK;
  ctx.last_error = EMEL_OK;
  emel::decoder::action::capture_rollback_error(ctx);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);

  ctx.ubatch_error = EMEL_OK;
  ctx.last_error = EMEL_OK;
  emel::decoder::action::capture_ubatch_error(ctx);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);

  ctx.phase_error = EMEL_OK;
  ctx.last_error = EMEL_OK;
  emel::decoder::action::ensure_last_error(ctx);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);

  emel::decoder::action::on_unexpected(
    emel::decoder::event::finalize_outputs{.error_out = &err},
    ctx);
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);
}

TEST_CASE("decoder_run_process_ubatch_reports_invalid_executor") {
  emel::decoder::action::context ctx{};
  int32_t err = EMEL_OK;
  bool rollback_needed = false;

  ctx.ubatches_total = 1;
  ctx.ubatches_processed = 0;
  ctx.ubatch_sizes[0] = 1;
  ctx.n_tokens = 1;
  ctx.token_indices_count = 1;
  ctx.ubatch_token_indices[0] = 0;
  ctx.ubatch_token_offsets[0] = 0;
  ctx.ubatch_token_offsets[1] = 1;
  ctx.kv_cache.reset();

  emel::decoder::action::run_process_ubatch(
    emel::decoder::event::process_ubatch{
      .error_out = &err,
      .rollback_needed_out = &rollback_needed,
    },
    ctx);

  CHECK(err == EMEL_ERR_BACKEND);
  CHECK(rollback_needed);
}

TEST_CASE("decoder_run_prepare_memory_batch_reports_errors") {
  emel::decoder::action::context ctx{};
  int32_t err = EMEL_OK;
  bool retryable = false;

  ctx.n_ubatch = 0;
  ctx.ubatches_total = 0;
  emel::decoder::action::run_prepare_memory_batch(
    emel::decoder::event::prepare_memory_batch{
      .error_out = &err,
      .retryable_out = &retryable,
    },
    ctx);
  CHECK(err == EMEL_ERR_BACKEND);

  ctx.n_ubatch = 1;
  ctx.ubatches_total = 1;
  ctx.n_tokens = emel::kv::cache::action::MAX_KV_CELLS + 1;
  err = EMEL_OK;
  retryable = false;
  emel::decoder::action::run_prepare_memory_batch(
    emel::decoder::event::prepare_memory_batch{
      .error_out = &err,
      .retryable_out = &retryable,
    },
    ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("decoder_run_process_ubatch_populates_positions_and_seq_metadata") {
  emel::decoder::action::context ctx{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  std::array<int32_t, 6> positions = {{10, 11, 20, 21, 30, 31}};
  std::array<uint64_t, 2> masks = {{1U, 2U}};
  std::array<int32_t, 2> primary_ids = {{5, 6}};
  int32_t err = EMEL_OK;
  bool rollback_needed = false;

  ctx.token_ids = tokens.data();
  ctx.n_tokens = static_cast<int32_t>(tokens.size());
  ctx.ubatches_total = 1;
  ctx.ubatches_processed = 0;
  ctx.ubatch_sizes[0] = 2;
  ctx.ubatch_outputs[0] = 2;
  ctx.token_indices_count = 2;
  ctx.ubatch_token_indices[0] = 0;
  ctx.ubatch_token_indices[1] = 1;
  ctx.ubatch_token_offsets[0] = 0;
  ctx.ubatch_token_offsets[1] = 2;

  ctx.positions = positions.data();
  ctx.positions_count = static_cast<int32_t>(positions.size());

  ctx.seq_masks = masks.data();
  ctx.seq_masks_count = static_cast<int32_t>(masks.size());
  ctx.seq_mask_words = 1;

  ctx.seq_primary_ids = primary_ids.data();
  ctx.seq_primary_ids_count = static_cast<int32_t>(primary_ids.size());

  ctx.compute_validate = compute_validate;
  ctx.compute_prepare_graph = compute_prepare_graph;
  ctx.compute_alloc_graph = compute_alloc_graph;
  ctx.compute_bind_inputs = compute_bind_inputs;
  ctx.compute_run_backend = compute_run_backend;
  ctx.compute_extract_outputs = compute_extract_outputs;

  int32_t ubatch_size = 2;
  CHECK(ctx.kv_cache->process_event(emel::kv::cache::event::prepare{
    .ubatch_sizes = &ubatch_size,
    .ubatch_count = 1,
    .requested_capacity = 8,
  }));

  emel::decoder::action::run_process_ubatch(
    emel::decoder::event::process_ubatch{
      .error_out = &err,
      .rollback_needed_out = &rollback_needed,
    },
    ctx);

  (void)err;
  (void)rollback_needed;
  CHECK(ctx.ubatch_positions[0] == 10);
  CHECK(ctx.ubatch_positions[1] == 11);
  CHECK(ctx.ubatch_positions[2] == 20);
  CHECK(ctx.ubatch_positions[3] == 21);
  CHECK(ctx.ubatch_positions[4] == 30);
  CHECK(ctx.ubatch_positions[5] == 31);
  CHECK(ctx.ubatch_seq_masks[0] == 1U);
  CHECK(ctx.ubatch_seq_masks[1] == 2U);
  CHECK(ctx.ubatch_seq_primary_ids[0] == 5);
  CHECK(ctx.ubatch_seq_primary_ids[1] == 6);
}

TEST_CASE("decoder_run_process_ubatch_reports_executor_errors") {
  emel::decoder::action::context ctx{};
  int32_t err = EMEL_OK;
  bool rollback_needed = false;

  ctx.ubatches_total = 1;
  ctx.ubatches_processed = 0;
  ctx.ubatch_sizes[0] = 0;
  emel::decoder::action::run_process_ubatch(
    emel::decoder::event::process_ubatch{
      .error_out = &err,
      .rollback_needed_out = &rollback_needed,
    },
    ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("decoder_actions_use_context_for_phase_runs") {
  emel::decoder::action::context ctx{};

  ctx.n_tokens = 1;
  ctx.n_ubatch = 1;
  ctx.ubatches_total = 1;
  ctx.ubatch_sizes[0] = 1;

  emel::decoder::action::run_update_memory(emel::decoder::event::decode{}, ctx);
  CHECK(ctx.phase_error == EMEL_OK);

  emel::decoder::action::run_prepare_memory_batch(emel::decoder::event::decode{}, ctx);
  CHECK(ctx.phase_error == EMEL_OK);

  emel::decoder::action::run_optimize_memory(emel::decoder::event::decode{}, ctx);
  CHECK(ctx.phase_error == EMEL_OK);

  emel::decoder::action::run_reserve_output(emel::decoder::event::decode{}, ctx);
  CHECK(ctx.phase_error == EMEL_OK);

  emel::decoder::action::reject_invalid_reserve_output(emel::decoder::event::decode{}, ctx);
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);

  ctx.ubatches_total = 0;
  emel::decoder::action::run_process_ubatch(emel::decoder::event::decode{}, ctx);
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);

  ctx.rollback_needed = false;
  emel::decoder::action::run_rollback_ubatch(emel::decoder::event::decode{}, ctx);
  CHECK(ctx.phase_error == EMEL_OK);
}

TEST_CASE("decoder_actions_report_missing_dependencies") {
  emel::decoder::action::context ctx{};
  int32_t err = EMEL_OK;

  ctx.memory_coordinator.reset();
  emel::decoder::action::run_update_memory(
    emel::decoder::event::update_memory{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  err = EMEL_OK;
  ctx.kv_cache.reset();
  emel::decoder::action::run_prepare_memory_batch(
    emel::decoder::event::prepare_memory_batch{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  err = EMEL_OK;
  ctx.memory_coordinator.reset();
  emel::decoder::action::run_optimize_memory(
    emel::decoder::event::optimize_memory{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("decoder_run_process_ubatch_reports_missing_executor") {
  emel::decoder::action::context ctx{};
  int32_t err = EMEL_OK;
  bool rollback_needed = false;

  ctx.ubatch_executor.reset();
  ctx.ubatches_total = 1;
  ctx.ubatches_processed = 0;
  ctx.ubatch_sizes[0] = 1;

  emel::decoder::action::run_process_ubatch(
    emel::decoder::event::process_ubatch{
      .error_out = &err,
      .rollback_needed_out = &rollback_needed,
    },
    ctx);
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK(rollback_needed);
}

TEST_CASE("decoder_run_rollback_ubatch_reports_missing_kv_cache") {
  emel::decoder::action::context ctx{};
  int32_t err = EMEL_OK;

  ctx.kv_cache.reset();
  emel::decoder::action::run_rollback_ubatch(
    emel::decoder::event::rollback_ubatch{
      .error_out = &err,
      .rollback_needed = true,
    },
    ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}
