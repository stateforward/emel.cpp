#include <array>
#include <doctest/doctest.h>

#include "emel/decoder/actions.hpp"
#include "emel/decoder/events.hpp"
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

}  // namespace

TEST_CASE("decoder_actions_return_early_when_error_out_is_null") {
  emel::decoder::action::context ctx{};

  emel::decoder::action::run_validate(emel::decoder::event::validate{}, ctx);
  emel::decoder::action::run_initialize_batch(emel::decoder::event::initialize_batch{}, ctx);
  emel::decoder::action::run_update_memory(emel::decoder::event::update_memory{}, ctx);
  emel::decoder::action::run_prepare_memory_batch(emel::decoder::event::prepare_memory_batch{}, ctx);
  emel::decoder::action::run_optimize_memory(emel::decoder::event::optimize_memory{}, ctx);
  emel::decoder::action::run_reserve_output(emel::decoder::event::reserve_output{}, ctx);
  emel::decoder::action::run_process_ubatch(emel::decoder::event::process_ubatch{}, ctx);
  emel::decoder::action::run_rollback_ubatch(emel::decoder::event::rollback_ubatch{}, ctx);
  emel::decoder::action::run_finalize_outputs(emel::decoder::event::finalize_outputs{}, ctx);
}

TEST_CASE("decoder_run_validate_checks_token_inputs") {
  emel::decoder::action::context ctx{};
  int32_t err = EMEL_OK;

  ctx.token_ids = nullptr;
  ctx.n_tokens = 0;
  emel::decoder::action::reject_invalid_validate(
    emel::decoder::event::validate{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  std::array<int32_t, 1> tokens = {{1}};
  ctx.token_ids = tokens.data();
  ctx.n_tokens = 1;
  err = EMEL_OK;
  emel::decoder::action::run_validate(emel::decoder::event::validate{.error_out = &err}, ctx);
  CHECK(err == EMEL_OK);
}

TEST_CASE("decoder_run_reserve_output_reports_negative_total") {
  emel::decoder::action::context ctx{};
  int32_t err = EMEL_OK;

  ctx.outputs_total = -1;
  emel::decoder::action::reject_invalid_reserve_output(
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
  CHECK(err == EMEL_OK);

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
  CHECK(err == EMEL_OK);
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
