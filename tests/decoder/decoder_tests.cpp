#include <array>
#include <boost/sml.hpp>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/decoder/actions.hpp"
#include "emel/decoder/guards.hpp"
#include "emel/decoder/sm.hpp"
#include "emel/emel.h"
#include "emel/kv/cache/actions.hpp"

namespace {

struct owner_probe {
  int32_t done_calls = 0;
  int32_t error_calls = 0;
  int32_t outputs = -1;
  int32_t err = EMEL_OK;
};

bool on_decode_event(
    void * owner_sm, const emel::decoder::events::owner_event & ev) {
  auto * owner = static_cast<owner_probe *>(owner_sm);
  if (owner == nullptr) return false;
  if (ev.type == emel::decoder::events::owner_event::kind::done) {
    owner->done_calls += 1;
    owner->outputs = ev.done.outputs;
  } else if (ev.type == emel::decoder::events::owner_event::kind::error) {
    owner->error_calls += 1;
    owner->err = ev.error.err;
  }
  return true;
}

}  // namespace

TEST_CASE("decoder_starts_initialized") {
  emel::decoder::sm machine{};
  CHECK(machine.is(boost::sml::state<emel::decoder::initialized>));
}

TEST_CASE("decoder_decode_rejects_invalid_payload_without_callback") {
  emel::decoder::sm machine{};
  int32_t error = EMEL_OK;

  CHECK_FALSE(machine.process_event(emel::decoder::event::decode{
    .token_ids = nullptr,
    .n_tokens = 0,
    .n_ubatch = 2,
    .owner_sm = nullptr,
    .dispatch_event = nullptr,
    .error_out = &error,
  }));

  CHECK(machine.is(boost::sml::state<emel::decoder::initialized>));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("decoder_decode_invalid_payload_dispatches_error_and_returns_false") {
  emel::decoder::sm machine{};
  owner_probe owner{};
  int32_t error = EMEL_OK;

  CHECK_FALSE(machine.process_event(emel::decoder::event::decode{
    .token_ids = nullptr,
    .n_tokens = 3,
    .n_ubatch = 2,
    .owner_sm = &owner,
    .dispatch_event = &on_decode_event,
    .error_out = &error,
  }));

  CHECK(machine.is(boost::sml::state<emel::decoder::initialized>));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(owner.done_calls == 0);
  CHECK(owner.error_calls == 1);
  CHECK(owner.err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("decoder_decode_rejects_default_ubatch_without_tokens") {
  emel::decoder::sm machine{};
  owner_probe owner{};
  int32_t error = EMEL_OK;

  CHECK_FALSE(machine.process_event(emel::decoder::event::decode{
    .token_ids = nullptr,
    .n_tokens = 0,
    .n_ubatch = 0,
    .owner_sm = &owner,
    .dispatch_event = &on_decode_event,
    .error_out = &error,
  }));

  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(owner.done_calls == 0);
  CHECK(owner.error_calls == 1);
}

TEST_CASE("decoder_guards_cover_progress_paths") {
  emel::decoder::action::context ctx{};
  ctx.ubatches_total = 3;
  ctx.ubatches_processed = 1;
  CHECK(emel::decoder::guard::has_more_ubatches(ctx));
  CHECK_FALSE(emel::decoder::guard::no_more_ubatches(ctx));

  ctx.ubatches_processed = 3;
  CHECK_FALSE(emel::decoder::guard::has_more_ubatches(ctx));
  CHECK(emel::decoder::guard::no_more_ubatches(ctx));
}

TEST_CASE("decoder_guard_reports_invalid_outputs_total") {
  emel::decoder::action::context ctx{};
  ctx.outputs_total = -1;
  CHECK(emel::decoder::guard::invalid_outputs_total(ctx));

  ctx.outputs_total = 0;
  CHECK_FALSE(emel::decoder::guard::invalid_outputs_total(ctx));
}

TEST_CASE("decoder_maps_memory_status_to_prepare_failure_classes") {
  using emel::decoder::action::classify_prepare_failure_from_memory_status;
  using emel::decoder::action::prepare_failure_kind;
  using emel::memory::coordinator::event::memory_status;

  CHECK(
      classify_prepare_failure_from_memory_status(memory_status::success) ==
      prepare_failure_kind::none);
  CHECK(
      classify_prepare_failure_from_memory_status(memory_status::failed_prepare) ==
      prepare_failure_kind::retryable);
  CHECK(
      classify_prepare_failure_from_memory_status(memory_status::no_update) ==
      prepare_failure_kind::permanent);
  CHECK(
      classify_prepare_failure_from_memory_status(memory_status::failed_compute) ==
      prepare_failure_kind::permanent);
}

TEST_CASE("decoder_prepare_memory_batch_reports_backend_on_memory_coordinator_failure") {
  emel::decoder::action::context ctx{};
  ctx.n_ubatch = 0;
  ctx.ubatches_total = 1;

  int32_t err = EMEL_OK;
  bool retryable = false;
  emel::decoder::action::run_prepare_memory_batch(
    emel::decoder::event::prepare_memory_batch{
      .error_out = &err,
      .retryable_out = &retryable,
    },
    ctx);

  CHECK(err == EMEL_ERR_BACKEND);
  CHECK_FALSE(retryable);
}

TEST_CASE("decoder_prepare_memory_batch_reports_kv_failure") {
  emel::decoder::action::context ctx{};
  ctx.n_ubatch = 1;
  ctx.ubatches_total = 1;
  ctx.n_tokens = emel::kv::cache::action::MAX_KV_CELLS + 1;

  int32_t err = EMEL_OK;
  bool retryable = false;
  emel::decoder::action::run_prepare_memory_batch(
    emel::decoder::event::prepare_memory_batch{
      .error_out = &err,
      .retryable_out = &retryable,
    },
    ctx);

  CHECK(err == EMEL_ERR_BACKEND);
  CHECK_FALSE(retryable);
}

TEST_CASE("decoder_process_ubatch_reports_invalid_ubatch_size") {
  emel::decoder::action::context ctx{};
  ctx.ubatches_total = 1;
  ctx.ubatches_processed = 0;
  ctx.ubatch_sizes[0] = 0;

  int32_t err = EMEL_OK;
  bool rollback_needed = false;
  emel::decoder::action::on_invalid_ubatch_size(
    emel::decoder::event::process_ubatch{
      .error_out = &err,
      .rollback_needed_out = &rollback_needed,
    },
    ctx);

  CHECK(err == EMEL_ERR_BACKEND);
  CHECK_FALSE(rollback_needed);
}

TEST_CASE("decoder_maps_memory_update_status_error_classification") {
  using emel::decoder::action::update_status_is_error;
  using emel::memory::coordinator::event::memory_status;

  CHECK_FALSE(update_status_is_error(memory_status::success));
  CHECK_FALSE(update_status_is_error(memory_status::no_update));
  CHECK(update_status_is_error(memory_status::failed_prepare));
  CHECK(update_status_is_error(memory_status::failed_compute));
}

TEST_CASE("decoder_classification_helpers_handle_unknown_memory_status_values") {
  using emel::decoder::action::classify_prepare_failure_from_memory_status;
  using emel::decoder::action::prepare_failure_kind;
  using emel::decoder::action::update_status_is_error;
  using emel::memory::coordinator::event::memory_status;

  const auto unknown_status = static_cast<memory_status>(99);
  CHECK(
      classify_prepare_failure_from_memory_status(unknown_status) ==
      prepare_failure_kind::permanent);
  CHECK(update_status_is_error(unknown_status));
}

TEST_CASE("decoder_finalize_action_reports_backend_error_when_output_count_mismatch") {
  emel::decoder::action::context ctx{};
  ctx.outputs_total = 4;
  ctx.outputs_processed = 3;

  int32_t error_out = EMEL_OK;
  emel::decoder::action::run_finalize_outputs(
      emel::decoder::event::finalize_outputs{
          .error_out = &error_out,
      },
      ctx);

  CHECK(error_out == EMEL_ERR_BACKEND);
}

TEST_CASE("decoder_action_helpers_cover_error_and_null_output_edges") {
  int32_t error_out = EMEL_OK;

  {
    emel::decoder::action::context ctx{};
    emel::decoder::action::run_validate(
        emel::decoder::event::validate{
            .error_out = nullptr,
        },
        ctx);

    ctx.n_tokens = 0;
    ctx.token_ids = nullptr;
    emel::decoder::action::reject_invalid_validate(
        emel::decoder::event::validate{
            .error_out = &error_out,
        },
        ctx);
    CHECK(error_out == EMEL_ERR_INVALID_ARGUMENT);

    error_out = EMEL_OK;
    std::array<int32_t, 1> valid_tokens = {{7}};
    ctx.n_tokens = 1;
    ctx.token_ids = valid_tokens.data();
    emel::decoder::action::run_validate(
        emel::decoder::event::validate{
            .error_out = &error_out,
        },
        ctx);
    CHECK(error_out == EMEL_OK);
  }

  {
    emel::decoder::action::context ctx{};
    error_out = EMEL_OK;
    ctx.n_tokens = 0;
    ctx.n_ubatch = 0;
    emel::decoder::action::run_initialize_batch(
        emel::decoder::event::initialize_batch{
            .error_out = &error_out,
        },
        ctx);
    CHECK(error_out == EMEL_ERR_BACKEND);
  }

  {
    emel::decoder::action::context ctx{};
    emel::decoder::action::run_update_memory(
        emel::decoder::event::update_memory{
            .error_out = nullptr,
        },
        ctx);
    emel::decoder::action::run_prepare_memory_batch(
        emel::decoder::event::prepare_memory_batch{
            .error_out = nullptr,
        },
        ctx);
    emel::decoder::action::run_optimize_memory(
        emel::decoder::event::optimize_memory{
            .error_out = nullptr,
        },
        ctx);

    error_out = EMEL_OK;
    emel::decoder::action::run_optimize_memory(
        emel::decoder::event::optimize_memory{
            .error_out = &error_out,
        },
        ctx);
    CHECK(error_out == EMEL_OK);

    error_out = EMEL_OK;
    ctx.outputs_total = -1;
    emel::decoder::action::reject_invalid_reserve_output(
        emel::decoder::event::reserve_output{
            .error_out = &error_out,
        },
        ctx);
    CHECK(error_out == EMEL_ERR_BACKEND);

    error_out = EMEL_OK;
    ctx.ubatches_total = 1;
    ctx.ubatches_processed = 1;
    emel::decoder::action::on_invalid_ubatch_size(
        emel::decoder::event::process_ubatch{
            .error_out = &error_out,
        },
        ctx);
    CHECK(error_out == EMEL_ERR_BACKEND);
  }

  {
    emel::decoder::action::context ctx{};
    error_out = EMEL_OK;
    ctx.n_tokens = 0;
    ctx.n_ubatch = 1;
    ctx.ubatches_total = 1;
    emel::decoder::action::on_invalid_ubatch_size(
        emel::decoder::event::process_ubatch{
            .error_out = &error_out,
        },
        ctx);
    CHECK(error_out == EMEL_ERR_BACKEND);

    error_out = EMEL_OK;
    ctx.outputs_total = 1;
    ctx.outputs_processed = 2;
    emel::decoder::action::run_rollback_ubatch(
        emel::decoder::event::rollback_ubatch{
            .error_out = &error_out,
            .rollback_needed = true,
        },
        ctx);
    CHECK(error_out == EMEL_ERR_BACKEND);
  }

  {
    owner_probe owner{};
    emel::decoder::action::context ctx{};
    int32_t error_out = EMEL_OK;
    emel::decoder::action::dispatch_decoding_done_to_owner(
        emel::decoder::events::decoding_done{
            .outputs = 12,
            .error_out = &error_out,
            .owner_sm = &owner,
            .dispatch_event = nullptr,
        },
        ctx);
    CHECK(error_out == EMEL_OK);
    CHECK(owner.done_calls == 0);

    emel::decoder::action::dispatch_decoding_error_to_owner(
        emel::decoder::events::decoding_error{
            .err = EMEL_ERR_BACKEND,
            .error_out = &error_out,
            .owner_sm = &owner,
            .dispatch_event = nullptr,
        },
        ctx);
    CHECK(error_out == EMEL_ERR_BACKEND);
    CHECK(owner.error_calls == 0);
  }
}

TEST_CASE("decoder_action_helpers_cover_memory_machine_failure_and_owner_error_dispatch") {
  int32_t error_out = EMEL_OK;
  std::array<int32_t, 2> tokens = {{1, 2}};

  {
    emel::decoder::action::context ctx{};
    ctx.token_ids = tokens.data();
    ctx.n_tokens = static_cast<int32_t>(tokens.size());
    ctx.n_ubatch = 1;

    using mem_base_t = emel::memory::coordinator::sm::base_type;
    auto & mem_base = static_cast<mem_base_t &>(*ctx.memory_coordinator);
    emel::memory::coordinator::event::memory_status status =
        emel::memory::coordinator::event::memory_status::success;
    CHECK(mem_base.process_event(emel::memory::coordinator::event::prepare_update{
      .optimize = false,
      .status_out = &status,
    }));

    emel::decoder::action::run_update_memory(
        emel::decoder::event::update_memory{
            .error_out = &error_out,
        },
        ctx);
    CHECK(error_out == EMEL_OK);
  }

  {
    emel::decoder::action::context ctx{};
    using mem_base_t = emel::memory::coordinator::sm::base_type;
    auto & mem_base = static_cast<mem_base_t &>(*ctx.memory_coordinator);
    emel::memory::coordinator::event::memory_status status =
        emel::memory::coordinator::event::memory_status::success;
    CHECK(mem_base.process_event(emel::memory::coordinator::event::prepare_update{
      .optimize = true,
      .status_out = &status,
    }));

    error_out = EMEL_OK;
    emel::decoder::action::run_optimize_memory(
        emel::decoder::event::optimize_memory{
            .error_out = &error_out,
        },
        ctx);
    CHECK(error_out == EMEL_OK);
  }

  {
    owner_probe owner{};
    emel::decoder::action::context ctx{};

    emel::decoder::action::dispatch_decoding_error_to_owner(
        emel::decoder::events::decoding_error{
            .err = EMEL_ERR_BACKEND,
            .error_out = &error_out,
            .owner_sm = &owner,
            .dispatch_event = &on_decode_event,
        },
        ctx);

    CHECK(error_out == EMEL_ERR_BACKEND);
    CHECK(owner.done_calls == 0);
    CHECK(owner.error_calls == 1);
    CHECK(owner.err == EMEL_ERR_BACKEND);
  }
}

TEST_CASE("decoder_action_helpers_cover_prepare_and_ubatch_failure_branches") {
  emel::decoder::action::context ctx{};
  int32_t error_out = EMEL_OK;
  std::array<int32_t, 2> tokens = {{1, 2}};

  ctx.token_ids = tokens.data();
  ctx.n_tokens = static_cast<int32_t>(tokens.size());
  ctx.n_ubatch = 0;
  ctx.ubatches_total = 1;
  emel::decoder::action::run_prepare_memory_batch(
      emel::decoder::event::prepare_memory_batch{
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_ERR_BACKEND);

  ctx.n_ubatch = 1;
  ctx.ubatches_total = 1;
  ctx.ubatch_sizes.fill(0);
  emel::decoder::action::run_prepare_memory_batch(
      emel::decoder::event::prepare_memory_batch{
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_ERR_BACKEND);

  {
    emel::decoder::action::context process_ctx{};
    process_ctx.ubatches_total = 1;
    process_ctx.ubatches_processed = 1;
    process_ctx.ubatch_sizes[0] = 1;

    emel::decoder::action::on_invalid_ubatch_size(
        emel::decoder::event::process_ubatch{
            .error_out = &error_out,
            .rollback_needed_out = nullptr,
        },
        process_ctx);
    CHECK(error_out == EMEL_ERR_BACKEND);
  }

  {
    emel::decoder::action::context rollback_ctx{};
    error_out = EMEL_OK;
    emel::decoder::action::run_rollback_ubatch(
        emel::decoder::event::rollback_ubatch{
            .error_out = &error_out,
            .rollback_needed = false,
        },
        rollback_ctx);
    CHECK(error_out == EMEL_OK);

    error_out = EMEL_OK;
    rollback_ctx.ubatches_processed = 2;
    emel::decoder::action::run_rollback_ubatch(
        emel::decoder::event::rollback_ubatch{
            .error_out = &error_out,
            .rollback_needed = true,
        },
        rollback_ctx);
    CHECK(error_out == EMEL_ERR_BACKEND);
  }

  {
    emel::decoder::action::context rollback_ok_ctx{};
    error_out = EMEL_OK;
    rollback_ok_ctx.ubatches_processed = 1;
    rollback_ok_ctx.outputs_total = 1;
    rollback_ok_ctx.outputs_processed = 2;
    int32_t ubatch_size = 1;
    int32_t kv_tokens = 0;
    CHECK(rollback_ok_ctx.kv_cache->process_event(emel::kv::cache::event::prepare{
      .ubatch_sizes = &ubatch_size,
      .ubatch_count = 1,
      .requested_capacity = 4,
    }));
    CHECK(rollback_ok_ctx.kv_cache->process_event(emel::kv::cache::event::apply_ubatch{
      .ubatch_index = 0,
      .kv_tokens_out = &kv_tokens,
    }));
    emel::decoder::action::run_rollback_ubatch(
        emel::decoder::event::rollback_ubatch{
            .error_out = &error_out,
            .rollback_needed = true,
        },
        rollback_ok_ctx);
    CHECK(error_out == EMEL_ERR_BACKEND);
  }
}

TEST_CASE("decoder_decode_fails_when_batch_splitter_cannot_emit_all_ubatches") {
  emel::decoder::sm machine{};
  owner_probe owner{};
  int32_t token = 42;
  int32_t error = EMEL_OK;

  CHECK_FALSE(machine.process_event(emel::decoder::event::decode{
    .token_ids = &token,
    .n_tokens = 5000,
    .n_ubatch = 1,
    .owner_sm = &owner,
    .dispatch_event = &on_decode_event,
    .error_out = &error,
  }));

  CHECK(machine.is(boost::sml::state<emel::decoder::initialized>));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(owner.done_calls == 0);
  CHECK(owner.error_calls == 1);
  CHECK(owner.err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("decoder_decode_fails_when_kv_capacity_request_exceeds_supported_limit") {
  emel::decoder::sm machine{};
  owner_probe owner{};
  int32_t token = 7;
  int32_t error = EMEL_OK;

  CHECK_FALSE(machine.process_event(emel::decoder::event::decode{
    .token_ids = &token,
    .n_tokens = 40000,
    .n_ubatch = 40000,
    .owner_sm = &owner,
    .dispatch_event = &on_decode_event,
    .error_out = &error,
  }));

  CHECK(machine.is(boost::sml::state<emel::decoder::initialized>));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(owner.done_calls == 0);
  CHECK(owner.error_calls == 1);
  CHECK(owner.err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("decoder_rejects_repeated_invalid_decode_requests") {
  emel::decoder::sm machine{};
  int32_t error = EMEL_OK;

  CHECK_FALSE(machine.process_event(emel::decoder::event::decode{
    .token_ids = nullptr,
    .n_tokens = 0,
    .n_ubatch = 1,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);

  error = EMEL_OK;
  CHECK_FALSE(machine.process_event(emel::decoder::event::decode{
    .token_ids = nullptr,
    .n_tokens = 0,
    .n_ubatch = 1,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(machine.is(boost::sml::state<emel::decoder::initialized>));
}
