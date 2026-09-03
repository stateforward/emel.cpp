#pragma once

#include <cstdint>
#include <span>
#include <string_view>

#include "emel/error/error.hpp"
#include "emel/model/needle/request/errors.hpp"

namespace emel::model::needle::request {

inline constexpr uint32_t k_default_max_new_tokens = 80u;
using timestamp_now_fn = uint64_t (*)() noexcept;
inline constexpr std::string_view k_phase_noncomparable_reason =
    "closed_reference_phase_contract_missing_token_counts_and_timestamps";

namespace events {

struct configured;
struct reset_done;
struct completed;
struct request_error;

} // namespace events

namespace event {

using configured_fn = void (*)(const events::configured &) noexcept;
using reset_done_fn = void (*)(const events::reset_done &) noexcept;
using completed_fn = void (*)(const events::completed &) noexcept;
using request_error_fn = void (*)(const events::request_error &) noexcept;

// Initialization-only. Both views are copied into request-owned fixed storage;
// tools_json must be a compact JSON array matching the public Needle constructor.
struct configure {
  std::string_view system = {};
  std::string_view tools_json = {};
  configured_fn on_done = nullptr;
  request_error_fn on_error = nullptr;
};

// Reset is deliberately a separate public event so callers can exclude it from
// an external wall timer exactly as the closed Cactus API does.
struct reset {
  reset_done_fn on_done = nullptr;
  request_error_fn on_error = nullptr;
};

// Runtime request. The borrowed query is consumed synchronously and is never
// retained after process_event returns. Every output span below remains valid
// until the next configure, reset, or complete dispatch on the same machine.
struct complete {
  std::string_view query = {};
  uint32_t max_new_tokens = k_default_max_new_tokens;
  completed_fn on_done = nullptr;
  request_error_fn on_error = nullptr;
};
// Proof-only preparation follows the production render/tokenize actions but
// deliberately stops before graph execution.
struct prepare {
  std::string_view query = {};
  uint32_t max_new_tokens = k_default_max_new_tokens;
};

struct configure_ctx {
  emel::error::type err = emel::error::cast(error::none);
};

struct reset_ctx {
  emel::error::type err = emel::error::cast(error::none);
};

struct complete_ctx {
  emel::error::type err = emel::error::cast(error::none);
  std::string_view normalized_envelope = {};
  std::span<const int32_t> generated_token_ids = {};
  uint32_t prompt_tokens = 0u;
  uint32_t generated_tokens = 0u;
  uint64_t prefill_nanoseconds = 0u;
  uint64_t decode_nanoseconds = 0u;
  timestamp_now_fn timestamp_now = nullptr;
};

struct configure_run {
  const configure &request;
  configure_ctx &ctx;
};

struct reset_run {
  const reset &request;
  reset_ctx &ctx;
};

struct complete_run {
  const complete &request;
  complete_ctx &ctx;
};

struct prepare_ctx {
  emel::error::type err = emel::error::cast(error::none);
};

struct prepare_run {
  const prepare &request;
  prepare_ctx &ctx;
};

} // namespace event

namespace events {

struct configured {
  const event::configure &request;
};

struct reset_done {
  const event::reset &request;
};

struct completed {
  const event::complete &request;
  std::string_view normalized_envelope = {};
  std::span<const int32_t> generated_token_ids = {};
  uint32_t prompt_tokens = 0u;
  uint32_t generated_tokens = 0u;
  uint64_t prefill_nanoseconds = 0u;
  uint64_t decode_nanoseconds = 0u;
};

struct request_error {
  emel::error::type err = emel::error::cast(error::none);
};

} // namespace events

} // namespace emel::model::needle::request
