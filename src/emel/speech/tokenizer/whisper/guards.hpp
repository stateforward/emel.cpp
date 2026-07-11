#pragma once

#include <cstddef>
#include <cstdint>

#include "emel/speech/tokenizer/whisper/any.hpp"
#include "emel/speech/tokenizer/whisper/detail.hpp"
#include "emel/speech/tokenizer/whisper/events.hpp"

namespace emel::speech::tokenizer::whisper::guard {

// The detokenize decision validates the whole request before any transcript
// bytes are written: the tokenizer JSON must carry the tiny control tokens, and
// the caller-owned spans must be well formed. The decode action writes through
// transcript.data() and iterates token_ids, so a direct component dispatch (via
// speech::tokenizer::any / whisper::sm, bypassing the transcriber's own
// recognize validation) with a non-empty span whose .data() is null would
// otherwise reach undefined behavior instead of a clean rejection.
struct guard_tokenizer_json_valid {
  bool operator()(const event::detokenize_run &runtime_ev) const noexcept {
    return runtime_ev.request.tokenizer_json.data() != nullptr &&
           !runtime_ev.request.tokenizer_json.empty() &&
           (runtime_ev.request.token_ids.empty() ||
            runtime_ev.request.token_ids.data() != nullptr) &&
           (runtime_ev.request.transcript.empty() ||
            runtime_ev.request.transcript.data() != nullptr) &&
           detail::validate_tiny_control_tokens(
               runtime_ev.request.tokenizer_json);
  }
};

struct guard_tokenizer_json_invalid {
  bool operator()(const event::detokenize_run &runtime_ev) const noexcept {
    return !guard_tokenizer_json_valid{}(runtime_ev);
  }
};

// Token IDs are vocab indices; negative values are malformed public input. The
// decode path renders unknown IDs through write_i32 for vocab lookup, and the
// minimum int32 needs dedicated magnitude handling there, so malformed IDs are
// rejected here before the decode action ever runs. The scan is a pure bounded
// read of the request span.
struct guard_token_ids_valid {
  bool operator()(const event::detokenize_run &runtime_ev) const noexcept {
    for (const int32_t token_id : runtime_ev.request.token_ids) {
      if (token_id < 0) {
        return false;
      }
    }
    return true;
  }
};

struct guard_detokenize_request_valid {
  bool operator()(const event::detokenize_run &runtime_ev) const noexcept {
    return guard_tokenizer_json_valid{}(runtime_ev) &&
           guard_token_ids_valid{}(runtime_ev);
  }
};

struct guard_token_ids_invalid {
  bool operator()(const event::detokenize_run &runtime_ev) const noexcept {
    return guard_tokenizer_json_valid{}(runtime_ev) &&
           !guard_token_ids_valid{}(runtime_ev);
  }
};

// Asset-validation guards: the validate flow checks the tokenizer JSON control
// tokens and the bound decode policy so owners can fail initialization instead
// of deferring the failure to the first detokenize after pipeline work ran.
struct guard_validate_json_valid {
  bool operator()(const event::validate_run &runtime_ev) const noexcept {
    return runtime_ev.request.tokenizer_json.data() != nullptr &&
           !runtime_ev.request.tokenizer_json.empty() &&
           detail::validate_tiny_control_tokens(
               runtime_ev.request.tokenizer_json);
  }
};

struct guard_validate_json_invalid {
  bool operator()(const event::validate_run &runtime_ev) const noexcept {
    return !guard_validate_json_valid{}(runtime_ev);
  }
};

struct guard_validate_supported {
  bool operator()(const event::validate_run &runtime_ev) const noexcept {
    return guard_validate_json_valid{}(runtime_ev) &&
           is_tiny_asr_decode_policy_supported(
               runtime_ev.request.decode_policy);
  }
};

struct guard_validate_policy_unsupported {
  bool operator()(const event::validate_run &runtime_ev) const noexcept {
    return guard_validate_json_valid{}(runtime_ev) &&
           !is_tiny_asr_decode_policy_supported(
               runtime_ev.request.decode_policy);
  }
};

struct guard_validate_has_error_out {
  bool operator()(const event::validate_run &runtime_ev) const noexcept {
    return runtime_ev.request.error_out != nullptr;
  }
};

struct guard_validate_no_error_out {
  bool operator()(const event::validate_run &runtime_ev) const noexcept {
    return runtime_ev.request.error_out == nullptr;
  }
};

struct guard_has_error_out {
  bool operator()(const event::detokenize_run &runtime_ev) const noexcept {
    return runtime_ev.request.error_out != nullptr;
  }
};

struct guard_no_error_out {
  bool operator()(const event::detokenize_run &runtime_ev) const noexcept {
    return runtime_ev.request.error_out == nullptr;
  }
};

struct guard_has_done_callback {
  bool operator()(const event::detokenize_run &runtime_ev) const noexcept {
    return static_cast<bool>(runtime_ev.request.on_done);
  }
};

struct guard_no_done_callback {
  bool operator()(const event::detokenize_run &runtime_ev) const noexcept {
    return !static_cast<bool>(runtime_ev.request.on_done);
  }
};

struct guard_has_error_callback {
  bool operator()(const event::detokenize_run &runtime_ev) const noexcept {
    return static_cast<bool>(runtime_ev.request.on_error);
  }
};

struct guard_no_error_callback {
  bool operator()(const event::detokenize_run &runtime_ev) const noexcept {
    return !static_cast<bool>(runtime_ev.request.on_error);
  }
};

} // namespace emel::speech::tokenizer::whisper::guard
