#pragma once

#include <cstddef>

#include "emel/speech/tokenizer/whisper/detail.hpp"
#include "emel/speech/tokenizer/whisper/events.hpp"

namespace emel::speech::tokenizer::whisper::guard {

struct guard_tokenizer_json_valid {
  bool operator()(const event::detokenize_run &runtime_ev) const noexcept {
    return runtime_ev.request.tokenizer_json.data() != nullptr &&
           !runtime_ev.request.tokenizer_json.empty() &&
           detail::validate_tiny_control_tokens(
               runtime_ev.request.tokenizer_json);
  }
};

struct guard_tokenizer_json_invalid {
  bool operator()(const event::detokenize_run &runtime_ev) const noexcept {
    return !guard_tokenizer_json_valid{}(runtime_ev);
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
