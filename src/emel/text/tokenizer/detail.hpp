#pragma once

#include <array>
#include <cstddef>
#include <string_view>

#include "emel/model/data.hpp"
#include "emel/text/tokenizer/errors.hpp"

namespace emel::text::tokenizer::detail {

template <class runtime_event_type>
constexpr decltype(auto)
unwrap_runtime_event(const runtime_event_type &ev) noexcept {
  if constexpr (requires { ev.event_; }) {
    return (ev.event_);
  } else {
    return (ev);
  }
}

template <class value_type>
inline void write_optional(value_type *destination, value_type &sink,
                           const value_type value) noexcept {
  value_type *destinations[2] = {&sink, destination};
  value_type *const target =
      destinations[static_cast<size_t>(destination != nullptr)];
  *target = value;
}

template <class event_type>
inline bool ignore_callback(void *, const event_type &) noexcept {
  return true;
}

template <class event_type>
inline void dispatch_optional_callback(void *owner,
                                       bool (*callback)(void *owner,
                                                        const event_type &),
                                       const event_type &payload) noexcept {
  const size_t callback_ready = static_cast<size_t>(callback != nullptr);
  const size_t owner_ready = static_cast<size_t>(owner != nullptr);
  const size_t valid = callback_ready & owner_ready;
  bool (*callbacks[2])(void *, const event_type &) = {
      ignore_callback<event_type>, callback};
  void *owners[2] = {nullptr, owner};
  callbacks[valid](owners[valid], payload);
}

inline int32_t select_error_code(const bool ok,
                                 const int32_t runtime_error) noexcept {
  const std::array<int32_t, 2> fallback_errors = {
      error_code(error::backend_error), runtime_error};
  const int32_t failure_error = fallback_errors[static_cast<size_t>(
      runtime_error != error_code(error::none))];
  const std::array<int32_t, 2> final_errors = {failure_error,
                                               error_code(error::none)};
  return final_errors[static_cast<size_t>(ok)];
}

template <class request_type, class done_event_type, class error_event_type>
inline void dispatch_result_callback(
    const bool ok, const request_type &request, const done_event_type &done_ev,
    const error_event_type &error_ev,
    void (*on_done)(const request_type &, const done_event_type &,
                    const error_event_type &) noexcept,
    void (*on_error)(const request_type &, const done_event_type &,
                     const error_event_type &) noexcept) noexcept {
  using dispatch_fn_type =
      void (*)(const request_type &, const done_event_type &,
               const error_event_type &) noexcept;
  const std::array<dispatch_fn_type, 2> dispatchers = {on_error, on_done};
  dispatchers[static_cast<size_t>(ok)](request, done_ev, error_ev);
}

inline emel::model::data::tokenizer_model
tokenizer_model_from_name(const std::string_view name) noexcept {
  using tokenizer_model = emel::model::data::tokenizer_model;

  if (name == "none" || name == "no_vocab") {
    return tokenizer_model::NONE;
  }
  if (name == "llama" || name == "gemma4") {
    return tokenizer_model::SPM;
  }
  if (name == "gpt2") {
    return tokenizer_model::BPE;
  }
  if (name == "bert") {
    return tokenizer_model::WPM;
  }
  if (name == "t5") {
    return tokenizer_model::UGM;
  }
  if (name == "rwkv") {
    return tokenizer_model::RWKV;
  }
  if (name == "plamo2") {
    return tokenizer_model::PLAMO2;
  }
  return tokenizer_model::UNKNOWN;
}

inline void apply_tokenizer_model_defaults(
    const std::string_view name,
    emel::model::data::vocab & vocab) noexcept {
  if (name == "llama") {
    vocab.bos_id = 1;
    vocab.eos_id = 2;
    vocab.unk_id = 0;
    vocab.add_bos = true;
    vocab.add_space_prefix = true;
    vocab.escape_whitespaces = true;
    return;
  }

  if (name == "bert") {
    vocab.bos_id = 101;
    vocab.unk_id = 100;
    vocab.sep_id = 102;
    vocab.pad_id = 0;
    vocab.mask_id = 103;
    vocab.add_bos = true;
    vocab.add_sep = true;
    return;
  }

  if (name == "gpt2") {
    vocab.bos_id = 11;
    vocab.eos_id = 11;
    return;
  }

  if (name == "t5") {
    vocab.eos_id = 1;
    vocab.unk_id = 2;
    vocab.pad_id = 0;
    return;
  }

  if (name == "plamo2") {
    vocab.bos_id = 1;
    vocab.eos_id = 2;
    vocab.unk_id = 0;
    vocab.pad_id = 3;
  }
}

} // namespace emel::text::tokenizer::detail
