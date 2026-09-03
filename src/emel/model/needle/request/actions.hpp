#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>

#include "emel/model/needle/request/context.hpp"
#include "emel/model/needle/request/events.hpp"
#include "emel/text/detokenizer/errors.hpp"
#include "emel/text/tokenizer/errors.hpp"
#include "emel/text/tokenizer/needle/events.hpp"
#include "emel/text/tokenizer/needle/sm.hpp"

namespace emel::model::needle::request::action {

inline constexpr std::string_view k_im_start = "<|im_start|>";
inline constexpr std::string_view k_im_end = "<|im_end|>";
inline constexpr std::string_view k_tools_start = "<tools>";
inline constexpr std::string_view k_tools_end = "</tools>";
inline constexpr std::string_view k_think_start = "<think>";
inline constexpr std::string_view k_think_end = "</think>";
inline constexpr std::string_view k_tool_call_start = "<tool_call>";
inline constexpr std::string_view k_tool_call_end = "</tool_call>";

inline void on_tokenizer_load_done(
    const emel::text::tokenizer::needle::events::load_done &) noexcept {}
inline void on_tokenizer_load_error(
    const emel::text::tokenizer::needle::events::load_error &) noexcept {}

inline const emel::text::tokenizer::needle::event::load_done_fn
    k_tokenizer_load_done = emel::text::tokenizer::needle::event::load_done_fn::
        from<&on_tokenizer_load_done>();
inline const emel::text::tokenizer::needle::event::load_error_fn
    k_tokenizer_load_error = emel::text::tokenizer::needle::event::load_error_fn::
        from<&on_tokenizer_load_error>();

inline void reset_response_outputs(context &ctx) noexcept {
  ctx.generated_id_count = 0u;
  ctx.generated_text_size = 0u;
  ctx.normalized_envelope_size = 0u;
  ctx.prefill_nanoseconds = 0u;
  ctx.decode_nanoseconds = 0u;
  ctx.detokenize_pending.fill(0u);
}

inline void reset_outputs(context &ctx) noexcept {
  ctx.prompt_size = 0u;
  ctx.prompt_id_count = 0u;
  reset_response_outputs(ctx);
}

inline void copy_bytes(char *destination,
                       const std::string_view source) noexcept {
  for (size_t i = 0u; i < source.size(); ++i) destination[i] = source[i];
}

inline void append_bytes(std::vector<char> &destination, size_t &size,
                         const std::string_view source) noexcept {
  for (size_t i = 0u; i < source.size(); ++i) destination[size + i] = source[i];
  size += source.size();
}

inline uint32_t argmax(const std::span<const float> logits) noexcept {
  uint32_t best = 0u;
  for (uint32_t index = 1u; index < logits.size(); ++index)
    best = logits[index] > logits[best] ? index : best;
  return best;
}

inline bool append_json_string(std::vector<char> &output, size_t &size,
                               const std::string_view value) noexcept {
  static constexpr char k_hex[] = "0123456789abcdef";
  if (size + 2u > output.size()) return false;
  output[size++] = '"';
  for (const unsigned char c : value) {
    const char *escape = nullptr;
    switch (c) {
    case '"': escape = "\\\""; break;
    case '\\': escape = "\\\\"; break;
    case '\b': escape = "\\b"; break;
    case '\f': escape = "\\f"; break;
    case '\n': escape = "\\n"; break;
    case '\r': escape = "\\r"; break;
    case '\t': escape = "\\t"; break;
    default: break;
    }
    if (escape != nullptr) {
      if (size + 2u > output.size()) return false;
      output[size++] = escape[0];
      output[size++] = escape[1];
    } else if (c < 0x20u) {
      if (size + 6u > output.size()) return false;
      output[size++] = '\\'; output[size++] = 'u';
      output[size++] = '0'; output[size++] = '0';
      output[size++] = k_hex[c >> 4u]; output[size++] = k_hex[c & 0x0fu];
    } else {
      if (size == output.size()) return false;
      output[size++] = static_cast<char>(c);
    }
  }
  if (size == output.size()) return false;
  output[size++] = '"';
  return true;
}

inline bool append_json_literal(std::vector<char> &output, size_t &size,
                                const std::string_view value) noexcept {
  if (size + value.size() > output.size()) return false;
  append_bytes(output, size, value);
  return true;
}

inline bool parse_route_call(const std::string_view generated,
                             std::string_view &reasoning,
                             std::string_view &calls) noexcept {
  reasoning = {};
  calls = {};
  const size_t think_begin = generated.find(k_think_start);
  const size_t think_end = generated.find(k_think_end);
  if (think_begin != std::string_view::npos &&
      think_end != std::string_view::npos && think_end > think_begin) {
    size_t begin = think_begin + k_think_start.size();
    if (begin < generated.size() && generated[begin] == '\n') ++begin;
    size_t end = think_end;
    if (end > begin && generated[end - 1u] == '\n') --end;
    reasoning = generated.substr(begin, end - begin);
  }
  const size_t call_begin = generated.find(k_tool_call_start);
  const size_t call_end = generated.find(k_tool_call_end);
  if (call_begin == std::string_view::npos || call_end == std::string_view::npos ||
      call_end <= call_begin + k_tool_call_start.size())
    return false;
  calls = generated.substr(call_begin + k_tool_call_start.size(),
                           call_end - call_begin - k_tool_call_start.size());
  return !calls.empty() && calls.front() == '[' && calls.back() == ']';
}
template <class runtime_event>
inline const auto &origin_event(const runtime_event &ev) noexcept {
  if constexpr (requires { ev.event_; })
    return ev.event_;
  else
    return ev;
}

struct effect_begin_configure {
  void operator()(const event::configure_run &ev, context &) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
  }
};
inline bool normalize_generated_response(context &ctx,
                                         const std::string_view generated) noexcept {
  std::string_view reasoning = {};
  std::string_view calls = {};
  if (!parse_route_call(generated, reasoning, calls)) return false;
  size_t &size = ctx.normalized_envelope_size;
  size = 0u;
  auto literal = [&](const std::string_view text) noexcept {
    return append_json_literal(ctx.normalized_envelope, size, text);
  };
  return literal("{\"error\":null,\"error_code\":null,\"function_calls\":") &&
         literal(calls) && literal(",\"reason\":null,\"reasoning\":") &&
         append_json_string(ctx.normalized_envelope, size, reasoning) &&
         literal(",\"success\":true,\"type\":\"call\",\"validation\":{\"negation\":false,\"ungrounded\":[]}}");
}

struct effect_initialize_assets {
  void operator()(const event::configure_run &ev, context &ctx) const noexcept {
    const auto blob = std::span<const uint8_t>{
        ctx.bound.tokenizer_blob.data,
        static_cast<size_t>(ctx.bound.tokenizer_blob.nbytes)};
    emel::text::tokenizer::needle::sm loader{};
    const bool loaded = loader.process_event(
        emel::text::tokenizer::needle::event::load{
            blob, *ctx.vocab, k_tokenizer_load_done, k_tokenizer_load_error});
    int32_t tokenizer_error =
        emel::text::tokenizer::error_code(emel::text::tokenizer::error::none);
    emel::text::tokenizer::event::bind tokenizer_bind{};
    tokenizer_bind.vocab = ctx.vocab.get();
    tokenizer_bind.preprocessor_variant =
        emel::text::tokenizer::preprocessor::preprocessor_kind::spm;
    tokenizer_bind.encoder_variant = emel::text::encoders::encoder_kind::spm;
    tokenizer_bind.error_out = &tokenizer_error;
    const bool tokenizer_bound = loaded && ctx.tokenizer->process_event(tokenizer_bind);
    int32_t detokenizer_error = emel::text::detokenizer::error_code(
        emel::text::detokenizer::error::none);
    const bool detokenizer_bound =
        tokenizer_bound && ctx.detokenizer->process_event(
                               emel::text::detokenizer::event::bind{
                                   *ctx.vocab, detokenizer_error});
    ctx.assets_ready =
        detokenizer_bound && ctx.vocab->bos_id >= 0 && ctx.vocab->eos_id >= 0;
    ev.ctx.err = ctx.assets_ready ? emel::error::cast(error::none)
                                  : emel::error::cast(error::not_initialized);
  }
};

struct effect_store_configuration {
  void operator()(const event::configure_run &ev, context &ctx) const noexcept {
    copy_bytes(ctx.system_storage.data(), ev.request.system);
    copy_bytes(ctx.tools_storage.data(), ev.request.tools_json);
    ctx.system_size = ev.request.system.size();
    ctx.tools_size = ev.request.tools_json.size();
    ctx.configured = true;
    ctx.reset_ready = false;
  }
};

struct effect_begin_reset {
  void operator()(const event::reset_run &ev, context &ctx) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    reset_outputs(ctx);
  }
};

struct effect_exec_reset {
  void operator()(const event::reset_run &ev, context &ctx) const noexcept {
    ctx.reset_ready = ctx.graph->process_event(
        needle::graph::event::init{.activation_quant = false});
    ev.ctx.err = ctx.reset_ready ? emel::error::cast(error::none)
                                 : emel::error::cast(error::graph_rejected);
  }
};

struct effect_begin_complete {
  void operator()(const event::complete_run &ev, context &ctx) const noexcept {
    ev.ctx.err = emel::error::cast(error::none);
    reset_response_outputs(ctx);
    ev.ctx.timestamp_now = ctx.timestamp_now;
  }
};

struct effect_render_prompt {
  template <class runtime_event>
  void operator()(const runtime_event &ev, context &ctx) const noexcept {
    const auto &run = origin_event(ev);
    ctx.prompt_size = 0u;
    ctx.prompt_id_count = 0u;
    size_t required = k_im_start.size() + 5u + k_tools_start.size() +
                      ctx.tools_size + k_tools_end.size() + 1u +
                      run.request.query.size() + k_im_end.size() + 1u +
                      k_im_start.size() + 10u;
    if (ctx.system_size != 0u)
      required += k_im_start.size() + 7u + ctx.system_size + k_im_end.size() + 1u;
    if (required > ctx.prompt_storage.size()) {
      run.ctx.err = emel::error::cast(error::capacity_exceeded);
      return;
    }
    if (ctx.system_size != 0u) {
      append_bytes(ctx.prompt_storage, ctx.prompt_size, k_im_start);
      append_bytes(ctx.prompt_storage, ctx.prompt_size, "system\n");
      append_bytes(ctx.prompt_storage, ctx.prompt_size,
                   {ctx.system_storage.data(), ctx.system_size});
      append_bytes(ctx.prompt_storage, ctx.prompt_size, k_im_end);
      append_bytes(ctx.prompt_storage, ctx.prompt_size, "\n");
    }
    append_bytes(ctx.prompt_storage, ctx.prompt_size, k_im_start);
    append_bytes(ctx.prompt_storage, ctx.prompt_size, "user\n");
    append_bytes(ctx.prompt_storage, ctx.prompt_size, k_tools_start);
    append_bytes(ctx.prompt_storage, ctx.prompt_size,
                 {ctx.tools_storage.data(), ctx.tools_size});
    append_bytes(ctx.prompt_storage, ctx.prompt_size, k_tools_end);
    append_bytes(ctx.prompt_storage, ctx.prompt_size, "\n");
    append_bytes(ctx.prompt_storage, ctx.prompt_size, run.request.query);
    append_bytes(ctx.prompt_storage, ctx.prompt_size, k_im_end);
    append_bytes(ctx.prompt_storage, ctx.prompt_size, "\n");
    append_bytes(ctx.prompt_storage, ctx.prompt_size, k_im_start);
    append_bytes(ctx.prompt_storage, ctx.prompt_size, "assistant\n");
  }
};

struct effect_tokenize_prompt {
  template <class runtime_event>
  void operator()(const runtime_event &ev, context &ctx) const noexcept {
    const auto &run = origin_event(ev);
    int32_t token_count = 0;
    int32_t tokenizer_error =
        emel::text::tokenizer::error_code(emel::text::tokenizer::error::none);
    emel::text::tokenizer::event::tokenize tokenize{};
    tokenize.vocab = ctx.vocab.get();
    tokenize.text = {ctx.prompt_storage.data(), ctx.prompt_size};
    tokenize.add_special = false;
    tokenize.parse_special = true;
    tokenize.token_ids_out = ctx.prompt_ids.data() + 1u;
    tokenize.token_capacity = static_cast<int32_t>(ctx.prompt_ids.size() - 1u);
    tokenize.token_count_out = &token_count;
    tokenize.error_out = &tokenizer_error;
    const bool accepted = ctx.tokenizer->process_event(tokenize);
    if (!accepted || token_count < 0 ||
        static_cast<size_t>(token_count) + 1u + run.request.max_new_tokens >
            ctx.prompt_ids.size() ||
        static_cast<uint64_t>(token_count) + 1u + run.request.max_new_tokens >
            ctx.bound.geo.max_seq_len) {
      run.ctx.err = emel::error::cast(accepted ? error::capacity_exceeded
                                               : error::tokenizer_rejected);
      return;
    }
    ctx.prompt_ids[0] = ctx.vocab->bos_id;
    ctx.prompt_id_count = static_cast<size_t>(token_count) + 1u;
  }
};

struct effect_prefill {
  void operator()(const event::complete_run &ev, context &ctx) const noexcept {
    const uint64_t begin = ev.ctx.timestamp_now();
    const bool accepted = ctx.graph->process_event(needle::graph::event::prefill{
        {ctx.prompt_ids.data(), ctx.prompt_id_count}, ctx.logits});
    const uint64_t end = ev.ctx.timestamp_now();
    ctx.prefill_nanoseconds = end - begin;
    ev.ctx.err = accepted ? emel::error::cast(error::none)
                           : emel::error::cast(error::graph_rejected);
    if (accepted)
      ctx.generated_ids[0] =
          static_cast<int32_t>(argmax(std::span<const float>{ctx.logits}));
  }
};

struct effect_decode_token {
  void operator()(const event::complete_run &ev, context &ctx) const noexcept {
    const int32_t token = ctx.generated_ids[ctx.generated_id_count];
    ++ctx.generated_id_count;
    const uint64_t begin = ev.ctx.timestamp_now();
    const bool accepted = ctx.graph->process_event(
        needle::graph::event::decode{token, ctx.logits});
    const uint64_t end = ev.ctx.timestamp_now();
    ctx.decode_nanoseconds += end - begin;
    ev.ctx.err = accepted ? emel::error::cast(error::none)
                           : emel::error::cast(error::graph_rejected);
    if (accepted)
      ctx.generated_ids[ctx.generated_id_count] =
          static_cast<int32_t>(argmax(std::span<const float>{ctx.logits}));
  }
};

struct effect_finish_generation {
  void operator()(const event::complete_run &, context &ctx) const noexcept {
    if (ctx.generated_id_count < ctx.generated_ids.size() &&
        ctx.generated_ids[ctx.generated_id_count] != ctx.vocab->eos_id)
      ++ctx.generated_id_count;
  }
};

struct effect_detokenize_generation {
  void operator()(const event::complete_run &ev, context &ctx) const noexcept {
    size_t pending_length = 0u;
    for (size_t i = 0u; i < ctx.generated_id_count; ++i) {
      size_t piece_length = 0u;
      size_t next_pending = pending_length;
      int32_t detokenizer_error = emel::text::detokenizer::error_code(
          emel::text::detokenizer::error::none);
      const bool accepted = ctx.detokenizer->process_event(
          emel::text::detokenizer::event::detokenize{
              ctx.generated_ids[i], true, ctx.detokenize_pending.data(),
              pending_length, ctx.detokenize_pending.size(),
              ctx.detokenize_piece.data(), ctx.detokenize_piece.size(),
              piece_length, next_pending, detokenizer_error});
      if (!accepted ||
          ctx.generated_text_size + piece_length > ctx.generated_text.size()) {
        ev.ctx.err = emel::error::cast(accepted ? error::capacity_exceeded
                                                 : error::detokenizer_rejected);
        return;
      }
      for (size_t byte = 0u; byte < piece_length; ++byte) {
        const char value = ctx.detokenize_piece[byte];
        if (value == '\xE2' && byte + 2u < piece_length &&
            static_cast<unsigned char>(ctx.detokenize_piece[byte + 1u]) == 0x96u &&
            static_cast<unsigned char>(ctx.detokenize_piece[byte + 2u]) == 0x81u) {
          ctx.generated_text[ctx.generated_text_size++] = ' ';
          byte += 2u;
        } else {
          ctx.generated_text[ctx.generated_text_size++] = value;
        }
      }
      pending_length = next_pending;
    }
  }
};

struct effect_normalize_response {
  void operator()(const event::complete_run &ev, context &ctx) const noexcept {
    const std::string_view generated{ctx.generated_text.data(),
                                     ctx.generated_text_size};
    const bool ok = normalize_generated_response(ctx, generated);
    ev.ctx.err = ok ? emel::error::cast(error::none)
                    : emel::error::cast(error::response_invalid);
  }
};

struct effect_publish_configured {
  void operator()(const event::configure_run &ev, context &) const noexcept {
    ev.request.on_done(events::configured{ev.request});
  }
};

struct effect_publish_reset_done {
  void operator()(const event::reset_run &ev, context &) const noexcept {
    ev.request.on_done(events::reset_done{ev.request});
  }
};

struct effect_publish_completed {
  void operator()(const event::complete_run &ev, context &ctx) const noexcept {
    ev.request.on_done(events::completed{
        .request = ev.request,
        .normalized_envelope = {ctx.normalized_envelope.data(),
                                ctx.normalized_envelope_size},
        .generated_token_ids = {ctx.generated_ids.data(),
                                ctx.generated_id_count},
        .prompt_tokens = static_cast<uint32_t>(ctx.prompt_id_count),
        .generated_tokens = static_cast<uint32_t>(ctx.generated_id_count),
        .prefill_nanoseconds = ctx.prefill_nanoseconds,
        .decode_nanoseconds = ctx.decode_nanoseconds,
    });
  }
};

struct effect_publish_error {
  template <class runtime_event>
  void operator()(const runtime_event &ev, context &) const noexcept {
    ev.request.on_error(events::request_error{ev.ctx.err});
  }
};

struct effect_mark_invalid {
  template <class runtime_event>
  void operator()(const runtime_event &ev, context &) const noexcept {
    if constexpr (requires { ev.event_.ctx.err; })
      ev.event_.ctx.err = emel::error::cast(error::invalid_request);
    else
      ev.ctx.err = emel::error::cast(error::invalid_request);
  }
};

struct effect_on_unexpected {
  template <class runtime_event>
  void operator()(const runtime_event &ev, context &) const noexcept {
    if constexpr (requires { ev.event_.ctx.err; })
      ev.event_.ctx.err = emel::error::cast(error::internal_error);
  }
};

} // namespace emel::model::needle::request::action
