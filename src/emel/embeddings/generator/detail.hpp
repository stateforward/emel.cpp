#pragma once

#include <cmath>
#include <cstdint>
#include <span>

#include "emel/embeddings/generator/context.hpp"
#include "emel/embeddings/generator/errors.hpp"
#include "emel/embeddings/generator/events.hpp"
#include "emel/model/data.hpp"
#include "emel/text/conditioner/errors.hpp"

namespace emel::embeddings::generator::detail {

inline constexpr int32_t k_audio_input_sample_rate = 16000;
inline constexpr int32_t k_audio_input_sample_count = 4000;

template <class runtime_event_type>
constexpr decltype(auto)
unwrap_runtime_event(const runtime_event_type & ev) noexcept {
  if constexpr (requires {
                  ev.request;
                  ev.ctx;
                }) {
    return (ev);
  } else if constexpr (requires { ev.event_; }) {
    return (ev.event_);
  } else {
    return (ev);
  }
}

inline emel::error::type to_error(const error err) noexcept {
  return emel::error::cast(err);
}

inline int32_t conditioner_error_code(const emel::text::conditioner::error err) noexcept {
  return emel::text::conditioner::detail::to_local_error_code(err);
}

template <class runtime_event_type>
inline void begin_benchmark_stages(runtime_event_type & runtime_ev) noexcept {
  auto & ev = unwrap_runtime_event(runtime_ev);
  auto & timings = ev.request.benchmark_timings_out;
  timings.prepare_ns = 0u;
  timings.encode_ns = 0u;
  timings.publish_ns = 0u;
  timings.total_ns = 0u;
  ev.ctx.total_start_ns = ev.request.benchmark_time_now();
  ev.ctx.prepare_start_ns = ev.ctx.total_start_ns;
  ev.ctx.prepare_end_ns = ev.ctx.total_start_ns;
  ev.ctx.encode_start_ns = ev.ctx.total_start_ns;
  ev.ctx.encode_end_ns = ev.ctx.total_start_ns;
  ev.ctx.publish_start_ns = ev.ctx.total_start_ns;
  ev.ctx.publish_end_ns = ev.ctx.total_start_ns;
}

template <class runtime_event_type>
inline void finish_benchmark_prepare(runtime_event_type & runtime_ev) noexcept {
  auto & ev = unwrap_runtime_event(runtime_ev);
  ev.ctx.prepare_end_ns = ev.request.benchmark_time_now();
  ev.ctx.encode_start_ns = ev.ctx.prepare_end_ns;
}

template <class runtime_event_type>
inline void finish_benchmark_encode(runtime_event_type & runtime_ev) noexcept {
  auto & ev = unwrap_runtime_event(runtime_ev);
  ev.ctx.encode_end_ns = ev.request.benchmark_time_now();
  ev.ctx.publish_start_ns = ev.ctx.encode_end_ns;
}

template <class runtime_event_type>
inline void finish_benchmark_publish(runtime_event_type & runtime_ev) noexcept {
  auto & ev = unwrap_runtime_event(runtime_ev);
  auto & timings = ev.request.benchmark_timings_out;
  ev.ctx.publish_end_ns = ev.request.benchmark_time_now();
  timings.prepare_ns = ev.ctx.prepare_end_ns - ev.ctx.prepare_start_ns;
  timings.encode_ns = ev.ctx.encode_end_ns - ev.ctx.encode_start_ns;
  timings.publish_ns = ev.ctx.publish_end_ns - ev.ctx.publish_start_ns;
  timings.total_ns = ev.ctx.publish_end_ns - ev.ctx.total_start_ns;
}

inline bool is_valid_preprocessor(
    const emel::text::tokenizer::preprocessor::preprocessor_kind value) noexcept {
  switch (value) {
    case emel::text::tokenizer::preprocessor::preprocessor_kind::spm:
    case emel::text::tokenizer::preprocessor::preprocessor_kind::bpe:
    case emel::text::tokenizer::preprocessor::preprocessor_kind::wpm:
    case emel::text::tokenizer::preprocessor::preprocessor_kind::ugm:
    case emel::text::tokenizer::preprocessor::preprocessor_kind::rwkv:
    case emel::text::tokenizer::preprocessor::preprocessor_kind::plamo2:
    case emel::text::tokenizer::preprocessor::preprocessor_kind::fallback:
      return true;
    default:
      return false;
  }
}

inline bool is_valid_encoder(const emel::text::encoders::encoder_kind value) noexcept {
  switch (value) {
    case emel::text::encoders::encoder_kind::spm:
    case emel::text::encoders::encoder_kind::bpe:
    case emel::text::encoders::encoder_kind::wpm:
    case emel::text::encoders::encoder_kind::ugm:
    case emel::text::encoders::encoder_kind::rwkv:
    case emel::text::encoders::encoder_kind::plamo2:
    case emel::text::encoders::encoder_kind::fallback:
      return true;
    default:
      return false;
  }
}

inline int32_t shared_embedding_size(const action::context & ctx) noexcept {
  return ctx.execution_contract.embedding_length > 0 ? ctx.execution_contract.embedding_length
                                                     : ctx.text_status.shared_embedding_size;
}

inline int32_t requested_output_dimension(const event::embed_text & request,
                                          const action::context & ctx) noexcept {
  const int32_t full_dimension = shared_embedding_size(ctx);
  return request.truncate_dimension > 0 ? request.truncate_dimension : full_dimension;
}

inline int32_t requested_output_dimension(const event::embed_image & request,
                                          const action::context & ctx) noexcept {
  const int32_t full_dimension = shared_embedding_size(ctx);
  return request.truncate_dimension > 0 ? request.truncate_dimension : full_dimension;
}

inline int32_t requested_output_dimension(const event::embed_audio & request,
                                          const action::context & ctx) noexcept {
  const int32_t full_dimension = shared_embedding_size(ctx);
  return request.truncate_dimension > 0 ? request.truncate_dimension : full_dimension;
}

inline bool is_supported_truncate_dimension(const action::context & ctx,
                                            const int32_t dimension) noexcept {
  const int32_t full_dimension = shared_embedding_size(ctx);
  if (dimension <= 0 || full_dimension <= 0) {
    return false;
  }
  if (dimension == full_dimension) {
    return true;
  }
  if (dimension > full_dimension) {
    return false;
  }
  for (uint32_t index = 0u; index < ctx.execution_contract.matryoshka_dimension_count; ++index) {
    if (ctx.execution_contract.matryoshka_dimensions[index] == dimension) {
      return true;
    }
  }
  return false;
}

inline bool is_valid_image_payload(std::span<const uint8_t> rgba,
                                   const int32_t width,
                                   const int32_t height) noexcept {
  if (rgba.data() == nullptr || width <= 0 || height <= 0) {
    return false;
  }
  const uint64_t pixel_count = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
  const uint64_t byte_count = pixel_count * 4u;
  return byte_count == rgba.size();
}

inline bool is_valid_audio_payload(std::span<const float> pcm,
                                   const int32_t sample_rate) noexcept {
  return pcm.data() != nullptr &&
      sample_rate == k_audio_input_sample_rate &&
      pcm.size() == static_cast<size_t>(k_audio_input_sample_count);
}

inline bool has_embed_callbacks(const event::embed_text_run & runtime_ev) noexcept {
  return static_cast<bool>(runtime_ev.request.on_done);
}

inline bool has_embed_callbacks(const event::embed_image_run & runtime_ev) noexcept {
  return static_cast<bool>(runtime_ev.request.on_done);
}

inline bool has_embed_callbacks(const event::embed_audio_run & runtime_ev) noexcept {
  return static_cast<bool>(runtime_ev.request.on_done);
}

inline bool has_embed_error_callback(const event::embed_text_run & runtime_ev) noexcept {
  return static_cast<bool>(runtime_ev.request.on_error);
}

inline bool has_embed_error_callback(const event::embed_image_run & runtime_ev) noexcept {
  return static_cast<bool>(runtime_ev.request.on_error);
}

inline bool has_embed_error_callback(const event::embed_audio_run & runtime_ev) noexcept {
  return static_cast<bool>(runtime_ev.request.on_error);
}

inline bool has_initialize_callback(const event::initialize_run & runtime_ev) noexcept {
  return static_cast<bool>(runtime_ev.request.on_done);
}

inline bool has_initialize_error_callback(const event::initialize_run & runtime_ev) noexcept {
  return static_cast<bool>(runtime_ev.request.on_error);
}

inline void set_error(const event::initialize_run & runtime_ev, const error err) noexcept {
  runtime_ev.ctx.err = to_error(err);
}

inline void set_error(const event::embed_text_run & runtime_ev, const error err) noexcept {
  runtime_ev.ctx.err = to_error(err);
}

inline void set_error(const event::embed_image_run & runtime_ev, const error err) noexcept {
  runtime_ev.ctx.err = to_error(err);
}

inline void set_error(const event::embed_audio_run & runtime_ev, const error err) noexcept {
  runtime_ev.ctx.err = to_error(err);
}

inline bool l2_normalize(std::span<float> values) noexcept {
  if (values.empty()) {
    return false;
  }

  float sum = 0.0f;
  for (const float value : values) {
    sum += value * value;
  }
  if (sum <= 0.0f) {
    return false;
  }

  const float inv_norm = 1.0f / std::sqrt(sum);
  for (float & value : values) {
    value *= inv_norm;
  }
  return true;
}

inline void write_initialize_error_out(const event::initialize_run & runtime_ev) noexcept {
  if (runtime_ev.request.error_out != nullptr) {
    *runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
}

inline void write_embed_error_out(const event::embed_text_run & runtime_ev) noexcept {
  runtime_ev.request.output_dimension_out = runtime_ev.ctx.output_dimension;
  if (runtime_ev.request.error_out != nullptr) {
    *runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
}

inline void write_embed_error_out(const event::embed_image_run & runtime_ev) noexcept {
  runtime_ev.request.output_dimension_out = runtime_ev.ctx.output_dimension;
  if (runtime_ev.request.error_out != nullptr) {
    *runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
}

inline void write_embed_error_out(const event::embed_audio_run & runtime_ev) noexcept {
  runtime_ev.request.output_dimension_out = runtime_ev.ctx.output_dimension;
  if (runtime_ev.request.error_out != nullptr) {
    *runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
}

inline void emit_initialize_done(const event::initialize_run & runtime_ev) noexcept {
  runtime_ev.request.on_done(events::initialize_done{.request = &runtime_ev.request});
}

inline void emit_initialize_error(const event::initialize_run & runtime_ev) noexcept {
  runtime_ev.request.on_error(events::initialize_error{
      .request = &runtime_ev.request,
      .err = runtime_ev.ctx.err,
  });
}

inline void emit_embed_done(const event::embed_text_run & runtime_ev) noexcept {
  runtime_ev.request.on_done(events::text_embedding_done{
      .request = &runtime_ev.request,
      .output_dimension = runtime_ev.ctx.output_dimension,
  });
}

inline void emit_embed_done(const event::embed_image_run & runtime_ev) noexcept {
  runtime_ev.request.on_done(events::image_embedding_done{
      .request = &runtime_ev.request,
      .output_dimension = runtime_ev.ctx.output_dimension,
  });
}

inline void emit_embed_done(const event::embed_audio_run & runtime_ev) noexcept {
  runtime_ev.request.on_done(events::audio_embedding_done{
      .request = &runtime_ev.request,
      .output_dimension = runtime_ev.ctx.output_dimension,
  });
}

inline void emit_embed_error(const event::embed_text_run & runtime_ev) noexcept {
  runtime_ev.request.on_error(events::text_embedding_error{
      .request = &runtime_ev.request,
      .err = runtime_ev.ctx.err,
  });
}

inline void emit_embed_error(const event::embed_image_run & runtime_ev) noexcept {
  runtime_ev.request.on_error(events::image_embedding_error{
      .request = &runtime_ev.request,
      .err = runtime_ev.ctx.err,
  });
}

inline void emit_embed_error(const event::embed_audio_run & runtime_ev) noexcept {
  runtime_ev.request.on_error(events::audio_embedding_error{
      .request = &runtime_ev.request,
      .err = runtime_ev.ctx.err,
  });
}

}  // namespace emel::embeddings::generator::detail
