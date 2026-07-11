#pragma once

#include <algorithm>
#include <array>
#include <cstring>
#include <span>

#include "emel/kernel/detail.hpp"
#include "emel/memory/events.hpp"
#include "emel/model/moshi/detail.hpp"
#include "emel/speech/predictor/moshi/context.hpp"
#include "emel/speech/predictor/moshi/detail.hpp"
#include "emel/speech/predictor/moshi/events.hpp"

namespace emel::speech::predictor::moshi::action {

namespace detail_ns {

inline emel::error::type to_error(const error value) noexcept {
  return emel::error::cast(value);
}

} // namespace detail_ns

template <class runtime_event_type> struct effect_mark_not_initialized {
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail_ns::to_error(error::not_initialized);
  }
};

template <class runtime_event_type>
struct effect_mark_not_initialized_and_store {
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail_ns::to_error(error::not_initialized);
    *runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

struct effect_mark_bind_failed {
  void operator()(const event::initialize_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail_ns::to_error(error::bind_failed);
  }
};

template <class runtime_event_type> struct effect_mark_memory_error {
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail_ns::to_error(error::memory);
  }
};

struct effect_mark_step_request_invalid {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    ev.ctx.err = detail_ns::to_error(error::request_shape);
  }
};

template <class runtime_event_type> struct effect_mark_graph_runtime_error {
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail_ns::to_error(error::graph_runtime);
  }
};

template <class runtime_event_type> struct effect_mark_voice_contract_error {
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail_ns::to_error(error::voice_contract);
  }
};

template <class runtime_event_type>
struct effect_mark_personaplex_prompt_error {
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail_ns::to_error(error::personaplex_prompt);
  }
};

struct effect_mark_voice_prompt_pending {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    ev.ctx.err = detail_ns::to_error(error::voice_prompt_pending);
  }
};

struct effect_bind_contract {
  void operator()(const event::initialize_run &runtime_ev,
                  context &ctx) const noexcept {
    ctx.session = {};
    ctx.session.model = &runtime_ev.request.model;
    ctx.session.sequence_id = runtime_ev.request.sequence_id;
    ctx.session.n_q = runtime_ev.request.model.moshi_lm.n_q;
    ctx.session.dep_q = runtime_ev.request.model.moshi_lm.dep_q;
    ctx.session.text_card = runtime_ev.request.model.moshi_lm.text_card;
    ctx.session.audio_card = runtime_ev.request.model.moshi_lm.card;
    (void)emel::model::moshi::detail::build_execution_contract(
        runtime_ev.request.model, ctx.session.contract);

    const auto &lm = runtime_ev.request.model.moshi_lm;
    ctx.lmgen = {};
    ctx.lmgen.codebook_count = lm.n_q + 1;
    ctx.lmgen.generated_dep_q = lm.dep_q;
    ctx.lmgen.cache_row_count = 2;
    std::fill(ctx.lmgen.cache.begin(), ctx.lmgen.cache.end(),
              k_token_ungenerated);
    std::fill(ctx.lmgen.initial.begin(), ctx.lmgen.initial.end(), lm.card);
    ctx.lmgen.initial[0] = lm.text_card;
    for (uint32_t index = 0; index < lm.delay_count; ++index) {
      ctx.lmgen.delays[index] = lm.delays[index];
      ctx.lmgen.max_delay = std::max(ctx.lmgen.max_delay, lm.delays[index]);
    }
    ctx.lmgen.cache_row_count = ctx.lmgen.max_delay + 2 + ctx.lmgen.delay_steps;
  }
};

struct effect_configure_standard_lmgen {
  void operator()(const event::initialize_run &, context &ctx) const noexcept {
    ctx.lmgen.delayed_dep_q = ctx.lmgen.generated_dep_q;
    ctx.lmgen.needed_tokens =
        ctx.lmgen.codebook_count - ctx.lmgen.delayed_dep_q - 1;
  }
};

template <class graph_actor_type> struct effect_initialize_graph {
  void operator()(const event::initialize_run &runtime_ev, context &,
                  graph_actor_type &graph) const noexcept {
    runtime_ev.ctx.graph_error = detail_ns::to_error(error::none);
    event::initialize_graph request{runtime_ev.request};
    request.error_out = &runtime_ev.ctx.graph_error;
    runtime_ev.ctx.graph_accepted = graph.process_event(request);
  }
};

struct effect_configure_personaplex_lmgen {
  void operator()(const event::initialize_run &, context &ctx) const noexcept {
    ctx.lmgen.delayed_dep_q = ctx.session.model->moshi_lm.inference_dep_q;
    ctx.lmgen.needed_tokens =
        ctx.lmgen.codebook_count - ctx.lmgen.delayed_dep_q - 1;
    ctx.lmgen.cache_row_count = ctx.lmgen.max_delay + 3;
  }
};

struct effect_bind_voice_contract {
  void operator()(const event::load_voice_run &runtime_ev,
                  context &ctx) const noexcept {
    ctx.voice = {};
    ctx.voice.model = &runtime_ev.request.voice_model;
    (void)emel::model::moshi::detail::build_execution_contract(
        runtime_ev.request.voice_model, ctx.voice.contract);
    const auto *embeddings = ctx.voice.contract.voice.embeddings.tensor;
    const auto *cache = ctx.voice.contract.voice.cache.tensor;
    ctx.voice.embedding_dim = static_cast<int32_t>(embeddings->dims[0]);
    ctx.voice.prompt_frame_count = static_cast<int32_t>(embeddings->dims[3]);
    ctx.voice.prompt_frame_index = 0;
    ctx.voice.cache_row_count = static_cast<int32_t>(cache->dims[0]);
    ctx.voice.cache_column_count = static_cast<int32_t>(cache->dims[1]);
    ctx.voice.loaded = true;
    ctx.voice.ready = false;
  }
};

struct effect_reserve_memory {
  void operator()(const event::initialize_run &runtime_ev,
                  context &ctx) const noexcept {
    runtime_ev.ctx.memory_error = static_cast<int32_t>(
        emel::error::cast(emel::memory::hybrid::error::none));
    runtime_ev.ctx.memory_accepted =
        ctx.memory.process_event(emel::memory::event::reserve{
            .max_sequences = runtime_ev.request.max_sequences,
            .max_blocks = runtime_ev.request.max_blocks,
            .block_tokens = runtime_ev.request.block_tokens,
            .error_out = &runtime_ev.ctx.memory_error,
        });
  }
};

struct effect_allocate_sequence {
  void operator()(const event::initialize_run &runtime_ev,
                  context &ctx) const noexcept {
    runtime_ev.ctx.memory_error = static_cast<int32_t>(
        emel::error::cast(emel::memory::hybrid::error::none));
    runtime_ev.ctx.memory_accepted =
        ctx.memory.process_event(emel::memory::event::allocate_sequence{
            .seq_id = runtime_ev.request.sequence_id,
            .error_out = &runtime_ev.ctx.memory_error,
        });
  }
};

struct effect_allocate_step_slot {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &ctx) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    ev.ctx.memory_error = static_cast<int32_t>(
        emel::error::cast(emel::memory::hybrid::error::none));
    ev.ctx.memory_block_count = 0;
    ev.ctx.memory_accepted =
        ctx.memory.process_event(emel::memory::event::allocate_slots{
            .seq_id = ctx.session.sequence_id,
            .token_count = 1,
            .block_count_out = &ev.ctx.memory_block_count,
            .error_out = &ev.ctx.memory_error,
            .copy_block = nullptr,
            .copy_block_user_data = nullptr,
        });
  }
};

struct effect_capture_memory {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &ctx) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    ev.ctx.memory_error = static_cast<int32_t>(
        emel::error::cast(emel::memory::hybrid::error::none));
    ev.ctx.memory_accepted =
        ctx.memory.process_event(emel::memory::event::capture_view{
            .snapshot_out = &ev.ctx.memory_snapshot,
            .error_out = &ev.ctx.memory_error,
        });
  }
};

struct effect_allocate_voice_prefill_slot {
  void operator()(const event::prefill_voice_run &runtime_ev,
                  context &ctx) const noexcept {
    runtime_ev.ctx.memory_error = static_cast<int32_t>(
        emel::error::cast(emel::memory::hybrid::error::none));
    runtime_ev.ctx.memory_block_count = 0;
    runtime_ev.ctx.memory_accepted =
        ctx.memory.process_event(emel::memory::event::allocate_slots{
            .seq_id = ctx.session.sequence_id,
            .token_count = 1,
            .block_count_out = &runtime_ev.ctx.memory_block_count,
            .error_out = &runtime_ev.ctx.memory_error,
            .copy_block = nullptr,
            .copy_block_user_data = nullptr,
        });
  }
};

struct effect_capture_voice_prefill_memory {
  void operator()(const event::prefill_voice_run &runtime_ev,
                  context &ctx) const noexcept {
    runtime_ev.ctx.memory_error = static_cast<int32_t>(
        emel::error::cast(emel::memory::hybrid::error::none));
    runtime_ev.ctx.memory_accepted =
        ctx.memory.process_event(emel::memory::event::capture_view{
            .snapshot_out = &runtime_ev.ctx.memory_snapshot,
            .error_out = &runtime_ev.ctx.memory_error,
        });
  }
};

struct effect_begin_voice_prefill {
  void operator()(const event::prefill_voice_run &runtime_ev,
                  const context &ctx) const noexcept {
    runtime_ev.ctx.err = detail_ns::to_error(error::none);
    runtime_ev.ctx.graph_error = detail_ns::to_error(error::none);
    runtime_ev.ctx.graph_accepted = false;
    runtime_ev.ctx.embedding_frame_ok = false;
    runtime_ev.ctx.complete = ctx.voice.ready;
    runtime_ev.ctx.remaining_frames =
        ctx.voice.prompt_frame_count - ctx.voice.prompt_frame_index;
    runtime_ev.ctx.text_token = ctx.session.model->moshi_lm.text_padding_id;
    std::fill(runtime_ev.ctx.input_sequence.begin(),
              runtime_ev.ctx.input_sequence.end(), k_token_zero);
    runtime_ev.ctx.input_sequence[0] =
        ctx.session.model->moshi_lm.text_padding_id;
    std::fill(runtime_ev.ctx.audio_tokens.begin(),
              runtime_ev.ctx.audio_tokens.end(), k_token_zero);
  }
};

struct effect_load_voice_embedding_frame_f32 {
  void operator()(const event::prefill_voice_run &runtime_ev,
                  const context &ctx) const noexcept {
    const auto *tensor = ctx.voice.contract.voice.embeddings.tensor;
    const auto *src = static_cast<const float *>(tensor->data);
    const int32_t dim = ctx.voice.embedding_dim;
    const size_t offset = static_cast<size_t>(ctx.voice.prompt_frame_index) *
                          static_cast<size_t>(dim);
    for (int32_t index = 0; index < dim; ++index) {
      runtime_ev.ctx.embedding_frame[static_cast<size_t>(index)] =
          src[offset + static_cast<size_t>(index)];
    }
    runtime_ev.ctx.embedding_frame_ok = true;
  }
};

struct effect_load_voice_embedding_frame_f16 {
  void operator()(const event::prefill_voice_run &runtime_ev,
                  const context &ctx) const noexcept {
    const auto *tensor = ctx.voice.contract.voice.embeddings.tensor;
    const auto *src = static_cast<const uint16_t *>(tensor->data);
    const int32_t dim = ctx.voice.embedding_dim;
    const size_t offset = static_cast<size_t>(ctx.voice.prompt_frame_index) *
                          static_cast<size_t>(dim);
    for (int32_t index = 0; index < dim; ++index) {
      runtime_ev.ctx.embedding_frame[static_cast<size_t>(index)] =
          emel::kernel::detail::quant::fp16_to_fp32(
              src[offset + static_cast<size_t>(index)]);
    }
    runtime_ev.ctx.embedding_frame_ok = true;
  }
};

struct effect_load_voice_embedding_frame_bf16 {
  void operator()(const event::prefill_voice_run &runtime_ev,
                  const context &ctx) const noexcept {
    const auto *tensor = ctx.voice.contract.voice.embeddings.tensor;
    const auto *src = static_cast<const uint16_t *>(tensor->data);
    const int32_t dim = ctx.voice.embedding_dim;
    const size_t offset = static_cast<size_t>(ctx.voice.prompt_frame_index) *
                          static_cast<size_t>(dim);
    for (int32_t index = 0; index < dim; ++index) {
      runtime_ev.ctx.embedding_frame[static_cast<size_t>(index)] =
          emel::kernel::detail::bf16_to_fp32(
              src[offset + static_cast<size_t>(index)]);
    }
    runtime_ev.ctx.embedding_frame_ok = true;
  }
};

template <class graph_actor_type> struct effect_run_voice_graph_runtime {
  void operator()(const event::prefill_voice_run &runtime_ev, context &ctx,
                  graph_actor_type &graph) const noexcept {
    runtime_ev.ctx.graph_error = detail_ns::to_error(error::none);
    event::graph_step graph_step{
        *ctx.session.model,
        runtime_ev.ctx.memory_snapshot,
        std::span<const int32_t>{runtime_ev.ctx.input_sequence.data(),
                                 static_cast<size_t>(ctx.lmgen.codebook_count)},
        std::span<int32_t>{runtime_ev.ctx.audio_tokens.data(),
                           static_cast<size_t>(ctx.lmgen.generated_dep_q)},
        runtime_ev.ctx.text_token,
    };
    graph_step.sequence_id = ctx.session.sequence_id;
    graph_step.input_embedding =
        std::span<const float>{runtime_ev.ctx.embedding_frame.data(),
                               static_cast<size_t>(ctx.voice.embedding_dim)};
    graph_step.forced_text_token = ctx.session.model->moshi_lm.text_padding_id;
    graph_step.error_out = &runtime_ev.ctx.graph_error;
    runtime_ev.ctx.graph_accepted = graph.process_event(graph_step);
  }
};

struct effect_advance_voice_prefill {
  void operator()(const event::prefill_voice_run &runtime_ev,
                  context &ctx) const noexcept {
    ++ctx.voice.prompt_frame_index;
    ++ctx.lmgen.offset;
    runtime_ev.ctx.remaining_frames =
        ctx.voice.prompt_frame_count - ctx.voice.prompt_frame_index;
    runtime_ev.ctx.complete =
        ctx.voice.prompt_frame_index >= ctx.voice.prompt_frame_count;
  }
};

struct effect_copy_voice_cache {
  void operator()(const event::prefill_voice_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto *tensor = ctx.voice.contract.voice.cache.tensor;
    const auto *src = static_cast<const int32_t *>(tensor->data);
    for (int32_t row = 0; row < ctx.voice.cache_row_count; ++row) {
      for (int32_t column = 0; column < ctx.voice.cache_column_count;
           ++column) {
        const size_t source_index =
            static_cast<size_t>(row) +
            static_cast<size_t>(column) *
                static_cast<size_t>(ctx.voice.cache_row_count);
        detail::cache_at(ctx.lmgen, row, column) = src[source_index];
      }
    }
    ctx.voice.ready = true;
    runtime_ev.ctx.complete = true;
    runtime_ev.ctx.remaining_frames = 0;
  }
};

struct effect_bind_personaplex_prompt {
  void operator()(const event::begin_personaplex_prompt_run &runtime_ev,
                  context &ctx) const noexcept {
    ctx.voice.pre_text_silence_remaining =
        runtime_ev.request.pre_text_silence_frames;
    ctx.voice.text_tokens_remaining = runtime_ev.request.text_token_count;
    ctx.voice.post_text_silence_remaining =
        runtime_ev.request.post_text_silence_frames;
    ctx.voice.prompt_started = true;
    ctx.voice.prompt_ready = false;
    runtime_ev.ctx.remaining_frames = ctx.voice.pre_text_silence_remaining +
                                      ctx.voice.text_tokens_remaining +
                                      ctx.voice.post_text_silence_remaining;
  }
};

struct effect_bind_empty_personaplex_prompt {
  void operator()(const event::begin_personaplex_prompt_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto &lm = ctx.session.model->moshi_lm;
    ctx.voice.pre_text_silence_remaining = lm.inference_pre_text_silence_frames;
    ctx.voice.text_tokens_remaining = 0;
    ctx.voice.post_text_silence_remaining =
        lm.inference_post_text_silence_frames;
    ctx.voice.prompt_started = true;
    ctx.voice.prompt_ready = false;
    runtime_ev.ctx.remaining_frames = ctx.voice.pre_text_silence_remaining +
                                      ctx.voice.post_text_silence_remaining;
  }
};

struct effect_allocate_personaplex_prompt_slot {
  void operator()(const event::prefill_personaplex_prompt_run &runtime_ev,
                  context &ctx) const noexcept {
    runtime_ev.ctx.memory_error = static_cast<int32_t>(
        emel::error::cast(emel::memory::hybrid::error::none));
    runtime_ev.ctx.memory_block_count = 0;
    runtime_ev.ctx.memory_accepted =
        ctx.memory.process_event(emel::memory::event::allocate_slots{
            .seq_id = ctx.session.sequence_id,
            .token_count = 1,
            .block_count_out = &runtime_ev.ctx.memory_block_count,
            .error_out = &runtime_ev.ctx.memory_error,
            .copy_block = nullptr,
            .copy_block_user_data = nullptr,
        });
  }
};

struct effect_capture_personaplex_prompt_memory {
  void operator()(const event::prefill_personaplex_prompt_run &runtime_ev,
                  context &ctx) const noexcept {
    runtime_ev.ctx.memory_error = static_cast<int32_t>(
        emel::error::cast(emel::memory::hybrid::error::none));
    runtime_ev.ctx.memory_accepted =
        ctx.memory.process_event(emel::memory::event::capture_view{
            .snapshot_out = &runtime_ev.ctx.memory_snapshot,
            .error_out = &runtime_ev.ctx.memory_error,
        });
  }
};

struct effect_begin_personaplex_prompt_prefill {
  void operator()(const event::prefill_personaplex_prompt_run &runtime_ev,
                  const context &ctx) const noexcept {
    runtime_ev.ctx.err = detail_ns::to_error(error::none);
    runtime_ev.ctx.graph_error = detail_ns::to_error(error::none);
    runtime_ev.ctx.graph_accepted = false;
    runtime_ev.ctx.complete = false;
    runtime_ev.ctx.remaining_frames = ctx.voice.pre_text_silence_remaining +
                                      ctx.voice.text_tokens_remaining +
                                      ctx.voice.post_text_silence_remaining;
    runtime_ev.ctx.text_token = ctx.session.model->moshi_lm.text_padding_id;
    std::fill(runtime_ev.ctx.prompt_sequence.begin(),
              runtime_ev.ctx.prompt_sequence.end(), k_token_zero);
    std::fill(runtime_ev.ctx.input_sequence.begin(),
              runtime_ev.ctx.input_sequence.end(), k_token_zero);
    std::fill(runtime_ev.ctx.audio_tokens.begin(),
              runtime_ev.ctx.audio_tokens.end(), k_token_zero);
  }
};

struct effect_build_personaplex_prompt_silence_frame {
  void operator()(const event::prefill_personaplex_prompt_run &runtime_ev,
                  const context &ctx) const noexcept {
    for (int32_t index = 0; index < ctx.lmgen.codebook_count; ++index) {
      runtime_ev.ctx.prompt_sequence[static_cast<size_t>(index)] =
          ctx.session.model->moshi_lm
              .inference_prompt_tokens[static_cast<size_t>(index)];
    }
  }
};

struct effect_build_personaplex_prompt_text_frame {
  void operator()(const event::prefill_personaplex_prompt_run &runtime_ev,
                  const context &ctx) const noexcept {
    effect_build_personaplex_prompt_silence_frame{}(runtime_ev, ctx);
    runtime_ev.ctx.prompt_sequence[0] = runtime_ev.request.text_token;
  }
};

struct effect_write_and_build_personaplex_prompt_input {
  void operator()(const event::prefill_personaplex_prompt_run &runtime_ev,
                  context &ctx) const noexcept {
    for (int32_t index = 0; index < ctx.lmgen.codebook_count; ++index) {
      const int32_t write_row = detail::cache_position(
          ctx.lmgen, ctx.lmgen.offset + ctx.lmgen.delays[index]);
      detail::cache_at(ctx.lmgen, write_row, index) =
          runtime_ev.ctx.prompt_sequence[static_cast<size_t>(index)];
    }

    const int32_t read_row =
        detail::cache_position(ctx.lmgen, ctx.lmgen.offset);
    for (int32_t index = 0; index < ctx.lmgen.codebook_count; ++index) {
      const bool use_initial = ctx.lmgen.offset <= ctx.lmgen.delays[index];
      const int32_t initial_lane = static_cast<int32_t>(use_initial);
      const int32_t cache_lane = static_cast<int32_t>(!use_initial);
      runtime_ev.ctx.input_sequence[static_cast<size_t>(index)] =
          (initial_lane * ctx.lmgen.initial[index]) +
          (cache_lane * detail::cache_at(ctx.lmgen, read_row, index));
    }
  }
};

template <class graph_actor_type>
struct effect_run_personaplex_prompt_graph_runtime {
  void operator()(const event::prefill_personaplex_prompt_run &runtime_ev,
                  context &ctx, graph_actor_type &graph) const noexcept {
    runtime_ev.ctx.graph_error = detail_ns::to_error(error::none);
    event::graph_step graph_step{
        *ctx.session.model,
        runtime_ev.ctx.memory_snapshot,
        std::span<const int32_t>{runtime_ev.ctx.input_sequence.data(),
                                 static_cast<size_t>(ctx.lmgen.codebook_count)},
        std::span<int32_t>{runtime_ev.ctx.audio_tokens.data(),
                           static_cast<size_t>(ctx.lmgen.generated_dep_q)},
        runtime_ev.ctx.text_token,
    };
    graph_step.sequence_id = ctx.session.sequence_id;
    graph_step.error_out = &runtime_ev.ctx.graph_error;
    runtime_ev.ctx.graph_accepted = graph.process_event(graph_step);
  }
};

struct effect_advance_personaplex_prompt_pre_silence {
  void operator()(const event::prefill_personaplex_prompt_run &,
                  context &ctx) const noexcept {
    --ctx.voice.pre_text_silence_remaining;
    ++ctx.lmgen.offset;
  }
};

struct effect_advance_personaplex_prompt_text {
  void operator()(const event::prefill_personaplex_prompt_run &,
                  context &ctx) const noexcept {
    --ctx.voice.text_tokens_remaining;
    ++ctx.lmgen.offset;
  }
};

struct effect_advance_personaplex_prompt_post_silence {
  void operator()(const event::prefill_personaplex_prompt_run &,
                  context &ctx) const noexcept {
    --ctx.voice.post_text_silence_remaining;
    ++ctx.lmgen.offset;
  }
};

struct effect_finish_personaplex_prompt {
  void operator()(const event::prefill_personaplex_prompt_run &runtime_ev,
                  context &ctx) const noexcept {
    ctx.voice.prompt_ready = true;
    runtime_ev.ctx.complete = true;
    runtime_ev.ctx.remaining_frames = 0;
  }
};

struct effect_publish_personaplex_prompt_pending {
  void operator()(const event::prefill_personaplex_prompt_run &runtime_ev,
                  const context &ctx) const noexcept {
    runtime_ev.ctx.complete = false;
    runtime_ev.ctx.remaining_frames = ctx.voice.pre_text_silence_remaining +
                                      ctx.voice.text_tokens_remaining +
                                      ctx.voice.post_text_silence_remaining;
  }
};

struct effect_store_personaplex_prompt_complete_out {
  void operator()(const event::prefill_personaplex_prompt_run &runtime_ev,
                  context &) const noexcept {
    *runtime_ev.request.complete_out = runtime_ev.ctx.complete;
  }
};

struct effect_store_personaplex_prompt_remaining_out {
  void operator()(const event::prefill_personaplex_prompt_run &runtime_ev,
                  context &) const noexcept {
    *runtime_ev.request.remaining_frames_out = runtime_ev.ctx.remaining_frames;
  }
};

struct effect_store_voice_complete_out {
  void operator()(const event::prefill_voice_run &runtime_ev,
                  context &) const noexcept {
    *runtime_ev.request.complete_out = runtime_ev.ctx.complete;
  }
};

struct effect_store_voice_remaining_out {
  void operator()(const event::prefill_voice_run &runtime_ev,
                  context &) const noexcept {
    *runtime_ev.request.remaining_frames_out = runtime_ev.ctx.remaining_frames;
  }
};

struct effect_begin_predict {
  void operator()(const event::predict_run &runtime_ev,
                  const context &) const noexcept {
    runtime_ev.ctx.err = detail_ns::to_error(error::none);
  }
};

struct effect_publish_predict {
  void operator()(const event::predict_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail_ns::to_error(error::none);
  }
};

struct effect_begin_sample {
  void operator()(const event::sample_run &runtime_ev,
                  const context &) const noexcept {
    runtime_ev.ctx.err = detail_ns::to_error(error::none);
    runtime_ev.ctx.graph_error = detail_ns::to_error(error::none);
    runtime_ev.ctx.graph_accepted = false;
  }
};

template <class graph_actor_type> struct effect_run_sample_graph {
  void operator()(const event::sample_run &runtime_ev, context &ctx,
                  graph_actor_type &graph) const noexcept {
    runtime_ev.ctx.graph_error = detail_ns::to_error(error::none);
    event::graph_step graph_step{
        *ctx.session.model,
        runtime_ev.request.prediction_workspace.memory,
        runtime_ev.request.model_tokens,
        runtime_ev.request.audio_tokens_out,
        runtime_ev.request.text_token_out,
    };
    graph_step.sequence_id = ctx.session.sequence_id;
    graph_step.error_out = &runtime_ev.ctx.graph_error;
    runtime_ev.ctx.graph_accepted = graph.process_event(graph_step);
  }
};

struct effect_publish_sample {
  void operator()(const event::sample_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail_ns::to_error(error::none);
  }
};

struct effect_store_sample_graph_error_out {
  void operator()(const event::sample_run &runtime_ev,
                  context &) const noexcept {
    *runtime_ev.request.graph_error_out = runtime_ev.ctx.graph_error;
  }
};

struct effect_capture_tokenizer_state {
  void operator()(const event::capture_tokenizer_state &ev,
                  const context &ctx) const noexcept {
    for (int32_t codebook = 0; codebook < ctx.lmgen.codebook_count;
         ++codebook) {
      for (int32_t row = 0; row < ctx.lmgen.cache_row_count; ++row) {
        const size_t destination =
            static_cast<size_t>(row) +
            static_cast<size_t>(codebook) *
                static_cast<size_t>(ctx.lmgen.cache_row_count);
        ev.cache_out[destination] = detail::cache_at(ctx.lmgen, row, codebook);
      }
    }
    ev.offset_out = ctx.lmgen.offset;
    ev.error_out = detail_ns::to_error(error::none);
  }
};

template <error error_value> struct effect_reject_capture_tokenizer_state {
  void operator()(const event::capture_tokenizer_state &ev,
                  const context &) const noexcept {
    ev.offset_out = 0;
    ev.error_out = detail_ns::to_error(error_value);
  }
};

struct effect_store_graph_error_out {
  template <class runtime_event_type>
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    *ev.request.graph_error_out = ev.ctx.graph_error;
  }
};

struct effect_reset_session {
  void operator()(const event::reset_run &, context &ctx) const noexcept {
    ctx.session = {};
    ctx.voice = {};
    ctx.lmgen = {};
  }
};

template <class runtime_event_type> struct effect_store_error_out {
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    *runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

struct effect_emit_initialize_done {
  void operator()(const event::initialize_run &runtime_ev,
                  context &ctx) const noexcept {
    runtime_ev.request.on_done(events::initialize_done{
        .request = &runtime_ev.request,
        .n_q = ctx.session.n_q,
        .dep_q = ctx.session.dep_q,
    });
  }
};

struct effect_emit_initialize_error {
  void operator()(const event::initialize_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.request.on_error(events::initialize_error{
        .request = &runtime_ev.request,
        .err = runtime_ev.ctx.err,
    });
  }
};

struct effect_emit_load_voice_done {
  void operator()(const event::load_voice_run &runtime_ev,
                  context &ctx) const noexcept {
    runtime_ev.request.on_done(events::load_voice_done{
        .request = &runtime_ev.request,
        .prompt_frames = ctx.voice.prompt_frame_count,
    });
  }
};

struct effect_emit_load_voice_error {
  void operator()(const event::load_voice_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.request.on_error(events::load_voice_error{
        .request = &runtime_ev.request,
        .err = runtime_ev.ctx.err,
    });
  }
};

struct effect_emit_prefill_voice_done {
  void operator()(const event::prefill_voice_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.request.on_done(events::prefill_voice_done{
        .request = &runtime_ev.request,
        .complete = runtime_ev.ctx.complete,
        .remaining_frames = runtime_ev.ctx.remaining_frames,
    });
  }
};

struct effect_emit_prefill_voice_error {
  void operator()(const event::prefill_voice_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.request.on_error(events::prefill_voice_error{
        .request = &runtime_ev.request,
        .err = runtime_ev.ctx.err,
    });
  }
};

struct effect_emit_begin_personaplex_prompt_done {
  void operator()(const event::begin_personaplex_prompt_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.request.on_done(events::begin_personaplex_prompt_done{
        .request = &runtime_ev.request,
        .remaining_frames = runtime_ev.ctx.remaining_frames,
    });
  }
};

struct effect_emit_begin_personaplex_prompt_error {
  void operator()(const event::begin_personaplex_prompt_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.request.on_error(events::begin_personaplex_prompt_error{
        .request = &runtime_ev.request,
        .err = runtime_ev.ctx.err,
    });
  }
};

struct effect_emit_prefill_personaplex_prompt_done {
  void operator()(const event::prefill_personaplex_prompt_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.request.on_done(events::prefill_personaplex_prompt_done{
        .request = &runtime_ev.request,
        .complete = runtime_ev.ctx.complete,
        .remaining_frames = runtime_ev.ctx.remaining_frames,
    });
  }
};

struct effect_emit_prefill_personaplex_prompt_error {
  void operator()(const event::prefill_personaplex_prompt_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.request.on_error(events::prefill_personaplex_prompt_error{
        .request = &runtime_ev.request,
        .err = runtime_ev.ctx.err,
    });
  }
};

struct effect_mark_unexpected {
  template <class unexpected_event_type>
  void operator()(const unexpected_event_type &ev, context &) const noexcept {
    if constexpr (requires { ev.ctx.err; }) {
      ev.ctx.err = detail_ns::to_error(error::unexpected_event);
    }
  }
};

struct effect_mark_unexpected_and_store {
  template <class unexpected_event_type>
  void operator()(const unexpected_event_type &ev,
                  context &ctx) const noexcept {
    effect_mark_unexpected{}(ev, ctx);
    if constexpr (requires {
                    ev.ctx.err;
                    ev.request.error_out;
                  }) {
      *ev.request.error_out = ev.ctx.err;
    }
  }
};

} // namespace emel::speech::predictor::moshi::action
