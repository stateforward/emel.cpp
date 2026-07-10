#pragma once

#include <cstddef>
#include <limits>

#include "emel/speech/generator/moshi/personaplex/session/context.hpp"
#include "emel/speech/generator/moshi/personaplex/session/events.hpp"

namespace emel::speech::generator::moshi::personaplex::session::guard {

struct guard_initialize_request_valid {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const auto &request = runtime_ev.request;
    const auto &lm = request.lm_model.moshi_lm;
    const auto &mimi = request.mimi_model.mimi;
    if (lm.num_layers <= 0 || lm.context <= 0 || lm.dim <= 0 ||
        lm.depformer_num_layers <= 0 || lm.depformer_context <= 0 ||
        lm.depformer_dim <= 0 || lm.inference_dep_q <= 0 || mimi.n_q <= 0 ||
        lm.card <= 0 || mimi.card <= 0) {
      return false;
    }
    const size_t temporal_layers = static_cast<size_t>(lm.num_layers);
    const size_t temporal_context = static_cast<size_t>(lm.context);
    const size_t depformer_layers =
        static_cast<size_t>(lm.depformer_num_layers);
    const size_t depformer_context = static_cast<size_t>(lm.depformer_context);
    if (temporal_context >
            std::numeric_limits<size_t>::max() / temporal_layers ||
        static_cast<size_t>(lm.dim) >
            std::numeric_limits<size_t>::max() /
                (temporal_layers * temporal_context) ||
        depformer_context >
            std::numeric_limits<size_t>::max() / depformer_layers ||
        static_cast<size_t>(lm.depformer_dim) >
            std::numeric_limits<size_t>::max() /
                (depformer_layers * depformer_context)) {
      return false;
    }
    const size_t temporal_elements =
        temporal_layers * temporal_context * static_cast<size_t>(lm.dim);
    const size_t depformer_elements = depformer_layers * depformer_context *
                                      static_cast<size_t>(lm.depformer_dim);
    return request.sampling.enabled && request.sampling.consume_forced_text &&
           request.sampling.audio_temperature > 0.0f &&
           request.sampling.text_temperature > 0.0f &&
           request.sampling.audio_top_k > 0 &&
           request.sampling.text_top_k > 0 && request.sampling.seed != 0u &&
           request.max_blocks > 0 && request.block_tokens > 0 &&
           lm.inference_dep_q == mimi.n_q && lm.card == mimi.card &&
           ctx.temporal_kv.key_cache.data() != nullptr &&
           ctx.temporal_kv.value_cache.data() != nullptr &&
           ctx.temporal_kv.key_cache.size() >= temporal_elements &&
           ctx.temporal_kv.value_cache.size() >= temporal_elements &&
           ctx.temporal_kv.layer_cache_offsets.size() >=
               static_cast<size_t>(lm.num_layers) &&
           ctx.temporal_kv.layer_count == lm.num_layers &&
           ctx.temporal_kv.position_capacity == lm.context &&
           ctx.temporal_kv.kv_dim == lm.dim &&
           ctx.depformer_kv.key_cache.data() != nullptr &&
           ctx.depformer_kv.value_cache.data() != nullptr &&
           ctx.depformer_kv.key_cache.size() >= depformer_elements &&
           ctx.depformer_kv.value_cache.size() >= depformer_elements &&
           ctx.depformer_kv.layer_cache_offsets.size() >=
               static_cast<size_t>(lm.depformer_num_layers) &&
           ctx.depformer_kv.layer_count == lm.depformer_num_layers &&
           ctx.depformer_kv.position_capacity == lm.depformer_context &&
           ctx.depformer_kv.kv_dim == lm.depformer_dim;
  }
};

struct guard_initialize_request_invalid {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_initialize_request_valid{}(runtime_ev, ctx);
  }
};

struct guard_encoder_initialize_succeeded {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.child_accepted && runtime_ev.ctx.child_err == 0 &&
           runtime_ev.ctx.frame_samples > 0 && runtime_ev.ctx.mimi_n_q > 0 &&
           runtime_ev.ctx.mimi_n_q ==
               runtime_ev.request.lm_model.moshi_lm.inference_dep_q;
  }
};

struct guard_encoder_initialize_failed {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_encoder_initialize_succeeded{}(runtime_ev, ctx);
  }
};

template <class runtime_event_type> struct guard_child_succeeded {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.child_accepted && runtime_ev.ctx.child_err == 0;
  }
};

template <class runtime_event_type> struct guard_child_failed {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_child_succeeded<runtime_event_type>{}(runtime_ev, ctx);
  }
};

template <class runtime_event_type> struct guard_phase_succeeded_incomplete {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.child_accepted && runtime_ev.ctx.child_err == 0 &&
           runtime_ev.ctx.graph_err == 0 && !runtime_ev.ctx.complete;
  }
};

template <class runtime_event_type> struct guard_phase_succeeded_complete {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.child_accepted && runtime_ev.ctx.child_err == 0 &&
           runtime_ev.ctx.graph_err == 0 && runtime_ev.ctx.complete;
  }
};

template <class runtime_event_type> struct guard_phase_failed {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    return !runtime_ev.ctx.child_accepted || runtime_ev.ctx.child_err != 0 ||
           runtime_ev.ctx.graph_err != 0;
  }
};

template <class runtime_event_type> struct guard_frame_request_valid {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    const auto &request = runtime_ev.request;
    const auto &payload = request.payload;
    return payload.pcm.data() != nullptr &&
           payload.input_codes_out.data() != nullptr &&
           payload.output_codes_out.data() != nullptr &&
           payload.pcm_out.data() != nullptr &&
           payload.pcm.size() == static_cast<size_t>(ctx.frame_samples) &&
           payload.input_codes_out.size() >=
               static_cast<size_t>(ctx.public_n_q) &&
           payload.output_codes_out.size() >=
               static_cast<size_t>(ctx.public_n_q) &&
           payload.pcm_out.size() >= static_cast<size_t>(ctx.frame_samples);
  }
};

template <class runtime_event_type> struct guard_frame_request_invalid {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_frame_request_valid<runtime_event_type>{}(runtime_ev, ctx);
  }
};

template <class runtime_event_type> struct guard_frame_generated_and_produced {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.child_accepted && runtime_ev.ctx.child_err == 0 &&
           runtime_ev.ctx.graph_err == 0 && runtime_ev.ctx.produced;
  }
};

template <class runtime_event_type>
struct guard_frame_generated_without_output {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.child_accepted && runtime_ev.ctx.child_err == 0 &&
           runtime_ev.ctx.graph_err == 0 && !runtime_ev.ctx.produced;
  }
};

template <class runtime_event_type> struct guard_frame_generate_failed {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    return !runtime_ev.ctx.child_accepted || runtime_ev.ctx.child_err != 0 ||
           runtime_ev.ctx.graph_err != 0;
  }
};

} // namespace emel::speech::generator::moshi::personaplex::session::guard
