#pragma once

#include "emel/embeddings/generator/detail.hpp"
#include "emel/embeddings/generator/omniembed/detail.hpp"

namespace emel::embeddings::generator::omniembed {

struct route {
  using context = emel::embeddings::generator::action::context;

  static bool reserve_scratch(action::context & ctx,
                              const emel::model::data & model) noexcept {
    return detail::reserve_scratch(ctx, model);
  }

  struct effect_prepare_image_input {
    template <class runtime_event_type>
    void operator()(const runtime_event_type & runtime_ev,
                    action::context & ctx) const noexcept {
      auto & ev = emel::embeddings::generator::detail::unwrap_runtime_event(runtime_ev);
      (void) detail::prepare_image_input(
          ctx, ev.request.rgba, ev.request.width, ev.request.height);
      emel::embeddings::generator::detail::finish_benchmark_prepare(ev);
    }
  };

  struct effect_prepare_audio_input {
    template <class runtime_event_type>
    void operator()(const runtime_event_type & runtime_ev,
                    action::context & ctx) const noexcept {
      auto & ev = emel::embeddings::generator::detail::unwrap_runtime_event(runtime_ev);
      (void) detail::prepare_audio_input(ctx, ev.request.pcm, ev.request.sample_rate);
      emel::embeddings::generator::detail::finish_benchmark_prepare(ev);
    }
  };

  struct effect_run_text_embedding {
    template <class runtime_event_type>
    void operator()(const runtime_event_type & runtime_ev,
                    action::context & ctx) const noexcept {
      auto & ev = emel::embeddings::generator::detail::unwrap_runtime_event(runtime_ev);
      ev.ctx.err = detail::run_text_embedding(ctx, ev.ctx.token_count);
      emel::embeddings::generator::detail::finish_benchmark_encode(ev);
    }
  };

  struct effect_run_image_embedding {
    template <class runtime_event_type>
    void operator()(const runtime_event_type & runtime_ev,
                    action::context & ctx) const noexcept {
      auto & ev = emel::embeddings::generator::detail::unwrap_runtime_event(runtime_ev);
      ev.ctx.err = detail::run_image_embedding(ctx);
      emel::embeddings::generator::detail::finish_benchmark_encode(ev);
    }
  };

  struct effect_run_audio_embedding {
    template <class runtime_event_type>
    void operator()(const runtime_event_type & runtime_ev,
                    action::context & ctx) const noexcept {
      auto & ev = emel::embeddings::generator::detail::unwrap_runtime_event(runtime_ev);
      ev.ctx.err = detail::run_audio_embedding(ctx);
      emel::embeddings::generator::detail::finish_benchmark_encode(ev);
    }
  };
};

}  // namespace emel::embeddings::generator::omniembed
