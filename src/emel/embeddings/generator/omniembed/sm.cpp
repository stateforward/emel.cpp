#include "emel/embeddings/generator/sm.hpp"

#include "emel/embeddings/generator/omniembed/detail.hpp"

namespace emel::embeddings::generator::component_route {

bool reserve_scratch(action::context & ctx,
                     const emel::model::data & model) noexcept {
  return omniembed::detail::reserve_scratch(ctx, model);
}

void prepare_image_input(const event::embed_image_run & ev,
                         action::context & ctx) noexcept {
  (void) omniembed::detail::prepare_image_input(
      ctx, ev.request.rgba, ev.request.width, ev.request.height);
  detail::finish_benchmark_prepare(ev);
}

void prepare_audio_input(const event::embed_audio_run & ev,
                         action::context & ctx) noexcept {
  (void) omniembed::detail::prepare_audio_input(
      ctx, ev.request.pcm, ev.request.sample_rate);
  detail::finish_benchmark_prepare(ev);
}

void run_text_embedding(const event::embed_text_run & ev,
                        action::context & ctx) noexcept {
  ev.ctx.err = omniembed::detail::run_text_embedding(ctx, ev.ctx.token_count);
  detail::finish_benchmark_encode(ev);
}

void run_image_embedding(const event::embed_image_run & ev,
                         action::context & ctx) noexcept {
  ev.ctx.err = omniembed::detail::run_image_embedding(ctx);
  detail::finish_benchmark_encode(ev);
}

void run_audio_embedding(const event::embed_audio_run & ev,
                         action::context & ctx) noexcept {
  ev.ctx.err = omniembed::detail::run_audio_embedding(ctx);
  detail::finish_benchmark_encode(ev);
}

}  // namespace emel::embeddings::generator::component_route
