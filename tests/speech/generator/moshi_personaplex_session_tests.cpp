#include <algorithm>
#include <cmath>
#include <memory>
#include <span>
#include <vector>

#include <doctest/doctest.h>
#include <stateforward/sml.hpp>

#include "emel/model/data.hpp"
#include "emel/speech/codec/mimi/any.hpp"
#include "emel/speech/generator/moshi/personaplex/session/any.hpp"
#include "moshi_fixture.hpp"

TEST_CASE("PersonaPlex session runs its injected actors end to end") {
  namespace mimi = emel::speech::codec::mimi;
  namespace personaplex = emel::speech::generator::moshi::personaplex::session;
  namespace sml = stateforward::sml;
  using emel::speech::generator::moshi::test::load_fixture_or_skip;

  auto codec = load_fixture_or_skip("mimi-tiny.gguf");
  auto lm = load_fixture_or_skip("moshi-tiny-lm.gguf");
  auto voice = load_fixture_or_skip("moshi-tiny-voice.gguf");
  if (codec.model == nullptr || lm.model == nullptr || voice.model == nullptr) {
    return;
  }

  std::vector<float> encoder_prepared(
      mimi::prepared_arena_floats(*codec.model));
  std::vector<float> encoder_state(mimi::state_arena_floats(*codec.model));
  std::vector<float> encoder_workspace(
      mimi::workspace_arena_floats(*codec.model));
  std::vector<float> encoder_frame(mimi::frame_arena_floats(*codec.model));
  std::vector<float> decoder_prepared(
      mimi::prepared_arena_floats(*codec.model));
  std::vector<float> decoder_state(mimi::state_arena_floats(*codec.model));
  std::vector<float> decoder_workspace(
      mimi::workspace_arena_floats(*codec.model));
  std::vector<float> decoder_frame(mimi::frame_arena_floats(*codec.model));

  auto &lm_config = lm.model->moshi_lm;
  const size_t temporal_per_layer = static_cast<size_t>(lm_config.context) *
                                    static_cast<size_t>(lm_config.dim);
  std::vector<uint16_t> temporal_keys(
      static_cast<size_t>(lm_config.num_layers) * temporal_per_layer, 0u);
  std::vector<uint16_t> temporal_values(temporal_keys.size(), 0u);
  std::vector<size_t> temporal_offsets(
      static_cast<size_t>(lm_config.num_layers), 0u);
  for (int32_t layer = 0; layer < lm_config.num_layers; ++layer) {
    temporal_offsets[static_cast<size_t>(layer)] =
        static_cast<size_t>(layer) * temporal_per_layer;
  }
  int32_t temporal_offset = 0;

  const size_t depformer_per_layer =
      static_cast<size_t>(lm_config.depformer_context) *
      static_cast<size_t>(lm_config.depformer_dim);
  std::vector<uint16_t> depformer_keys(
      static_cast<size_t>(lm_config.depformer_num_layers) * depformer_per_layer,
      0u);
  std::vector<uint16_t> depformer_values(depformer_keys.size(), 0u);
  std::vector<size_t> depformer_offsets(
      static_cast<size_t>(lm_config.depformer_num_layers), 0u);
  for (int32_t layer = 0; layer < lm_config.depformer_num_layers; ++layer) {
    depformer_offsets[static_cast<size_t>(layer)] =
        static_cast<size_t>(layer) * depformer_per_layer;
  }
  int32_t depformer_offset = 0;

  personaplex::dependencies dependencies{
      .temporal_kv =
          {
              .key_cache = std::span<uint16_t>{temporal_keys},
              .value_cache = std::span<uint16_t>{temporal_values},
              .layer_cache_offsets = std::span<const size_t>{temporal_offsets},
              .offset = &temporal_offset,
              .layer_count = lm_config.num_layers,
              .position_capacity = lm_config.context,
              .block_tokens = 1,
              .kv_dim = lm_config.dim,
          },
      .depformer_kv =
          {
              .key_cache = std::span<uint16_t>{depformer_keys},
              .value_cache = std::span<uint16_t>{depformer_values},
              .layer_cache_offsets = std::span<const size_t>{depformer_offsets},
              .offset = &depformer_offset,
              .layer_count = lm_config.depformer_num_layers,
              .position_capacity = lm_config.depformer_context,
              .block_tokens = 1,
              .kv_dim = lm_config.depformer_dim,
          },
  };
  personaplex::sm session{dependencies};
  emel::error::type err = emel::error::cast(personaplex::error::none);
  personaplex::event::initialize initialize{
      .mimi_model = *codec.model,
      .lm_model = *lm.model,
      .voice_model = *voice.model,
      .encoder_storage =
          {
              .prepared = std::span<float>{encoder_prepared},
              .state = std::span<float>{encoder_state},
              .workspace = std::span<float>{encoder_workspace},
              .frame = std::span<float>{encoder_frame},
          },
      .decoder_storage =
          {
              .prepared = std::span<float>{decoder_prepared},
              .state = std::span<float>{decoder_state},
              .workspace = std::span<float>{decoder_workspace},
              .frame = std::span<float>{decoder_frame},
          },
      .sampling =
          {
              .enabled = true,
              .consume_forced_text = true,
              .audio_temperature = 0.8f,
              .text_temperature = 0.7f,
              .audio_top_k = 250,
              .text_top_k = 25,
              .seed = 1234,
          },
      .max_blocks = 16,
      .block_tokens = 4,
      .error_out = err,
  };

  auto guard_context =
      std::make_unique<personaplex::action::context>(dependencies);
  personaplex::event::initialize_ctx initialize_ctx{};
  personaplex::event::initialize_run initialize_run{initialize, initialize_ctx};
  const auto request_is_valid = [&] {
    return personaplex::guard::guard_initialize_request_valid{}(initialize_run,
                                                                *guard_context);
  };
  REQUIRE(request_is_valid());

  const int32_t temporal_layer_count = lm_config.num_layers;
  lm_config.num_layers = 0;
  CHECK_FALSE(request_is_valid());
  lm_config.num_layers = temporal_layer_count;
  const int32_t temporal_context_size = lm_config.context;
  lm_config.context = 0;
  CHECK_FALSE(request_is_valid());
  lm_config.context = temporal_context_size;
  const int32_t temporal_dimension = lm_config.dim;
  lm_config.dim = 0;
  CHECK_FALSE(request_is_valid());
  lm_config.dim = temporal_dimension;
  const int32_t depformer_layer_count = lm_config.depformer_num_layers;
  lm_config.depformer_num_layers = 0;
  CHECK_FALSE(request_is_valid());
  lm_config.depformer_num_layers = depformer_layer_count;
  const int32_t depformer_context_size = lm_config.depformer_context;
  lm_config.depformer_context = 0;
  CHECK_FALSE(request_is_valid());
  lm_config.depformer_context = depformer_context_size;
  const int32_t depformer_dimension = lm_config.depformer_dim;
  lm_config.depformer_dim = 0;
  CHECK_FALSE(request_is_valid());
  lm_config.depformer_dim = depformer_dimension;
  const int32_t inference_dep_q = lm_config.inference_dep_q;
  lm_config.inference_dep_q = 0;
  CHECK_FALSE(request_is_valid());
  lm_config.inference_dep_q = inference_dep_q;
  const int32_t codec_n_q = codec.model->mimi.n_q;
  codec.model->mimi.n_q = 0;
  CHECK_FALSE(request_is_valid());
  codec.model->mimi.n_q = codec_n_q;
  const int32_t lm_card = lm_config.card;
  lm_config.card = 0;
  CHECK_FALSE(request_is_valid());
  lm_config.card = lm_card;
  const int32_t codec_card = codec.model->mimi.card;
  codec.model->mimi.card = 0;
  CHECK_FALSE(request_is_valid());
  codec.model->mimi.card = codec_card;

  initialize.sampling.enabled = false;
  CHECK_FALSE(request_is_valid());
  initialize.sampling.enabled = true;
  initialize.sampling.consume_forced_text = false;
  CHECK_FALSE(request_is_valid());
  initialize.sampling.consume_forced_text = true;
  initialize.sampling.audio_temperature = 0.0f;
  CHECK_FALSE(request_is_valid());
  initialize.sampling.audio_temperature = 0.8f;
  initialize.sampling.text_temperature = 0.0f;
  CHECK_FALSE(request_is_valid());
  initialize.sampling.text_temperature = 0.7f;
  initialize.sampling.audio_top_k = 0;
  CHECK_FALSE(request_is_valid());
  initialize.sampling.audio_top_k = 250;
  initialize.sampling.text_top_k = 0;
  CHECK_FALSE(request_is_valid());
  initialize.sampling.text_top_k = 25;
  initialize.sampling.seed = 0;
  CHECK_FALSE(request_is_valid());
  initialize.sampling.seed = 1234;
  initialize.max_blocks = 0;
  CHECK_FALSE(request_is_valid());
  initialize.max_blocks = 16;
  initialize.block_tokens = 0;
  CHECK_FALSE(request_is_valid());
  initialize.block_tokens = 4;

  lm_config.inference_dep_q = inference_dep_q + 1;
  CHECK_FALSE(request_is_valid());
  lm_config.inference_dep_q = inference_dep_q;
  lm_config.card = lm_card + 1;
  CHECK_FALSE(request_is_valid());
  lm_config.card = lm_card;

  const auto temporal_key_view = guard_context->temporal_kv.key_cache;
  guard_context->temporal_kv.key_cache = {};
  CHECK_FALSE(request_is_valid());
  guard_context->temporal_kv.key_cache = temporal_key_view;
  const auto temporal_value_view = guard_context->temporal_kv.value_cache;
  guard_context->temporal_kv.value_cache = {};
  CHECK_FALSE(request_is_valid());
  guard_context->temporal_kv.value_cache = temporal_value_view;
  int32_t *const temporal_offset_ptr = guard_context->temporal_kv.offset;
  guard_context->temporal_kv.offset = nullptr;
  CHECK_FALSE(request_is_valid());
  guard_context->temporal_kv.offset = temporal_offset_ptr;
  const auto temporal_layer_views =
      guard_context->temporal_kv.layer_cache_offsets;
  guard_context->temporal_kv.layer_cache_offsets = {};
  CHECK_FALSE(request_is_valid());
  guard_context->temporal_kv.layer_cache_offsets = temporal_layer_views;
  --guard_context->temporal_kv.layer_count;
  CHECK_FALSE(request_is_valid());
  ++guard_context->temporal_kv.layer_count;
  --guard_context->temporal_kv.position_capacity;
  CHECK_FALSE(request_is_valid());
  ++guard_context->temporal_kv.position_capacity;
  guard_context->temporal_kv.block_tokens = 0;
  CHECK_FALSE(request_is_valid());
  guard_context->temporal_kv.block_tokens = 1;
  --guard_context->temporal_kv.kv_dim;
  CHECK_FALSE(request_is_valid());
  ++guard_context->temporal_kv.kv_dim;

  const auto depformer_key_view = guard_context->depformer_kv.key_cache;
  guard_context->depformer_kv.key_cache = {};
  CHECK_FALSE(request_is_valid());
  guard_context->depformer_kv.key_cache = depformer_key_view;
  const auto depformer_value_view = guard_context->depformer_kv.value_cache;
  guard_context->depformer_kv.value_cache = {};
  CHECK_FALSE(request_is_valid());
  guard_context->depformer_kv.value_cache = depformer_value_view;
  int32_t *const depformer_offset_ptr = guard_context->depformer_kv.offset;
  guard_context->depformer_kv.offset = nullptr;
  CHECK_FALSE(request_is_valid());
  guard_context->depformer_kv.offset = depformer_offset_ptr;
  const auto depformer_layer_views =
      guard_context->depformer_kv.layer_cache_offsets;
  guard_context->depformer_kv.layer_cache_offsets = {};
  CHECK_FALSE(request_is_valid());
  guard_context->depformer_kv.layer_cache_offsets = depformer_layer_views;
  --guard_context->depformer_kv.layer_count;
  CHECK_FALSE(request_is_valid());
  ++guard_context->depformer_kv.layer_count;
  --guard_context->depformer_kv.position_capacity;
  CHECK_FALSE(request_is_valid());
  ++guard_context->depformer_kv.position_capacity;
  guard_context->depformer_kv.block_tokens = 0;
  CHECK_FALSE(request_is_valid());
  guard_context->depformer_kv.block_tokens = 1;
  --guard_context->depformer_kv.kv_dim;
  CHECK_FALSE(request_is_valid());
  ++guard_context->depformer_kv.kv_dim;
  guard_context.reset();

  REQUIRE(session.process_event(initialize));

  for (int32_t step = 0;
       step < 16 && session.is(sml::state<personaplex::state_voice_prefill>);
       ++step) {
    personaplex::event::advance_voice advance{.error_out = err};
    REQUIRE(session.process_event(advance));
  }
  for (int32_t step = 0;
       step < 16 && session.is(sml::state<personaplex::state_prompt_prefill>);
       ++step) {
    personaplex::event::advance_prompt advance{
        .text_token = -1,
        .error_out = err,
    };
    REQUIRE(session.process_event(advance));
  }
  REQUIRE(session.is(sml::state<personaplex::state_live>));

  const int32_t frame_samples = static_cast<int32_t>(
      std::lround(static_cast<double>(codec.model->mimi.sample_rate) /
                  static_cast<double>(codec.model->mimi.frame_rate)));
  const int32_t public_n_q = lm_config.inference_dep_q;
  std::vector<float> pcm(static_cast<size_t>(frame_samples), 0.0f);
  for (int32_t index = 0; index < frame_samples; ++index) {
    pcm[static_cast<size_t>(index)] =
        0.04f * static_cast<float>((index % 97) - 48) / 48.0f;
  }
  std::vector<int32_t> input_codes(static_cast<size_t>(public_n_q), -1);
  std::vector<int32_t> output_codes(static_cast<size_t>(public_n_q), -1);
  std::vector<float> output_pcm(static_cast<size_t>(frame_samples), 0.0f);
  int32_t text_token = -1;
  bool produced = false;
  personaplex::event::live_frame live{
      .payload =
          {
              .pcm = std::span<const float>{pcm},
              .input_codes_out = std::span<int32_t>{input_codes},
              .output_codes_out = std::span<int32_t>{output_codes},
              .text_token_out = text_token,
              .produced_out = produced,
              .pcm_out = std::span<float>{output_pcm},
          },
      .error_out = err,
  };
  REQUIRE(session.process_event(live));

  personaplex::event::begin_flush begin_flush{.error_out = err};
  REQUIRE(session.process_event(begin_flush));
  std::fill(pcm.begin(), pcm.end(), 0.0f);
  for (int32_t step = 0; step < 8 && !produced; ++step) {
    personaplex::event::flush_frame flush{
        .payload =
            {
                .pcm = std::span<const float>{pcm},
                .input_codes_out = std::span<int32_t>{input_codes},
                .output_codes_out = std::span<int32_t>{output_codes},
                .text_token_out = text_token,
                .produced_out = produced,
                .pcm_out = std::span<float>{output_pcm},
            },
        .error_out = err,
    };
    REQUIRE(session.process_event(flush));
  }
  REQUIRE(produced);
  CHECK(std::any_of(output_pcm.begin(), output_pcm.end(),
                    [](const float sample) { return sample != 0.0f; }));

  personaplex::event::finish finish{.error_out = err};
  REQUIRE(session.process_event(finish));
  CHECK(session.is(sml::state<personaplex::state_done>));
}

TEST_CASE("PersonaPlex session rejects invalid injected dependencies") {
  namespace personaplex = emel::speech::generator::moshi::personaplex::session;
  namespace sml = stateforward::sml;

  auto model = std::make_unique<emel::model::data>();
  personaplex::dependencies dependencies{};
  personaplex::sm session{dependencies};
  emel::error::type err = emel::error::cast(personaplex::error::none);
  personaplex::event::initialize initialize{
      .mimi_model = *model,
      .lm_model = *model,
      .voice_model = *model,
      .sampling =
          {
              .enabled = true,
              .consume_forced_text = true,
              .audio_temperature = 0.8f,
              .text_temperature = 0.7f,
              .audio_top_k = 250,
              .text_top_k = 25,
              .seed = 1234,
          },
      .max_blocks = 4,
      .block_tokens = 4,
      .error_out = err,
  };

  CHECK_FALSE(session.process_event(initialize));
  CHECK(err == emel::error::cast(personaplex::error::invalid_request));
  CHECK(session.is(sml::state<personaplex::state_failed>));
}

TEST_CASE("PersonaPlex session models unexpected event ordering") {
  namespace personaplex = emel::speech::generator::moshi::personaplex::session;
  namespace sml = stateforward::sml;

  personaplex::dependencies dependencies{};
  personaplex::sm session{dependencies};
  emel::error::type err = emel::error::cast(personaplex::error::none);
  personaplex::event::finish finish{.error_out = err};

  CHECK_FALSE(session.process_event(finish));
  CHECK(err == emel::error::cast(personaplex::error::unexpected_event));
  CHECK(session.is(sml::state<personaplex::state_failed>));
}

TEST_CASE("PersonaPlex session guards and effects expose failure routes") {
  namespace personaplex = emel::speech::generator::moshi::personaplex::session;

  auto model = std::make_unique<emel::model::data>();
  personaplex::dependencies dependencies{};
  auto context = std::make_unique<personaplex::action::context>(dependencies);
  emel::error::type err = emel::error::cast(personaplex::error::none);
  personaplex::event::initialize initialize{
      .mimi_model = *model,
      .lm_model = *model,
      .voice_model = *model,
      .error_out = err,
  };
  personaplex::event::initialize_ctx initialize_ctx{};
  personaplex::event::initialize_run initialize_run{initialize, initialize_ctx};

  CHECK(personaplex::guard::guard_encoder_initialize_failed{}(initialize_run,
                                                              *context));
  initialize_ctx.child_accepted = true;
  initialize_ctx.frame_samples = 1920;
  initialize_ctx.mimi_n_q = 8;
  model->moshi_lm.inference_dep_q = 8;
  CHECK(personaplex::guard::guard_encoder_initialize_succeeded{}(initialize_run,
                                                                 *context));
  CHECK(personaplex::guard::guard_child_succeeded<
        personaplex::event::initialize_run>{}(initialize_run, *context));
  CHECK_FALSE(personaplex::guard::guard_child_failed<
              personaplex::event::initialize_run>{}(initialize_run, *context));
  personaplex::action::effect_fail_initialize_child<
      personaplex::error::codec_initialize_failed>{}(initialize_run, *context);
  CHECK(err == emel::error::cast(personaplex::error::codec_initialize_failed));

  personaplex::event::advance_voice voice{.error_out = err};
  personaplex::event::phase_ctx voice_ctx{};
  personaplex::event::advance_voice_run voice_run{voice, voice_ctx};
  voice_ctx.child_accepted = true;
  CHECK(personaplex::guard::guard_phase_succeeded_incomplete<
        personaplex::event::advance_voice_run>{}(voice_run, *context));
  voice_ctx.complete = true;
  CHECK(personaplex::guard::guard_phase_succeeded_complete<
        personaplex::event::advance_voice_run>{}(voice_run, *context));
  voice_ctx.graph_err = 1;
  CHECK(personaplex::guard::guard_phase_failed<
        personaplex::event::advance_voice_run>{}(voice_run, *context));
  personaplex::action::effect_fail_advance_voice<
      personaplex::error::voice_prefill_failed>{}(voice_run, *context);
  CHECK(err == emel::error::cast(personaplex::error::voice_prefill_failed));

  personaplex::event::advance_prompt prompt{
      .text_token = -1,
      .error_out = err,
  };
  personaplex::event::phase_ctx prompt_ctx{};
  personaplex::event::advance_prompt_run prompt_run{prompt, prompt_ctx};
  personaplex::action::effect_fail_advance_prompt<
      personaplex::error::prompt_prefill_failed>{}(prompt_run, *context);
  CHECK(err == emel::error::cast(personaplex::error::prompt_prefill_failed));

  int32_t text_token = -1;
  bool produced = true;
  personaplex::event::live_frame live{
      .payload =
          {
              .text_token_out = text_token,
              .produced_out = produced,
          },
      .error_out = err,
  };
  personaplex::event::frame_ctx frame_ctx{};
  personaplex::event::live_frame_run live_run{live, frame_ctx};
  CHECK(personaplex::guard::guard_frame_request_invalid<
        personaplex::event::live_frame_run>{}(live_run, *context));
  frame_ctx.child_accepted = true;
  frame_ctx.produced = true;
  CHECK(personaplex::guard::guard_frame_generated_and_produced<
        personaplex::event::live_frame_run>{}(live_run, *context));
  frame_ctx.produced = false;
  CHECK(personaplex::guard::guard_frame_generated_without_output<
        personaplex::event::live_frame_run>{}(live_run, *context));
  frame_ctx.child_err = 1;
  CHECK(personaplex::guard::guard_frame_generate_failed<
        personaplex::event::live_frame_run>{}(live_run, *context));
  personaplex::action::effect_fail_frame<personaplex::event::live_frame_run,
                                         personaplex::error::generate_failed>{}(
      live_run, *context);
  CHECK_FALSE(produced);
  CHECK(err == emel::error::cast(personaplex::error::generate_failed));

  personaplex::event::begin_flush begin_flush{.error_out = err};
  personaplex::event::simple_ctx begin_flush_ctx{};
  personaplex::event::begin_flush_run begin_flush_run{begin_flush,
                                                      begin_flush_ctx};
  personaplex::action::effect_fail_begin_flush_invalid{}(begin_flush_run,
                                                         *context);
  CHECK(err == emel::error::cast(personaplex::error::invalid_request));

  personaplex::event::finish finish{.error_out = err};
  personaplex::event::simple_ctx finish_ctx{};
  personaplex::event::finish_run finish_run{finish, finish_ctx};
  personaplex::action::effect_fail_finish_invalid{}(finish_run, *context);
  CHECK(err == emel::error::cast(personaplex::error::invalid_request));
}
