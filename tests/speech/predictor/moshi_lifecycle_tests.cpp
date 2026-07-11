#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <doctest/doctest.h>

#include "../../memory/recording_kv_actor.hpp"
#include "emel/logits/sampler/sm.hpp"
#include "emel/memory/streaming/sm.hpp"
#include "emel/model/moshi/detail.hpp"
#include "emel/speech/predictor/moshi/any.hpp"
#include "emel/speech/predictor/moshi/executor/any.hpp"
#include "moshi_fixture.hpp"

namespace {

namespace moshi = emel::speech::predictor::moshi;
namespace moshi_executor = emel::speech::predictor::moshi::executor;
namespace memory_streaming = emel::memory::streaming;
namespace model_moshi = emel::model::moshi::detail;
using emel::speech::predictor::moshi::test::load_fixture_or_skip;
using emel::speech::predictor::moshi::test::read_binary_file;
using emel::speech::predictor::moshi::test::repo_root;

inline constexpr emel::error::type k_no_error =
    emel::error::cast(moshi::error::none);

moshi_executor::dependencies make_executor_dependencies(
    emel::kernel::sm &kernel, const moshi_executor::kv_views &kv = {},
    emel::logits::sampler::sm *sampler = nullptr) noexcept {
  return moshi_executor::dependencies{
      .kv = kv,
      .kernel = kernel,
      .sampler = sampler,
      .policy =
          moshi_executor::action::policies{
              .rms_norm_epsilon = 1.0e-8f,
              .zero_seed_state = 123459876u,
              .sampling_modulus = 2147483647u,
              .token_zero = -1,
          },
      .capacity =
          moshi_executor::action::capacities{
              .hidden_dim = moshi_executor::detail::k_max_hidden_dim,
              .temporal_context =
                  moshi_executor::detail::k_max_temporal_context,
              .depformer_context =
                  moshi_executor::detail::k_max_depformer_context,
              .sampling_card = moshi_executor::detail::k_max_sampling_card,
              .sampling_top_k = moshi_executor::detail::k_max_sampling_top_k,
          },
  };
}

template <class graph_actor_type>
moshi::action::dependencies<graph_actor_type>
make_predictor_dependencies(graph_actor_type &graph,
                            emel::memory::hybrid::sm &memory) noexcept {
  return moshi::action::dependencies<graph_actor_type>{
      .memory = memory,
      .graph = graph,
      .policy =
          moshi::action::policies{
              .token_zero = -1,
              .token_ungenerated = -2,
              .prediction_step_size = 1,
              .prediction_output_count = 1,
          },
  };
}

void configure_predictor_initialize(moshi::event::initialize &request) {
  int32_t maximum_delay = 0;
  for (uint32_t index = 0; index < request.model.moshi_lm.delay_count;
       ++index) {
    maximum_delay =
        std::max(maximum_delay, request.model.moshi_lm.delays[index]);
  }
  request.max_sequences = 1;
  request.max_blocks = 64;
  request.block_tokens = 4;
  request.sequence_id = 0;
  request.codebook_capacity = request.model.moshi_lm.n_q + 1;
  request.delay_cache_row_capacity = maximum_delay + 3;
}

struct recording_graph_executor {
  int32_t call_count = 0;
  int32_t external_embedding_count = 0;
  int32_t first_sequence_length = -1;
  int32_t second_sequence_length = -1;
  int32_t last_sequence_length = -1;
  int32_t first_forced_text_token = -1;
  int32_t last_forced_text_token = -1;
  float first_embedding_sample = 0.0f;
  std::array<int32_t, moshi::event::k_max_codebooks> first_input = {};
  std::array<int32_t, moshi::event::k_max_codebooks> last_input = {};

  bool process_event(const moshi::event::initialize_graph &request) noexcept {
    *request.error_out = k_no_error;
    return true;
  }
  bool process_event(const moshi::event::graph_step &step);
};

struct rejecting_graph_executor {
  bool process_event(const moshi::event::initialize_graph &request) noexcept {
    *request.error_out = k_no_error;
    return true;
  }
  bool process_event(const moshi::event::graph_step &) noexcept {
    return false;
  }
};

struct temporal_kv_probe {
  int32_t call_count = 0;
  int32_t sequence_id = -1;
  int32_t sequence_length = -1;
  std::vector<uint16_t> key_cache = {};
  std::vector<uint16_t> value_cache = {};
  std::vector<size_t> layer_offsets = {};
};

struct depformer_kv_probe {
  int32_t call_count = 0;
  int32_t sequence_id = -1;
  int32_t sequence_length = -1;
  std::vector<uint16_t> key_cache = {};
  std::vector<uint16_t> value_cache = {};
  std::vector<size_t> layer_offsets = {};
};

struct moshi_callback_probe {
  int32_t done_count = 0;
  int32_t error_count = 0;
  int32_t n_q = -1;
  int32_t dep_q = -1;
  int32_t prompt_frames = -1;
  int32_t remaining_frames = -1;
  bool complete = false;
  bool produced = false;
  emel::error::type err = k_no_error;
  const void *request = nullptr;

  void on_generator_initialize_done(
      const moshi::events::initialize_done &done) noexcept {
    ++done_count;
    request = done.request;
    n_q = done.n_q;
    dep_q = done.dep_q;
  }

  void on_generator_initialize_error(
      const moshi::events::initialize_error &error) noexcept {
    ++error_count;
    request = error.request;
    err = error.err;
  }

  void on_generator_load_voice_done(
      const moshi::events::load_voice_done &done) noexcept {
    ++done_count;
    request = done.request;
    prompt_frames = done.prompt_frames;
  }

  void on_generator_load_voice_error(
      const moshi::events::load_voice_error &error) noexcept {
    ++error_count;
    request = error.request;
    err = error.err;
  }

  void on_generator_prefill_voice_done(
      const moshi::events::prefill_voice_done &done) noexcept {
    ++done_count;
    request = done.request;
    complete = done.complete;
    remaining_frames = done.remaining_frames;
  }

  void on_generator_prefill_voice_error(
      const moshi::events::prefill_voice_error &error) noexcept {
    ++error_count;
    request = error.request;
    err = error.err;
  }

  void on_generator_begin_prompt_done(
      const moshi::events::begin_personaplex_prompt_done &done) noexcept {
    ++done_count;
    request = done.request;
    remaining_frames = done.remaining_frames;
  }

  void on_generator_begin_prompt_error(
      const moshi::events::begin_personaplex_prompt_error &error) noexcept {
    ++error_count;
    request = error.request;
    err = error.err;
  }

  void on_generator_prefill_prompt_done(
      const moshi::events::prefill_personaplex_prompt_done &done) noexcept {
    ++done_count;
    request = done.request;
    complete = done.complete;
    remaining_frames = done.remaining_frames;
  }

  void on_generator_prefill_prompt_error(
      const moshi::events::prefill_personaplex_prompt_error &error) noexcept {
    ++error_count;
    request = error.request;
    err = error.err;
  }

  void on_executor_initialize_done(
      const moshi_executor::events::initialize_done &done) noexcept {
    ++done_count;
    request = done.request;
    n_q = done.n_q;
    dep_q = done.dep_q;
  }

  void on_executor_initialize_error(
      const moshi_executor::events::initialize_error &error) noexcept {
    ++error_count;
    request = error.request;
    err = error.err;
  }
};

bool recording_graph_executor::process_event(
    const moshi::event::graph_step &step) {
  auto &executor = *this;
  ++executor.call_count;
  if (!step.input_embedding.empty()) {
    ++executor.external_embedding_count;
    if (executor.external_embedding_count == 1) {
      executor.first_forced_text_token = step.forced_text_token;
      executor.first_embedding_sample = step.input_embedding[0];
    }
  }
  std::fill(executor.last_input.begin(), executor.last_input.end(), -99);
  std::copy(step.input_sequence.begin(), step.input_sequence.end(),
            executor.last_input.begin());
  executor.last_sequence_length = step.memory_snapshot.sequence_length(0);
  executor.last_forced_text_token = step.forced_text_token;
  if (executor.call_count == 1) {
    std::fill(executor.first_input.begin(), executor.first_input.end(), -99);
    std::copy(step.input_sequence.begin(), step.input_sequence.end(),
              executor.first_input.begin());
    executor.first_sequence_length = step.memory_snapshot.sequence_length(0);
  }
  if (executor.call_count == 2) {
    executor.second_sequence_length = step.memory_snapshot.sequence_length(0);
  }
  if (step.error_out != nullptr) {
    *step.error_out = k_no_error;
  }
  step.text_token_out = 100 + executor.call_count;
  for (size_t index = 0; index < step.audio_tokens_out.size(); ++index) {
    step.audio_tokens_out[index] = static_cast<int32_t>(
        1000 + executor.call_count * 10 + static_cast<int32_t>(index));
  }
  return true;
}

bool temporal_kv_probe_bind(void *cache_ptr, const emel::model::data &model,
                            const emel::memory::view::snapshot &snapshot,
                            const int32_t sequence_id,
                            moshi_executor::detail::temporal_kv_view &view) {
  auto &probe = *static_cast<temporal_kv_probe *>(cache_ptr);
  ++probe.call_count;
  probe.sequence_id = sequence_id;
  probe.sequence_length = snapshot.sequence_length(sequence_id);
  const int32_t position_capacity = model.moshi_lm.context;
  const int32_t kv_dim = model.moshi_lm.dim;
  const int32_t layer_count = model.moshi_lm.num_layers;
  REQUIRE(position_capacity > 0);
  REQUIRE(kv_dim > 0);
  REQUIRE(layer_count > 0);
  const size_t per_layer =
      static_cast<size_t>(position_capacity) * static_cast<size_t>(kv_dim);
  probe.layer_offsets.resize(static_cast<size_t>(layer_count));
  for (int32_t layer = 0; layer < layer_count; ++layer) {
    probe.layer_offsets[static_cast<size_t>(layer)] =
        static_cast<size_t>(layer) * per_layer;
  }
  probe.key_cache.assign(static_cast<size_t>(layer_count) * per_layer, 0u);
  probe.value_cache.assign(probe.key_cache.size(), 0u);
  view.key_cache =
      std::span<uint16_t>{probe.key_cache.data(), probe.key_cache.size()};
  view.value_cache =
      std::span<uint16_t>{probe.value_cache.data(), probe.value_cache.size()};
  view.layer_cache_offsets = probe.layer_offsets;
  view.layer_count = layer_count;
  view.position_capacity = position_capacity;
  view.kv_dim = kv_dim;
  return true;
}

bool depformer_kv_probe_bind(void *cache_ptr, const emel::model::data &model,
                             const emel::memory::view::snapshot &snapshot,
                             const int32_t sequence_id,
                             moshi_executor::detail::depformer_kv_view &view) {
  auto &probe = *static_cast<depformer_kv_probe *>(cache_ptr);
  ++probe.call_count;
  probe.sequence_id = sequence_id;
  probe.sequence_length = snapshot.sequence_length(sequence_id);
  const int32_t position_capacity = model.moshi_lm.depformer_context;
  const int32_t kv_dim = model.moshi_lm.depformer_dim;
  const int32_t layer_count = model.moshi_lm.depformer_num_layers;
  REQUIRE(position_capacity > 0);
  REQUIRE(kv_dim > 0);
  REQUIRE(layer_count > 0);
  const size_t per_layer =
      static_cast<size_t>(position_capacity) * static_cast<size_t>(kv_dim);
  if (probe.key_cache.empty()) {
    probe.layer_offsets.resize(static_cast<size_t>(layer_count));
    for (int32_t layer = 0; layer < layer_count; ++layer) {
      probe.layer_offsets[static_cast<size_t>(layer)] =
          static_cast<size_t>(layer) * per_layer;
    }
    probe.key_cache.assign(static_cast<size_t>(layer_count) * per_layer, 0u);
    probe.value_cache.assign(probe.key_cache.size(), 0u);
  }
  view.key_cache =
      std::span<uint16_t>{probe.key_cache.data(), probe.key_cache.size()};
  view.value_cache =
      std::span<uint16_t>{probe.value_cache.data(), probe.value_cache.size()};
  view.layer_cache_offsets = probe.layer_offsets;
  view.layer_count = layer_count;
  view.position_capacity = position_capacity;
  view.kv_dim = kv_dim;
  return true;
}

moshi_executor::kv_views
prepare_kv_views(const emel::model::data &model, temporal_kv_probe *temporal,
                 depformer_kv_probe *depformer,
                 memory_streaming::sm *temporal_positions = nullptr,
                 memory_streaming::sm *depformer_positions = nullptr) {
  moshi_executor::kv_views views{
      .temporal_positions = temporal_positions,
      .depformer_positions = depformer_positions,
  };
  emel::memory::view::snapshot snapshot{};
  if (temporal != nullptr) {
    REQUIRE(
        temporal_kv_probe_bind(temporal, model, snapshot, 0, views.temporal));
  }
  if (depformer != nullptr) {
    REQUIRE(depformer_kv_probe_bind(depformer, model, snapshot, 0,
                                    views.depformer));
  }
  return views;
}

void initialize_streaming_position(memory_streaming::sm &position) {
  int32_t err =
      static_cast<int32_t>(emel::error::cast(memory_streaming::error::none));
  REQUIRE(position.process_event(
      memory_streaming::event::initialize{.error_out = err}));
  REQUIRE(err == static_cast<int32_t>(
                     emel::error::cast(memory_streaming::error::none)));
}

} // namespace

TEST_CASE("speech_moshi_predictor_rejects_predict_before_initialize") {
  recording_graph_executor graph{};
  emel::memory::hybrid::sm memory{};
  moshi::sm generator{make_predictor_dependencies(graph, memory)};
  std::array<int32_t, 4> input = {0, 0, 0, 0};
  moshi::event::predict::workspace workspace{};
  emel::error::type err = k_no_error;

  moshi::event::predict predict{std::span<const int32_t>{input}, workspace, 1,
                                1};
  predict.error_out = &err;

  CHECK_FALSE(generator.process_event(predict));
  CHECK(err == emel::error::cast(moshi::error::not_initialized));
}

TEST_CASE("speech_moshi_generator_rejects_non_lm_model_contract") {
  auto model = std::make_unique<emel::model::data>();
  model->moshi_component_id = emel::model::data::moshi_component::mimi;
  recording_graph_executor graph{};
  emel::memory::hybrid::sm memory{};
  moshi::sm generator{make_predictor_dependencies(graph, memory)};
  emel::error::type err = k_no_error;

  moshi::event::initialize init{*model};
  init.error_out = &err;

  CHECK_FALSE(generator.process_event(init));
  CHECK(err == emel::error::cast(moshi::error::bind_failed));
}

TEST_CASE("speech_moshi_generator_initializes_lm_with_injected_kv") {
  auto fixture = load_fixture_or_skip("moshi-tiny-lm.gguf");
  if (fixture.model == nullptr) {
    return;
  }
  REQUIRE(model_moshi::validate_execution_contract(*fixture.model) ==
          emel::error::cast(emel::model::loader::error::none));

  emel::memory::test::recording_kv_actor kv{};
  emel::memory::hybrid::sm memory{emel::memory::hybrid::bind_kv_actor(kv)};
  recording_graph_executor graph{};
  moshi::sm generator{make_predictor_dependencies(graph, memory)};
  emel::error::type err = k_no_error;

  moshi::event::initialize init{*fixture.model};
  configure_predictor_initialize(init);
  init.max_sequences = 2;
  init.max_blocks = 16;
  init.block_tokens = 4;
  init.error_out = &err;

  REQUIRE(generator.process_event(init));
  CHECK(err == k_no_error);
  CHECK(kv.reserve_count == 1);
  CHECK(kv.allocate_sequence_count == 1);
}

TEST_CASE("speech_moshi_predictor_rejects_missing_initialize_capacity") {
  auto fixture = load_fixture_or_skip("moshi-tiny-lm.gguf");
  if (fixture.model == nullptr) {
    return;
  }

  recording_graph_executor graph{};
  emel::memory::hybrid::sm memory{};
  moshi::sm generator{make_predictor_dependencies(graph, memory)};
  emel::error::type err = k_no_error;
  moshi::event::initialize init{*fixture.model};
  init.error_out = &err;

  CHECK_FALSE(generator.process_event(init));
  CHECK(err == emel::error::cast(moshi::error::bind_failed));
}

TEST_CASE("speech_moshi_predictor_rejects_missing_token_policy") {
  auto fixture = load_fixture_or_skip("moshi-tiny-lm.gguf");
  if (fixture.model == nullptr) {
    return;
  }

  recording_graph_executor graph{};
  emel::memory::hybrid::sm memory{};
  moshi::sm generator{
      moshi::action::dependencies{.memory = memory, .graph = graph}};
  emel::error::type err = k_no_error;
  moshi::event::initialize init{*fixture.model};
  configure_predictor_initialize(init);
  init.error_out = &err;

  CHECK_FALSE(generator.process_event(init));
  CHECK(err == emel::error::cast(moshi::error::bind_failed));
}

TEST_CASE("speech_moshi_generator_personaplex_uses_model_inference_codebooks") {
  auto model = std::make_unique<emel::model::data>();
  model->moshi_lm.n_q = 16;
  model->moshi_lm.dep_q = 16;
  model->moshi_lm.inference_dep_q = 8;
  model->moshi_lm.text_padding_id = 3;
  model->moshi_lm.depformer_weights_per_step = true;
  model->moshi_lm.delay_count = 17;
  model->moshi_lm.inference_prompt_token_count = 17;
  model->moshi_lm.inference_prompt_tokens[0] = 3;
  model->moshi_lm.delays[0] = 0;
  for (uint32_t index = 1; index < model->moshi_lm.delay_count; ++index) {
    model->moshi_lm.delays[index] = 1;
    model->moshi_lm.inference_prompt_tokens[index] =
        static_cast<int32_t>(index);
  }

  moshi::event::initialize init{*model};
  moshi::event::initialize_ctx init_ctx{};
  moshi::event::initialize_run run{init, init_ctx};
  emel::memory::hybrid::sm memory{};
  moshi::action::context ctx{memory};

  moshi::action::effect_bind_contract{}(run, ctx);
  CHECK(moshi::guard::guard_personaplex_lmgen{}(run, ctx));
  moshi::action::effect_configure_personaplex_lmgen{}(run, ctx);

  CHECK(ctx.lmgen.generated_dep_q == 16);
  CHECK(ctx.lmgen.delayed_dep_q == model->moshi_lm.inference_dep_q);
  CHECK(ctx.lmgen.needed_tokens == 8);
  CHECK(ctx.lmgen.cache_row_count == 4);
}

TEST_CASE(
    "speech_moshi_generator_loads_personaplex_voice_and_prefills_prompt") {
  auto fixture = load_fixture_or_skip("moshi-tiny-lm.gguf");
  auto voice = load_fixture_or_skip("moshi-tiny-voice.gguf");
  if (fixture.model == nullptr || voice.model == nullptr) {
    return;
  }

  emel::memory::test::recording_kv_actor kv{};
  emel::memory::hybrid::sm memory{emel::memory::hybrid::bind_kv_actor(kv)};
  recording_graph_executor graph{};
  moshi::sm generator{make_predictor_dependencies(graph, memory)};
  emel::error::type err = k_no_error;

  moshi::event::initialize init{*fixture.model};
  configure_predictor_initialize(init);
  init.max_blocks = 32;
  init.block_tokens = 4;
  init.error_out = &err;
  REQUIRE(generator.process_event(init));
  CHECK(err == k_no_error);

  const int32_t prompt_frames = 5;
  moshi::event::load_voice load{*voice.model};
  load.error_out = &err;
  REQUIRE(generator.process_event(load));
  CHECK(err == k_no_error);

  std::array<int32_t, moshi::event::k_max_codebooks> model_tokens = {};
  moshi::event::predict::workspace workspace{};
  moshi::event::predict blocked_predict{
      std::span<const int32_t>{
          model_tokens.data(),
          static_cast<size_t>(fixture.model->moshi_lm.n_q + 1)},
      workspace, 1, 1};
  blocked_predict.error_out = &err;
  CHECK_FALSE(generator.process_event(blocked_predict));
  CHECK(err == emel::error::cast(moshi::error::voice_prompt_pending));

  for (int32_t index = 0; index < prompt_frames; ++index) {
    bool complete = true;
    int32_t remaining = -1;
    emel::error::type graph_err = emel::error::cast(moshi::error::none);
    err = k_no_error;
    moshi::event::prefill_voice prefill{};
    prefill.error_out = &err;
    prefill.graph_error_out = &graph_err;
    prefill.complete_out = &complete;
    prefill.remaining_frames_out = &remaining;
    REQUIRE(generator.process_event(prefill));
    CHECK(err == k_no_error);
    CHECK(graph_err == k_no_error);
    CHECK(complete == (index + 1 == prompt_frames));
    CHECK(remaining == prompt_frames - index - 1);
  }

  CHECK(graph.external_embedding_count == prompt_frames);
  CHECK(graph.first_forced_text_token ==
        fixture.model->moshi_lm.text_padding_id);
  CHECK(graph.first_embedding_sample != 0.0f);
  CHECK(kv.allocate_slots_count >= prompt_frames);
  CHECK(kv.capture_view_count >= prompt_frames);
}

TEST_CASE("speech_moshi_generator_loads_personaplex_voice_embedding_dtypes") {
  auto snapshot = std::make_unique<emel::memory::view::snapshot>();
  moshi::event::prefill_voice prefill{};
  moshi::event::prefill_voice_ctx prefill_ctx{*snapshot};
  moshi::event::prefill_voice_run prefill_run{prefill, prefill_ctx};
  emel::memory::hybrid::sm memory{};
  moshi::action::context ctx{memory};
  emel::model::data::tensor_record tensor{};
  std::array<uint16_t, 4> data = {0x3c00u, 0x4000u, 0x4200u, 0x4400u};
  tensor.data = data.data();
  tensor.dims[0] = 2;
  tensor.dims[3] = 2;
  ctx.voice.contract.voice.embeddings.tensor = &tensor;
  ctx.voice.embedding_dim = 2;
  ctx.voice.prompt_frame_index = 1;

  tensor.type = emel::kernel::detail::dtype_f16;
  moshi::action::effect_load_voice_embedding_frame_f16{}(prefill_run, ctx);
  CHECK(prefill_ctx.embedding_frame_ok);
  CHECK(prefill_ctx.embedding_frame[0] != 0.0f);
  CHECK(prefill_ctx.embedding_frame[1] != 0.0f);

  prefill_ctx.embedding_frame_ok = false;
  prefill_ctx.embedding_frame[0] = 0.0f;
  prefill_ctx.embedding_frame[1] = 0.0f;
  tensor.type = emel::kernel::detail::dtype_bf16;
  moshi::action::effect_load_voice_embedding_frame_bf16{}(prefill_run, ctx);
  CHECK(prefill_ctx.embedding_frame_ok);
  CHECK(prefill_ctx.embedding_frame[0] != 0.0f);
  CHECK(prefill_ctx.embedding_frame[1] != 0.0f);
}

TEST_CASE("speech_moshi_generator_prefills_personaplex_system_prompt_before_"
          "generation") {
  auto fixture = load_fixture_or_skip("moshi-tiny-lm.gguf");
  auto voice = load_fixture_or_skip("moshi-tiny-voice.gguf");
  if (fixture.model == nullptr || voice.model == nullptr) {
    return;
  }

  emel::memory::test::recording_kv_actor kv{};
  emel::memory::hybrid::sm memory{emel::memory::hybrid::bind_kv_actor(kv)};
  recording_graph_executor graph{};
  moshi::sm generator{make_predictor_dependencies(graph, memory)};
  emel::error::type err = k_no_error;

  moshi::event::initialize init{*fixture.model};
  configure_predictor_initialize(init);
  init.max_blocks = 48;
  init.block_tokens = 4;
  init.error_out = &err;
  REQUIRE(generator.process_event(init));
  CHECK(err == k_no_error);

  moshi::event::load_voice load{*voice.model};
  load.error_out = &err;
  REQUIRE(generator.process_event(load));
  CHECK(err == k_no_error);

  const int32_t prompt_frames = 5;
  for (int32_t index = 0; index < prompt_frames; ++index) {
    bool complete = false;
    int32_t remaining = -1;
    moshi::event::prefill_voice prefill{};
    prefill.error_out = &err;
    prefill.complete_out = &complete;
    prefill.remaining_frames_out = &remaining;
    REQUIRE(generator.process_event(prefill));
    CHECK(err == k_no_error);
    CHECK(complete == (index + 1 == prompt_frames));
    CHECK(remaining == prompt_frames - index - 1);
  }

  const int32_t needed_tokens = (fixture.model->moshi_lm.n_q + 1) -
                                fixture.model->moshi_lm.inference_dep_q - 1;
  REQUIRE(needed_tokens >= 0);
  std::array<int32_t, moshi::event::k_max_codebooks> input = {};
  std::array<int32_t, moshi::event::k_max_codebooks> output = {};
  moshi::event::predict::workspace workspace{};
  std::fill(input.begin(), input.end(), 0);
  int32_t text_token = -1;
  err = k_no_error;
  moshi::event::predict blocked_predict{
      std::span<const int32_t>{
          input.data(), static_cast<size_t>(fixture.model->moshi_lm.n_q + 1)},
      workspace, 1, 1};
  blocked_predict.error_out = &err;
  CHECK_FALSE(generator.process_event(blocked_predict));
  CHECK(err == emel::error::cast(moshi::error::voice_prompt_pending));

  err = k_no_error;
  moshi::event::begin_personaplex_prompt begin{};
  begin.text_token_count = 2;
  begin.pre_text_silence_frames = 1;
  begin.post_text_silence_frames = 1;
  begin.error_out = &err;
  REQUIRE(generator.process_event(begin));
  CHECK(err == k_no_error);

  const std::array<int32_t, 4> prompt_text_tokens = {-1, 42, 43, -1};
  const std::array<int32_t, 4> expected_text_inputs = {3, 42, 43, 3};
  for (int32_t index = 0; index < 4; ++index) {
    bool complete = true;
    int32_t remaining = -1;
    emel::error::type graph_err = emel::error::cast(moshi::error::none);
    err = k_no_error;
    moshi::event::prefill_personaplex_prompt prefill{};
    prefill.text_token = prompt_text_tokens[static_cast<size_t>(index)];
    prefill.error_out = &err;
    prefill.graph_error_out = &graph_err;
    prefill.complete_out = &complete;
    prefill.remaining_frames_out = &remaining;
    REQUIRE(generator.process_event(prefill));
    CHECK(err == k_no_error);
    CHECK(graph_err == k_no_error);
    CHECK(complete == (index == 3));
    CHECK(remaining == 3 - index);
    CHECK(graph.last_input[0] ==
          expected_text_inputs[static_cast<size_t>(index)]);
    CHECK(graph.last_forced_text_token == -1);
  }

  CHECK(graph.external_embedding_count == prompt_frames);
  CHECK(graph.call_count == prompt_frames + 4);
  CHECK(kv.allocate_slots_count >= prompt_frames + 4);
  CHECK(kv.capture_view_count >= prompt_frames + 4);

  text_token = -1;
  err = k_no_error;
  emel::error::type graph_err = emel::error::cast(moshi::error::graph_runtime);
  moshi::event::predict first_predict{
      std::span<const int32_t>{
          input.data(), static_cast<size_t>(fixture.model->moshi_lm.n_q + 1)},
      workspace, 1, 1};
  first_predict.error_out = &err;
  REQUIRE(generator.process_event(first_predict));
  CHECK(generator.is(stateforward::sml::state<moshi::state_prediction_ready>));
  moshi::event::execute first_execute{
      workspace,
      std::span<const int32_t>{
          input.data(), static_cast<size_t>(fixture.model->moshi_lm.n_q + 1)}};
  first_execute.error_out = &err;
  first_execute.graph_error_out = &graph_err;
  REQUIRE(generator.process_event(first_execute));
  CHECK(generator.is(stateforward::sml::state<moshi::state_execution_ready>));
  moshi::event::sample undersized_sample{
      workspace,
      std::span<const int32_t>{
          input.data(), static_cast<size_t>(fixture.model->moshi_lm.n_q + 1)},
      std::span<int32_t>{output.data(), static_cast<size_t>(
                                            fixture.model->moshi_lm.dep_q - 1)},
      text_token};
  undersized_sample.error_out = &err;
  CHECK_FALSE(generator.process_event(undersized_sample));
  CHECK(err == emel::error::cast(moshi::error::request_shape));
  CHECK(generator.is(stateforward::sml::state<moshi::state_execution_ready>));
  err = k_no_error;
  moshi::event::sample first_sample{
      workspace,
      std::span<const int32_t>{
          input.data(), static_cast<size_t>(fixture.model->moshi_lm.n_q + 1)},
      std::span<int32_t>{output.data(),
                         static_cast<size_t>(fixture.model->moshi_lm.dep_q)},
      text_token};
  first_sample.error_out = &err;
  first_sample.graph_error_out = &graph_err;
  REQUIRE(generator.process_event(first_sample));
  CHECK(generator.is(stateforward::sml::state<moshi::state_session_ready>));
  CHECK(err == k_no_error);
  CHECK(graph_err == k_no_error);
  CHECK(graph.call_count == prompt_frames + 6);
}

TEST_CASE("speech_moshi_generator_accepts_personaplex_voice_without_system_"
          "prompt") {
  auto fixture = load_fixture_or_skip("moshi-tiny-lm.gguf");
  auto voice = load_fixture_or_skip("moshi-tiny-voice.gguf");
  if (fixture.model == nullptr || voice.model == nullptr) {
    return;
  }

  emel::memory::test::recording_kv_actor kv{};
  emel::memory::hybrid::sm memory{emel::memory::hybrid::bind_kv_actor(kv)};
  recording_graph_executor graph{};
  moshi::sm generator{make_predictor_dependencies(graph, memory)};
  emel::error::type err = k_no_error;

  moshi::event::initialize init{*fixture.model};
  configure_predictor_initialize(init);
  init.max_blocks = 48;
  init.block_tokens = 4;
  init.error_out = &err;
  REQUIRE(generator.process_event(init));
  CHECK(err == k_no_error);

  moshi::event::load_voice load{*voice.model};
  load.error_out = &err;
  REQUIRE(generator.process_event(load));
  CHECK(err == k_no_error);

  const int32_t prompt_frames = 5;
  for (int32_t index = 0; index < prompt_frames; ++index) {
    bool complete = false;
    moshi::event::prefill_voice prefill{};
    prefill.error_out = &err;
    prefill.complete_out = &complete;
    REQUIRE(generator.process_event(prefill));
    CHECK(err == k_no_error);
    CHECK(complete == (index + 1 == prompt_frames));
  }

  moshi::event::begin_personaplex_prompt begin{};
  begin.error_out = &err;
  err = k_no_error;
  REQUIRE(generator.process_event(begin));
  CHECK(err == k_no_error);

  const int32_t needed_tokens = (fixture.model->moshi_lm.n_q + 1) -
                                fixture.model->moshi_lm.inference_dep_q - 1;
  REQUIRE(needed_tokens >= 0);
  std::array<int32_t, moshi::event::k_max_codebooks> input = {};
  std::array<int32_t, moshi::event::k_max_codebooks> output = {};
  moshi::event::predict::workspace workspace{};
  std::fill(input.begin(), input.end(), 0);
  int32_t text_token = -1;
  emel::error::type graph_err = emel::error::cast(moshi::error::graph_runtime);
  moshi::event::predict predict{
      std::span<const int32_t>{
          input.data(), static_cast<size_t>(fixture.model->moshi_lm.n_q + 1)},
      workspace, 1, 1};
  predict.error_out = &err;

  err = k_no_error;
  CHECK_FALSE(generator.process_event(predict));
  CHECK(err == emel::error::cast(moshi::error::voice_prompt_pending));

  const int32_t personaplex_prompt_frames =
      fixture.model->moshi_lm.inference_pre_text_silence_frames +
      fixture.model->moshi_lm.inference_post_text_silence_frames;
  for (int32_t index = 0; index < personaplex_prompt_frames; ++index) {
    bool complete = true;
    int32_t remaining = -1;
    graph_err = emel::error::cast(moshi::error::graph_runtime);
    err = k_no_error;
    moshi::event::prefill_personaplex_prompt prefill{};
    prefill.text_token = -1;
    prefill.error_out = &err;
    prefill.graph_error_out = &graph_err;
    prefill.complete_out = &complete;
    prefill.remaining_frames_out = &remaining;

    REQUIRE(generator.process_event(prefill));
    CHECK(err == k_no_error);
    CHECK(graph_err == k_no_error);
    CHECK(complete == (index + 1 == personaplex_prompt_frames));
    CHECK(remaining == personaplex_prompt_frames - index - 1);
    CHECK(graph.last_input[0] == fixture.model->moshi_lm.text_padding_id);
    CHECK(graph.last_forced_text_token == -1);
  }

  err = k_no_error;
  graph_err = emel::error::cast(moshi::error::graph_runtime);
  REQUIRE(generator.process_event(predict));
  moshi::event::execute execute{
      workspace,
      std::span<const int32_t>{
          input.data(), static_cast<size_t>(fixture.model->moshi_lm.n_q + 1)}};
  execute.error_out = &err;
  execute.graph_error_out = &graph_err;
  REQUIRE(generator.process_event(execute));
  moshi::event::sample sample{
      workspace,
      std::span<const int32_t>{
          input.data(), static_cast<size_t>(fixture.model->moshi_lm.n_q + 1)},
      std::span<int32_t>{output.data(),
                         static_cast<size_t>(fixture.model->moshi_lm.dep_q)},
      text_token};
  sample.error_out = &err;
  sample.graph_error_out = &graph_err;
  REQUIRE(generator.process_event(sample));
  CHECK(err == k_no_error);
  CHECK(graph_err == k_no_error);
  CHECK(graph.call_count == prompt_frames + personaplex_prompt_frames + 2);
}

TEST_CASE("speech_moshi_predictor_reports_rejected_graph_event") {
  auto fixture = load_fixture_or_skip("moshi-tiny-lm.gguf");
  if (fixture.model == nullptr) {
    return;
  }

  emel::memory::test::recording_kv_actor kv{};
  emel::memory::hybrid::sm memory{emel::memory::hybrid::bind_kv_actor(kv)};
  rejecting_graph_executor graph{};
  moshi::sm generator{make_predictor_dependencies(graph, memory)};
  emel::error::type err = k_no_error;
  moshi::event::initialize init{*fixture.model};
  configure_predictor_initialize(init);
  init.error_out = &err;
  REQUIRE(generator.process_event(init));

  std::array<int32_t, moshi::event::k_max_codebooks> input = {};
  moshi::event::predict::workspace workspace{};
  moshi::event::execute premature_execute{
      workspace,
      std::span<const int32_t>{
          input.data(), static_cast<size_t>(fixture.model->moshi_lm.n_q + 1)}};
  premature_execute.error_out = &err;
  CHECK_FALSE(generator.process_event(premature_execute));
  CHECK(err == emel::error::cast(moshi::error::request_shape));

  std::array<int32_t, moshi::event::k_max_codebooks> premature_output = {};
  int32_t premature_text_token = -1;
  err = k_no_error;
  moshi::event::sample premature_sample{
      workspace,
      std::span<const int32_t>{
          input.data(), static_cast<size_t>(fixture.model->moshi_lm.n_q + 1)},
      std::span<int32_t>{premature_output.data(),
                         static_cast<size_t>(fixture.model->moshi_lm.dep_q)},
      premature_text_token};
  premature_sample.error_out = &err;
  CHECK_FALSE(generator.process_event(premature_sample));
  CHECK(err == emel::error::cast(moshi::error::request_shape));

  err = k_no_error;
  moshi::event::predict predict{
      std::span<const int32_t>{
          input.data(), static_cast<size_t>(fixture.model->moshi_lm.n_q + 1)},
      workspace, 1, 1};
  predict.error_out = &err;

  REQUIRE(generator.process_event(predict));
  moshi::event::execute execute{
      workspace,
      std::span<const int32_t>{
          input.data(), static_cast<size_t>(fixture.model->moshi_lm.n_q + 1)}};
  execute.error_out = &err;

  CHECK_FALSE(generator.process_event(execute));
  CHECK(err == emel::error::cast(moshi::error::graph_runtime));
  CHECK(kv.allocate_slots_count == 1);
  CHECK(kv.capture_view_count == 1);
}

TEST_CASE("speech_moshi_executor_initializes_lm_fixture") {
  auto fixture = load_fixture_or_skip("moshi-tiny-lm.gguf");
  if (fixture.model == nullptr) {
    return;
  }

  emel::kernel::sm kernel{};
  emel::logits::sampler::sm sampler{};
  moshi_executor::sm executor{make_executor_dependencies(kernel, {}, &sampler)};
  emel::error::type err = emel::error::cast(moshi_executor::error::none);
  moshi_executor::event::initialize init{*fixture.model};
  init.error_out = &err;

  REQUIRE(executor.process_event(init));
  CHECK(err == emel::error::cast(moshi_executor::error::none));
}

TEST_CASE("speech_moshi_executor_rejects_missing_numeric_policy") {
  auto fixture = load_fixture_or_skip("moshi-tiny-lm.gguf");
  if (fixture.model == nullptr) {
    return;
  }

  emel::kernel::sm kernel{};
  moshi_executor::sm executor{moshi_executor::dependencies{.kernel = kernel}};
  emel::error::type err = emel::error::cast(moshi_executor::error::none);
  moshi_executor::event::initialize init{*fixture.model};
  init.error_out = &err;

  CHECK_FALSE(executor.process_event(init));
  CHECK(err == emel::error::cast(moshi_executor::error::bind_failed));
}

TEST_CASE("speech_moshi_executor_rejects_missing_capacity") {
  auto fixture = load_fixture_or_skip("moshi-tiny-lm.gguf");
  if (fixture.model == nullptr) {
    return;
  }

  emel::kernel::sm kernel{};
  auto dependencies = make_executor_dependencies(kernel);
  dependencies.capacity.hidden_dim = 0u;
  moshi_executor::sm executor{dependencies};
  emel::error::type err = emel::error::cast(moshi_executor::error::none);
  moshi_executor::event::initialize init{*fixture.model};
  init.error_out = &err;

  CHECK_FALSE(executor.process_event(init));
  CHECK(err == emel::error::cast(moshi_executor::error::bind_failed));
}

TEST_CASE("speech_moshi_executor_models_zero_seed_and_top_k_clamp") {
  auto fixture = load_fixture_or_skip("moshi-tiny-lm.gguf");
  if (fixture.model == nullptr) {
    return;
  }

  emel::kernel::sm kernel{};
  emel::logits::sampler::sm sampler{};
  moshi_executor::sm executor{make_executor_dependencies(kernel, {}, &sampler)};
  emel::error::type err = emel::error::cast(moshi_executor::error::none);
  moshi_executor::event::initialize init{*fixture.model};
  init.sampling_enabled = true;
  init.sampling_audio_temperature = 0.8f;
  init.sampling_text_temperature = 0.7f;
  init.sampling_audio_top_k = 1;
  init.sampling_text_top_k = fixture.model->moshi_lm.text_card + 1;
  init.sampling_seed = 0u;
  init.error_out = &err;

  REQUIRE(executor.process_event(init));
  CHECK(err == emel::error::cast(moshi_executor::error::none));
}

TEST_CASE("speech_moshi_executor_normalizes_nonzero_sampling_seed") {
  auto fixture = load_fixture_or_skip("moshi-tiny-lm.gguf");
  if (fixture.model == nullptr) {
    return;
  }

  emel::kernel::sm kernel{};
  auto dependencies = make_executor_dependencies(kernel);
  moshi_executor::action::context ctx{dependencies};
  moshi_executor::event::initialize init{*fixture.model};
  init.sampling_seed = 2147483647u;
  moshi_executor::event::initialize_ctx init_ctx{};
  moshi_executor::event::initialize_run init_run{init, init_ctx};

  moshi_executor::action::effect_bind_nonzero_sampling_seed{}(init_run, ctx);
  CHECK(ctx.sampling.random_state > 0u);
  CHECK(ctx.sampling.random_state < 2147483647u);
}

TEST_CASE("speech_moshi_executor_rejects_invalid_sampling_configuration") {
  auto fixture = load_fixture_or_skip("moshi-tiny-lm.gguf");
  if (fixture.model == nullptr) {
    return;
  }

  const auto rejected = [&fixture](const float audio_temperature,
                                   const float text_temperature,
                                   const int32_t audio_top_k,
                                   const int32_t text_top_k) {
    emel::kernel::sm kernel{};
    emel::logits::sampler::sm sampler{};
    moshi_executor::sm executor{
        make_executor_dependencies(kernel, {}, &sampler)};
    emel::error::type err = emel::error::cast(moshi_executor::error::none);
    moshi_executor::event::initialize init{*fixture.model};
    init.sampling_enabled = true;
    init.sampling_audio_temperature = audio_temperature;
    init.sampling_text_temperature = text_temperature;
    init.sampling_audio_top_k = audio_top_k;
    init.sampling_text_top_k = text_top_k;
    init.sampling_seed = 1234u;
    init.error_out = &err;
    return !executor.process_event(init) &&
           err == emel::error::cast(moshi_executor::error::bind_failed);
  };

  CHECK(rejected(std::numeric_limits<float>::quiet_NaN(), 0.7f, 1, 1));
  CHECK(rejected(0.8f, std::numeric_limits<float>::quiet_NaN(), 1, 1));
  CHECK(rejected(0.8f, 0.7f, 0, 1));
  CHECK(rejected(0.8f, 0.7f, 1, 0));
}

TEST_CASE("speech_moshi_machines_report_unexpected_events_after_init") {
  auto fixture = load_fixture_or_skip("moshi-tiny-lm.gguf");
  if (fixture.model == nullptr) {
    return;
  }

  emel::kernel::sm kernel{};
  moshi_executor::sm executor{make_executor_dependencies(kernel)};
  emel::error::type executor_err = k_no_error;
  moshi_executor::event::initialize executor_init{*fixture.model};
  executor_init.error_out = &executor_err;
  REQUIRE(executor.process_event(executor_init));
  executor_err = k_no_error;
  CHECK_FALSE(executor.process_event(executor_init));
  CHECK(executor_err ==
        emel::error::cast(moshi_executor::error::unexpected_event));

  emel::memory::hybrid::sm memory{};
  recording_graph_executor graph{};
  moshi::sm predictor{make_predictor_dependencies(graph, memory)};
  emel::error::type predictor_err = k_no_error;
  moshi::event::initialize predictor_init{*fixture.model};
  configure_predictor_initialize(predictor_init);
  predictor_init.error_out = &predictor_err;
  REQUIRE(predictor.process_event(predictor_init));
  predictor_err = k_no_error;
  CHECK_FALSE(predictor.process_event(predictor_init));
  CHECK(predictor_err == emel::error::cast(moshi::error::unexpected_event));
}

TEST_CASE("speech_moshi_executor_requires_sampler_for_sampling") {
  auto fixture = load_fixture_or_skip("moshi-tiny-lm.gguf");
  if (fixture.model == nullptr) {
    return;
  }

  emel::kernel::sm kernel{};
  moshi_executor::sm executor{make_executor_dependencies(kernel)};
  emel::error::type err = emel::error::cast(moshi_executor::error::none);
  moshi_executor::event::initialize init{*fixture.model};
  init.sampling_enabled = true;
  init.sampling_audio_temperature = 0.8f;
  init.sampling_text_temperature = 0.7f;
  init.sampling_audio_top_k = 250;
  init.sampling_text_top_k = 25;
  init.sampling_seed = 1234u;
  init.error_out = &err;

  CHECK_FALSE(executor.process_event(init));
  CHECK(err == emel::error::cast(moshi_executor::error::bind_failed));
}

TEST_CASE("speech_moshi_executor_sampling_matches_reference_probability_path") {
  std::array<float, 2048> logits = {};
  uint32_t logit_state = 1153427707u;
  for (float &logit : logits) {
    logit_state = static_cast<uint32_t>(
        (static_cast<uint64_t>(logit_state) * 16807u) % 2147483647u);
    logit = (static_cast<float>(logit_state) / 2147483647.0f - 0.5f) * 16.0f;
  }
  std::array<float, 250> top_probabilities = {};
  std::array<int32_t, 250> top_indices = {};
  std::array<int32_t, 2048> sorted_indices = {};
  uint32_t random_state = 1234u;
  int32_t best_index = -1;
  float best_score = 0.0f;

  emel::logits::sampler::sm sampler{};
  emel::error::type err = emel::error::cast(emel::logits::sampler::error::none);
  emel::logits::sampler::event::sample_temperature_top_k request{
      logits,
      2048,
      0.8f,
      250,
      sorted_indices,
      top_probabilities,
      top_indices,
      random_state,
      best_index,
      best_score,
      err};

  REQUIRE(sampler.process_event(request));
  CHECK(err == emel::error::cast(emel::logits::sampler::error::none));

  CHECK(best_index == 1582);
  CHECK(random_state == 1237704734u);
}

TEST_CASE("speech_moshi_executor_rejects_graph_step_before_initialize") {
  auto fixture = load_fixture_or_skip("moshi-tiny-lm.gguf");
  if (fixture.model == nullptr) {
    return;
  }

  emel::kernel::sm kernel{};
  moshi_executor::sm executor{make_executor_dependencies(kernel)};
  std::array<int32_t, 4> input = {0, 1, 2, 3};
  std::array<int32_t, 4> output = {};
  int32_t text_token = -1;
  emel::error::type err = emel::error::cast(moshi_executor::error::none);
  emel::memory::view::snapshot snapshot{};

  moshi::event::graph_step step{*fixture.model, snapshot,
                                std::span<const int32_t>{input},
                                std::span<int32_t>{output}, text_token};
  step.error_out = &err;

  CHECK_FALSE(executor.process_event(step));
  CHECK(err == emel::error::cast(moshi_executor::error::not_initialized));
}

TEST_CASE("speech_moshi_executor_runtime_choice_uses_explicit_transitions") {
  const auto bytes =
      read_binary_file(repo_root() / "src" / "emel" / "speech" / "predictor" /
                       "moshi" / "executor" / "sm.hpp");
  const std::string source{bytes.begin(), bytes.end()};
  const auto action_bytes =
      read_binary_file(repo_root() / "src" / "emel" / "speech" / "predictor" /
                       "moshi" / "executor" / "actions.hpp");
  const std::string actions_source{action_bytes.begin(), action_bytes.end()};
  const auto detail_bytes =
      read_binary_file(repo_root() / "src" / "emel" / "speech" / "predictor" /
                       "moshi" / "executor" / "detail.hpp");
  const std::string detail_source{detail_bytes.begin(), detail_bytes.end()};
  const auto predictor_guard_bytes =
      read_binary_file(repo_root() / "src" / "emel" / "speech" /
                       "predictor" / "moshi" / "guards.hpp");
  const std::string predictor_guards_source{predictor_guard_bytes.begin(),
                                            predictor_guard_bytes.end()};

  CHECK(source.find("infer_debug_phase") == std::string::npos);
  CHECK(source.find("debug_phase_out != nullptr") == std::string::npos);
  CHECK(source.find("effect_store_debug_phase_out") == std::string::npos);
  CHECK(source.find("if (ctx.depformer_logits_ok)") == std::string::npos);
  CHECK(actions_source.find("if (runtime_ev.ctx.temporal_out_norm_ok)") ==
        std::string::npos);
  CHECK(actions_source.find("add_scaled_embedding") == std::string::npos);
  CHECK(actions_source.find("if (!ctx.kernel.process_event(rows_ev))") ==
        std::string::npos);
  CHECK(actions_source.find("if (!ctx.kernel.process_event(projection_ev))") ==
        std::string::npos);
  CHECK(actions_source.find("if (!detail::bind_tensor_view") ==
        std::string::npos);
  CHECK(actions_source.find("if (runtime_ev.ctx.depformer_kv_bound") ==
        std::string::npos);
  CHECK(actions_source.find("(void)detail::bind_tensor_view") ==
        std::string::npos);
  CHECK(actions_source.find("effect_bind_input_text_embedding") !=
        std::string::npos);
  CHECK(actions_source.find("effect_run_embedding_row_fetch") !=
        std::string::npos);
  CHECK(actions_source.find("effect_apply_input_text_embedding_row") !=
        std::string::npos);
  CHECK(actions_source.find("effect_bind_input_audio_embedding") !=
        std::string::npos);
  CHECK(actions_source.find("effect_bind_depformer_text_input_projection") !=
        std::string::npos);
  CHECK(actions_source.find("effect_bind_depformer_audio_input_projection") !=
        std::string::npos);
  CHECK(actions_source.find("effect_run_depformer_input_projection") !=
        std::string::npos);
  CHECK(actions_source.find("effect_bind_depformer_text_input_embedding") !=
        std::string::npos);
  CHECK(actions_source.find("effect_bind_depformer_audio_input_embedding") !=
        std::string::npos);
  CHECK(actions_source.find("effect_bind_temporal_layer_projection") !=
        std::string::npos);
  CHECK(actions_source.find("effect_bind_temporal_layer_out_projection") !=
        std::string::npos);
  CHECK(actions_source.find("effect_bind_depformer_layer_projection") !=
        std::string::npos);
  CHECK(actions_source.find("effect_bind_depformer_layer_out_projection") !=
        std::string::npos);
  CHECK(actions_source.find("effect_bind_temporal_layer_gating_in") !=
        std::string::npos);
  CHECK(actions_source.find("effect_bind_temporal_layer_gating_out") !=
        std::string::npos);
  CHECK(actions_source.find("effect_bind_text_token_logits") !=
        std::string::npos);
  CHECK(actions_source.find("effect_bind_depformer_layer_gating_in") !=
        std::string::npos);
  CHECK(actions_source.find("effect_bind_depformer_layer_gating_out") !=
        std::string::npos);
  CHECK(actions_source.find("effect_bind_depformer_token_logits") !=
        std::string::npos);
  CHECK(source.find("state_sampling_seed_decision") != std::string::npos);
  CHECK(source.find("state_text_sampling_top_k_decision") != std::string::npos);
  CHECK(source.find("state_audio_sampling_top_k_decision") !=
        std::string::npos);
  CHECK(source.find("guard_sampling_seed_zero") != std::string::npos);
  CHECK(actions_source.find("effect_bind_zero_sampling_seed") !=
        std::string::npos);
  CHECK(detail_source.find("random_state == 0u") == std::string::npos);
  CHECK(predictor_guards_source.find("const int64_t row_count") !=
        std::string::npos);
  CHECK(predictor_guards_source.find("const int64_t frame_count") !=
        std::string::npos);
  CHECK(predictor_guards_source.find("const int64_t default_frame_count") !=
        std::string::npos);
  CHECK(predictor_guards_source.find("embeddings->dims[0] <=") !=
        std::string::npos);
  CHECK(actions_source.find("effect_publish_temporal_out_norm") !=
        std::string::npos);
  CHECK(actions_source.find("struct effect_run_temporal_layer_norm {") ==
        std::string::npos);
  CHECK(actions_source.find("struct effect_run_temporal_layer_norm2 {") ==
        std::string::npos);
  CHECK(actions_source.find("effect_run_temporal_layer_norm_rms") !=
        std::string::npos);
  CHECK(actions_source.find("effect_run_temporal_layer_norm_scale") !=
        std::string::npos);
  CHECK(actions_source.find("effect_run_temporal_layer_norm2_rms") !=
        std::string::npos);
  CHECK(actions_source.find("effect_run_temporal_layer_norm2_scale") !=
        std::string::npos);
  CHECK(actions_source.find("struct effect_run_temporal_out_norm {") ==
        std::string::npos);
  CHECK(actions_source.find("effect_run_temporal_out_norm_rms") !=
        std::string::npos);
  CHECK(actions_source.find("effect_run_temporal_out_norm_scale") !=
        std::string::npos);
  CHECK(actions_source.find("effect_apply_temporal_layer_silu_gate") ==
        std::string::npos);
  CHECK(actions_source.find("effect_apply_depformer_layer_silu_gate") ==
        std::string::npos);
  CHECK(actions_source.find("if (!ctx.kernel.process_event(silu_ev))") ==
        std::string::npos);
  CHECK(actions_source.find("effect_run_temporal_layer_silu_gate_silu") !=
        std::string::npos);
  CHECK(actions_source.find("effect_run_temporal_layer_silu_gate_mul") !=
        std::string::npos);
  CHECK(actions_source.find("effect_run_depformer_layer_silu_gate_silu") !=
        std::string::npos);
  CHECK(actions_source.find("effect_run_depformer_layer_silu_gate_mul") !=
        std::string::npos);
  CHECK(actions_source.find("struct effect_run_depformer_layer_norm {") ==
        std::string::npos);
  CHECK(actions_source.find("struct effect_run_depformer_layer_norm2 {") ==
        std::string::npos);
  CHECK(actions_source.find("effect_run_depformer_layer_norm_rms") !=
        std::string::npos);
  CHECK(actions_source.find("effect_run_depformer_layer_norm_scale") !=
        std::string::npos);
  CHECK(actions_source.find("effect_run_depformer_layer_norm2_rms") !=
        std::string::npos);
  CHECK(actions_source.find("effect_run_depformer_layer_norm2_scale") !=
        std::string::npos);
  CHECK(source.find("state_temporal_layer_norm_rms_result_decision") !=
        std::string::npos);
  CHECK(source.find("state_input_text_embedding_bind_result_decision") !=
        std::string::npos);
  CHECK(source.find("state_input_audio_embedding_bind_result_decision") !=
        std::string::npos);
  CHECK(source.find("state_depformer_input_projection_bind_result_decision") !=
        std::string::npos);
  CHECK(source.find("state_depformer_input_embedding_bind_result_decision") !=
        std::string::npos);
  CHECK(source.find("state_temporal_layer_projection_bind_result_decision") !=
        std::string::npos);
  CHECK(
      source.find("state_temporal_layer_out_projection_bind_result_decision") !=
      std::string::npos);
  CHECK(source.find("state_depformer_layer_projection_bind_result_decision") !=
        std::string::npos);
  CHECK(source.find(
            "state_depformer_layer_out_projection_bind_result_decision") !=
        std::string::npos);
  CHECK(source.find("state_temporal_layer_gating_in_bind_result_decision") !=
        std::string::npos);
  CHECK(source.find("state_temporal_layer_gating_out_bind_result_decision") !=
        std::string::npos);
  CHECK(source.find("state_text_logits_bind_result_decision") !=
        std::string::npos);
  CHECK(source.find("state_depformer_layer_gating_in_bind_result_decision") !=
        std::string::npos);
  CHECK(source.find("state_depformer_layer_gating_out_bind_result_decision") !=
        std::string::npos);
  CHECK(source.find("state_depformer_logits_bind_result_decision") !=
        std::string::npos);
  CHECK(source.find("state_temporal_layer_norm2_rms_result_decision") !=
        std::string::npos);
  CHECK(source.find("state_temporal_out_norm_rms_result_decision") !=
        std::string::npos);
  CHECK(source.find("state_temporal_layer_silu_gate_silu_result_decision") !=
        std::string::npos);
  CHECK(source.find("state_depformer_layer_silu_gate_silu_result_decision") !=
        std::string::npos);
  CHECK(source.find("state_depformer_layer_norm_rms_result_decision") !=
        std::string::npos);
  CHECK(source.find("state_depformer_layer_norm2_rms_result_decision") !=
        std::string::npos);
  CHECK(source.find("guard_temporal_layer_norm_rms_succeeded") !=
        std::string::npos);
  CHECK(source.find("guard_embedding_view_bound") != std::string::npos);
  CHECK(source.find("guard_embedding_row_succeeded") != std::string::npos);
  CHECK(source.find("guard_projection_view_bound") != std::string::npos);
  CHECK(source.find("guard_depformer_input_projection_bound") !=
        std::string::npos);
  CHECK(source.find("guard_depformer_text_input_projection_succeeded") !=
        std::string::npos);
  CHECK(source.find("guard_depformer_audio_input_projection_succeeded") !=
        std::string::npos);
  CHECK(source.find("guard_temporal_layer_norm2_rms_succeeded") !=
        std::string::npos);
  CHECK(source.find("guard_temporal_out_norm_rms_succeeded") !=
        std::string::npos);
  CHECK(source.find("guard_temporal_layer_silu_gate_silu_succeeded") !=
        std::string::npos);
  CHECK(source.find("guard_depformer_layer_silu_gate_silu_succeeded") !=
        std::string::npos);
  CHECK(source.find("guard_depformer_layer_norm_rms_succeeded") !=
        std::string::npos);
  CHECK(source.find("guard_depformer_layer_norm2_rms_succeeded") !=
        std::string::npos);
}

TEST_CASE("speech_moshi_generator_graph_outputs_use_explicit_transitions") {
  const auto action_bytes =
      read_binary_file(repo_root() / "src" / "emel" / "speech" / "predictor" /
                       "moshi" / "actions.hpp");
  const std::string actions_source{action_bytes.begin(), action_bytes.end()};
  const auto sm_bytes =
      read_binary_file(repo_root() / "src" / "emel" / "speech" / "predictor" /
                       "moshi" / "sm.hpp");
  const std::string sm_source{sm_bytes.begin(), sm_bytes.end()};

  CHECK(actions_source.find(
            "if (runtime_ev.request.graph_error_out != nullptr)") ==
        std::string::npos);
  CHECK(actions_source.find("graph_debug_phase_out") == std::string::npos);
  CHECK(actions_source.find("effect_store_graph_error_out") !=
        std::string::npos);
  CHECK(actions_source.find("effect_store_graph_debug_phase_out") ==
        std::string::npos);
  CHECK(sm_source.find("state_execute_graph_error_out_decision") !=
        std::string::npos);
  CHECK(sm_source.find("state_sample_error_out_decision") != std::string::npos);
  CHECK(sm_source.find("state_graph_debug_phase_out_decision") ==
        std::string::npos);
}

TEST_CASE("speech_moshi_generator_and_executor_cover_explicit_error_guards") {
  auto fixture = load_fixture_or_skip("moshi-tiny-lm.gguf");
  if (fixture.model == nullptr) {
    return;
  }

  emel::memory::hybrid::sm memory{};
  moshi::action::context generator_ctx{memory};
  moshi::event::initialize init{*fixture.model};
  configure_predictor_initialize(init);
  moshi::event::initialize_ctx init_ctx{};
  moshi::event::initialize_run init_run{init, init_ctx};
  generator_ctx.lmgen.codebook_count = fixture.model->moshi_lm.n_q + 1;
  generator_ctx.lmgen.generated_dep_q = fixture.model->moshi_lm.dep_q;
  moshi::action::effect_configure_standard_lmgen{}(init_run, generator_ctx);
  CHECK(generator_ctx.lmgen.delayed_dep_q == fixture.model->moshi_lm.dep_q);

  moshi::event::predict::workspace generator_workspace{};
  moshi::event::predict_ctx generator_predict_ctx{generator_workspace.memory};
  auto generator_snapshot = std::make_unique<emel::memory::view::snapshot>();
  std::array<int32_t, moshi::event::k_max_codebooks> generator_input = {};
  moshi::event::predict generator_predict{
      std::span<const int32_t>{generator_input.data(), 1}, generator_workspace,
      1, 1};
  moshi::event::predict_run generator_predict_run{generator_predict,
                                                  generator_predict_ctx};
  moshi::action::effect_mark_not_initialized<moshi::event::predict_run>{}(
      generator_predict_run, generator_ctx);
  CHECK(generator_predict_ctx.err ==
        emel::error::cast(moshi::error::not_initialized));
  moshi::action::effect_mark_memory_error<moshi::event::predict_run>{}(
      generator_predict_run, generator_ctx);
  CHECK(generator_predict_ctx.err == emel::error::cast(moshi::error::memory));
  moshi::action::effect_mark_step_request_invalid{}(generator_predict_run,
                                                    generator_ctx);
  CHECK(generator_predict_ctx.err ==
        emel::error::cast(moshi::error::request_shape));
  moshi::action::effect_mark_voice_contract_error<moshi::event::predict_run>{}(
      generator_predict_run, generator_ctx);
  CHECK(generator_predict_ctx.err ==
        emel::error::cast(moshi::error::voice_contract));
  moshi::action::effect_mark_personaplex_prompt_error<
      moshi::event::predict_run>{}(generator_predict_run, generator_ctx);
  CHECK(generator_predict_ctx.err ==
        emel::error::cast(moshi::error::personaplex_prompt));
  emel::error::type unexpected_err = k_no_error;
  generator_predict.error_out = &unexpected_err;
  moshi::action::effect_mark_unexpected{}(generator_predict_run, generator_ctx);
  CHECK(generator_predict_ctx.err ==
        emel::error::cast(moshi::error::unexpected_event));
  moshi::action::effect_mark_unexpected_and_store{}(generator_predict_run,
                                                    generator_ctx);
  CHECK(unexpected_err == emel::error::cast(moshi::error::unexpected_event));
  CHECK(moshi::guard::guard_memory_rejected{}(generator_predict_run,
                                              generator_ctx));
  CHECK(moshi::guard::guard_standard_lmgen{}(init_run, generator_ctx) !=
        moshi::guard::guard_personaplex_lmgen{}(init_run, generator_ctx));
  CHECK(moshi::guard::guard_graph_initialize_failed{}(init_run, generator_ctx));
  CHECK(moshi::guard::guard_predict_request_shape_invalid{}(
      generator_predict_run, generator_ctx));

  moshi::event::begin_personaplex_prompt begin_prompt{};
  moshi::event::begin_personaplex_prompt_ctx begin_prompt_ctx{};
  moshi::event::begin_personaplex_prompt_run begin_prompt_run{begin_prompt,
                                                              begin_prompt_ctx};
  CHECK(moshi::guard::guard_personaplex_prompt_begin_invalid{}(begin_prompt_run,
                                                               generator_ctx));

  auto prompt_snapshot = std::make_unique<emel::memory::view::snapshot>();
  moshi::event::prefill_personaplex_prompt prompt_prefill{};
  moshi::event::prefill_personaplex_prompt_ctx prompt_prefill_ctx{
      *prompt_snapshot};
  moshi::event::prefill_personaplex_prompt_run prompt_prefill_run{
      prompt_prefill, prompt_prefill_ctx};
  CHECK(moshi::guard::guard_personaplex_prompt_prefill_request_invalid{}(
      prompt_prefill_run, generator_ctx));
  CHECK(moshi::guard::guard_personaplex_prompt_phase_invalid{}(
      prompt_prefill_run, generator_ctx));
  CHECK(moshi::guard::guard_no_personaplex_prompt_complete_out{}(
      prompt_prefill_run, generator_ctx));
  CHECK(moshi::guard::guard_no_personaplex_prompt_remaining_out{}(
      prompt_prefill_run, generator_ctx));

  generator_ctx.session.model = fixture.model.get();
  generator_ctx.session.text_card = fixture.model->moshi_lm.text_card;
  generator_ctx.voice.loaded = true;
  generator_ctx.voice.ready = true;
  generator_ctx.voice.prompt_started = true;
  generator_ctx.voice.prompt_ready = false;
  generator_ctx.voice.text_tokens_remaining = 1;
  prompt_prefill.text_token = fixture.model->moshi_lm.text_card;
  CHECK(moshi::guard::guard_personaplex_prompt_phase_invalid{}(
      prompt_prefill_run, generator_ctx));

  moshi_callback_probe callback_probe{};
  init.on_done =
      emel::callback<void(const moshi::events::initialize_done &)>::from<
          moshi_callback_probe,
          &moshi_callback_probe::on_generator_initialize_done>(&callback_probe);
  init.on_error =
      emel::callback<void(const moshi::events::initialize_error &)>::from<
          moshi_callback_probe,
          &moshi_callback_probe::on_generator_initialize_error>(
          &callback_probe);
  generator_ctx.session.n_q = fixture.model->moshi_lm.n_q;
  generator_ctx.session.dep_q = fixture.model->moshi_lm.dep_q;
  init_ctx.err = emel::error::cast(moshi::error::bind_failed);
  moshi::action::effect_emit_initialize_done{}(init_run, generator_ctx);
  CHECK(callback_probe.n_q == fixture.model->moshi_lm.n_q);
  CHECK(callback_probe.dep_q == fixture.model->moshi_lm.dep_q);
  CHECK(callback_probe.request == &init);
  moshi::action::effect_emit_initialize_error{}(init_run, generator_ctx);
  CHECK(callback_probe.err == emel::error::cast(moshi::error::bind_failed));

  moshi::event::load_voice load_voice{*fixture.model};
  moshi::event::load_voice_ctx load_voice_ctx{};
  moshi::event::load_voice_run load_voice_run{load_voice, load_voice_ctx};
  CHECK(moshi::guard::guard_voice_contract_invalid{}(load_voice_run,
                                                     generator_ctx));
  load_voice.on_done =
      emel::callback<void(const moshi::events::load_voice_done &)>::from<
          moshi_callback_probe,
          &moshi_callback_probe::on_generator_load_voice_done>(&callback_probe);
  load_voice.on_error =
      emel::callback<void(const moshi::events::load_voice_error &)>::from<
          moshi_callback_probe,
          &moshi_callback_probe::on_generator_load_voice_error>(
          &callback_probe);
  generator_ctx.voice.prompt_frame_count = 7;
  load_voice_ctx.err = emel::error::cast(moshi::error::voice_contract);
  moshi::action::effect_emit_load_voice_done{}(load_voice_run, generator_ctx);
  CHECK(callback_probe.prompt_frames == 7);
  CHECK(callback_probe.request == &load_voice);
  moshi::action::effect_emit_load_voice_error{}(load_voice_run, generator_ctx);
  CHECK(callback_probe.err == emel::error::cast(moshi::error::voice_contract));

  moshi::event::prefill_voice prefill_voice{};
  emel::error::type generator_graph_err = k_no_error;
  bool voice_complete = false;
  int32_t voice_remaining = -1;
  prefill_voice.graph_error_out = &generator_graph_err;
  prefill_voice.complete_out = &voice_complete;
  prefill_voice.remaining_frames_out = &voice_remaining;
  prefill_voice.on_done =
      emel::callback<void(const moshi::events::prefill_voice_done &)>::from<
          moshi_callback_probe,
          &moshi_callback_probe::on_generator_prefill_voice_done>(
          &callback_probe);
  prefill_voice.on_error =
      emel::callback<void(const moshi::events::prefill_voice_error &)>::from<
          moshi_callback_probe,
          &moshi_callback_probe::on_generator_prefill_voice_error>(
          &callback_probe);
  moshi::event::prefill_voice_ctx prefill_voice_ctx{*generator_snapshot};
  prefill_voice_ctx.graph_error =
      emel::error::cast(moshi::error::graph_runtime);
  prefill_voice_ctx.complete = true;
  prefill_voice_ctx.remaining_frames = 3;
  prefill_voice_ctx.err =
      emel::error::cast(moshi::error::graph_runtime_unavailable);
  moshi::event::prefill_voice_run prefill_voice_run{prefill_voice,
                                                    prefill_voice_ctx};
  moshi::action::effect_store_graph_error_out{}(prefill_voice_run,
                                                generator_ctx);
  moshi::action::effect_store_voice_complete_out{}(prefill_voice_run,
                                                   generator_ctx);
  moshi::action::effect_store_voice_remaining_out{}(prefill_voice_run,
                                                    generator_ctx);
  CHECK(generator_graph_err == emel::error::cast(moshi::error::graph_runtime));
  CHECK(voice_complete);
  CHECK(voice_remaining == 3);
  moshi::action::effect_emit_prefill_voice_done{}(prefill_voice_run,
                                                  generator_ctx);
  CHECK(callback_probe.complete);
  CHECK(callback_probe.remaining_frames == 3);
  CHECK(callback_probe.request == &prefill_voice);
  moshi::action::effect_emit_prefill_voice_error{}(prefill_voice_run,
                                                   generator_ctx);
  CHECK(callback_probe.err ==
        emel::error::cast(moshi::error::graph_runtime_unavailable));

  begin_prompt.on_done = emel::callback<void(
      const moshi::events::begin_personaplex_prompt_done
          &)>::from<moshi_callback_probe,
                    &moshi_callback_probe::on_generator_begin_prompt_done>(
      &callback_probe);
  begin_prompt.on_error = emel::callback<void(
      const moshi::events::begin_personaplex_prompt_error
          &)>::from<moshi_callback_probe,
                    &moshi_callback_probe::on_generator_begin_prompt_error>(
      &callback_probe);
  begin_prompt_ctx.remaining_frames = 5;
  begin_prompt_ctx.err = emel::error::cast(moshi::error::personaplex_prompt);
  moshi::action::effect_emit_begin_personaplex_prompt_done{}(begin_prompt_run,
                                                             generator_ctx);
  CHECK(callback_probe.remaining_frames == 5);
  CHECK(callback_probe.request == &begin_prompt);
  moshi::action::effect_emit_begin_personaplex_prompt_error{}(begin_prompt_run,
                                                              generator_ctx);
  CHECK(callback_probe.err ==
        emel::error::cast(moshi::error::personaplex_prompt));

  bool prompt_complete = false;
  int32_t prompt_remaining = -1;
  prompt_prefill.complete_out = &prompt_complete;
  prompt_prefill.remaining_frames_out = &prompt_remaining;
  prompt_prefill.on_done = emel::callback<void(
      const moshi::events::prefill_personaplex_prompt_done
          &)>::from<moshi_callback_probe,
                    &moshi_callback_probe::on_generator_prefill_prompt_done>(
      &callback_probe);
  prompt_prefill.on_error = emel::callback<void(
      const moshi::events::prefill_personaplex_prompt_error
          &)>::from<moshi_callback_probe,
                    &moshi_callback_probe::on_generator_prefill_prompt_error>(
      &callback_probe);
  prompt_prefill_ctx.complete = true;
  prompt_prefill_ctx.remaining_frames = 2;
  prompt_prefill_ctx.err =
      emel::error::cast(moshi::error::graph_runtime_unavailable);
  moshi::action::effect_store_personaplex_prompt_complete_out{}(
      prompt_prefill_run, generator_ctx);
  moshi::action::effect_store_personaplex_prompt_remaining_out{}(
      prompt_prefill_run, generator_ctx);
  CHECK(prompt_complete);
  CHECK(prompt_remaining == 2);
  moshi::action::effect_emit_prefill_personaplex_prompt_done{}(
      prompt_prefill_run, generator_ctx);
  CHECK(callback_probe.complete);
  CHECK(callback_probe.remaining_frames == 2);
  CHECK(callback_probe.request == &prompt_prefill);
  moshi::action::effect_emit_prefill_personaplex_prompt_error{}(
      prompt_prefill_run, generator_ctx);
  CHECK(callback_probe.err ==
        emel::error::cast(moshi::error::graph_runtime_unavailable));

  const auto &lm = fixture.model->moshi_lm;
  emel::memory::view::snapshot snapshot{};
  snapshot.max_sequences = 4;
  snapshot.sequence_active[0] = 1;
  snapshot.sequence_length_values[0] = 1;
  snapshot.sequence_kv_block_count[0] = 1;
  snapshot.sequence_kv_blocks[0][0] = 0;

  emel::kernel::sm guard_kernel{};
  moshi_executor::action::context executor_ctx{
      make_executor_dependencies(guard_kernel)};
  std::array<int32_t, moshi::event::k_max_codebooks> input = {};
  std::array<int32_t, moshi::event::k_max_codebooks> output = {};
  std::array<float, moshi::event::k_max_voice_embedding_dim> embedding = {};
  input.fill(executor_ctx.policy.token_zero);
  input[0] = lm.text_padding_id;
  input[1] = 0;
  int32_t text_token = -1;
  moshi::event::graph_step step{
      *fixture.model, snapshot,
      std::span<const int32_t>{input.data(), static_cast<size_t>(lm.n_q + 1)},
      std::span<int32_t>{output.data(), static_cast<size_t>(lm.dep_q)},
      text_token};
  step.input_embedding =
      std::span<const float>{embedding.data(), static_cast<size_t>(lm.dim)};
  moshi_executor::event::step_ctx step_ctx{};
  moshi_executor::event::step_run step_run{step, step_ctx};
  executor_ctx.session.model = fixture.model.get();
  executor_ctx.session.codebook_count = lm.n_q + 1;
  executor_ctx.session.dep_q = lm.dep_q;
  executor_ctx.session.text_card = lm.text_card;
  executor_ctx.session.audio_card = lm.card;
  executor_ctx.session.hidden_dim = lm.dim;
  REQUIRE(emel::model::moshi::detail::build_execution_contract(
              *fixture.model, executor_ctx.session.contract) == k_no_error);

  moshi_executor::event::initialize executor_init{*fixture.model};
  moshi_executor::event::initialize_ctx executor_init_ctx{};
  moshi_executor::event::initialize_run executor_init_run{executor_init,
                                                          executor_init_ctx};
  CHECK(moshi_executor::guard::guard_bound_root_operands_supported{}(
      executor_init_run, executor_ctx));
  const auto *bound_text_embedding =
      executor_ctx.session.contract.lm.text_embedding.tensor;
  executor_ctx.session.contract.lm.text_embedding.tensor = nullptr;
  CHECK(moshi_executor::guard::guard_bound_root_operands_unsupported{}(
      executor_init_run, executor_ctx));
  executor_ctx.session.contract.lm.text_embedding.tensor = bound_text_embedding;
  for (int32_t layer_index = 0; layer_index < lm.num_layers; ++layer_index) {
    auto &layer = executor_ctx.session.contract.lm
                      .temporal_layers[static_cast<size_t>(layer_index)];
    const auto *projection = layer.split_input_projection.tensor != nullptr
                                 ? layer.split_input_projection.tensor
                                 : layer.fused_input_projection.tensor;
    REQUIRE(projection != nullptr);
    layer.split_input_projection.tensor = projection;
    layer.fused_input_projection.tensor = projection;
  }
  CHECK(
      moshi_executor::guard::guard_temporal_split_projection_layout_supported{}(
          executor_init_run, executor_ctx));
  moshi_executor::action::effect_bind_temporal_split_projection_layout{}(
      executor_init_run, executor_ctx);
  CHECK(executor_ctx.session.temporal_input_projections[0] ==
        executor_ctx.session.contract.lm.temporal_layers[0]
            .split_input_projection.tensor);
  for (int32_t layer_index = 0; layer_index < lm.num_layers; ++layer_index) {
    executor_ctx.session.contract.lm
        .temporal_layers[static_cast<size_t>(layer_index)]
        .split_input_projection.tensor = nullptr;
  }
  CHECK(
      moshi_executor::guard::guard_temporal_fused_projection_layout_supported{}(
          executor_init_run, executor_ctx));
  moshi_executor::action::effect_bind_temporal_fused_projection_layout{}(
      executor_init_run, executor_ctx);
  CHECK_FALSE(
      moshi_executor::guard::guard_temporal_projection_layout_unsupported{}(
          executor_init_run, executor_ctx));
  for (int32_t layer_index = 0; layer_index < lm.depformer_num_layers;
       ++layer_index) {
    for (int32_t codebook = 0; codebook < lm.dep_q; ++codebook) {
      auto &codebook_layer =
          executor_ctx.session.contract.lm
              .depformer_layers[static_cast<size_t>(layer_index)]
              .codebooks[static_cast<size_t>(codebook)];
      const auto *projection =
          codebook_layer.split_input_projection.tensor != nullptr
              ? codebook_layer.split_input_projection.tensor
              : codebook_layer.fused_input_projection.tensor;
      REQUIRE(projection != nullptr);
      codebook_layer.split_input_projection.tensor = projection;
      codebook_layer.fused_input_projection.tensor = projection;
    }
  }
  CHECK(moshi_executor::guard::
            guard_depformer_split_projection_layout_supported{}(
                executor_init_run, executor_ctx));
  moshi_executor::action::effect_bind_depformer_split_projection_layout{}(
      executor_init_run, executor_ctx);
  CHECK(executor_ctx.session.depformer_input_projections[0][0] ==
        executor_ctx.session.contract.lm.depformer_layers[0]
            .codebooks[0]
            .split_input_projection.tensor);
  for (int32_t layer_index = 0; layer_index < lm.depformer_num_layers;
       ++layer_index) {
    for (int32_t codebook = 0; codebook < lm.dep_q; ++codebook) {
      executor_ctx.session.contract.lm
          .depformer_layers[static_cast<size_t>(layer_index)]
          .codebooks[static_cast<size_t>(codebook)]
          .split_input_projection.tensor = nullptr;
    }
  }
  CHECK(moshi_executor::guard::
            guard_depformer_fused_projection_layout_supported{}(
                executor_init_run, executor_ctx));
  moshi_executor::action::effect_bind_depformer_fused_projection_layout{}(
      executor_init_run, executor_ctx);
  CHECK_FALSE(
      moshi_executor::guard::guard_depformer_projection_layout_unsupported{}(
          executor_init_run, executor_ctx));
  executor_init.on_done =
      emel::callback<void(const moshi_executor::events::initialize_done &)>::
          from<moshi_callback_probe,
               &moshi_callback_probe::on_executor_initialize_done>(
              &callback_probe);
  executor_init.on_error =
      emel::callback<void(const moshi_executor::events::initialize_error &)>::
          from<moshi_callback_probe,
               &moshi_callback_probe::on_executor_initialize_error>(
              &callback_probe);
  executor_init_ctx.err = emel::error::cast(moshi_executor::error::bind_failed);
  moshi_executor::action::effect_emit_initialize_done{}(executor_init_run,
                                                        executor_ctx);
  CHECK(callback_probe.n_q == lm.n_q);
  CHECK(callback_probe.dep_q == lm.dep_q);
  CHECK(callback_probe.request == &executor_init);
  moshi_executor::action::effect_emit_initialize_error{}(executor_init_run,
                                                         executor_ctx);
  CHECK(callback_probe.err ==
        emel::error::cast(moshi_executor::error::bind_failed));
  executor_init_ctx.err = k_no_error;
  moshi_executor::action::effect_mark_bind_failed{}(executor_init_run,
                                                    executor_ctx);
  CHECK(executor_init_ctx.err ==
        emel::error::cast(moshi_executor::error::bind_failed));

  emel::error::type executor_step_err = k_no_error;
  step.error_out = &executor_step_err;
  moshi_executor::action::effect_mark_model_mismatch{}(step_run, executor_ctx);
  CHECK(step_ctx.err ==
        emel::error::cast(moshi_executor::error::model_mismatch));
  moshi_executor::action::effect_mark_request_shape{}(step_run, executor_ctx);
  CHECK(step_ctx.err ==
        emel::error::cast(moshi_executor::error::request_shape));
  moshi_executor::action::effect_store_error_out<
      moshi_executor::event::step_run>{}(step_run, executor_ctx);
  CHECK(executor_step_err ==
        emel::error::cast(moshi_executor::error::request_shape));
  moshi_executor::action::effect_mark_unexpected_and_store{}(step_run,
                                                             executor_ctx);
  CHECK(step_ctx.err ==
        emel::error::cast(moshi_executor::error::unexpected_event));
  CHECK(executor_step_err ==
        emel::error::cast(moshi_executor::error::unexpected_event));

  CHECK(moshi_executor::guard::guard_step_model_matches{}(step_run,
                                                          executor_ctx));
  executor_ctx.session.model = nullptr;
  CHECK(moshi_executor::guard::guard_step_model_mismatch{}(step_run,
                                                           executor_ctx));
  executor_ctx.session.model = fixture.model.get();
  CHECK(
      moshi_executor::guard::guard_step_shape_valid{}(step_run, executor_ctx));
  executor_ctx.session.codebook_count += 1;
  CHECK(moshi_executor::guard::guard_step_shape_invalid{}(step_run,
                                                          executor_ctx));
  executor_ctx.session.codebook_count = lm.n_q + 1;

  CHECK(moshi_executor::guard::guard_external_input_embedding_supported{}(
      step_run, executor_ctx));
  CHECK(moshi_executor::guard::guard_input_embedding_supported{}(step_run,
                                                                 executor_ctx));
  step.input_embedding = {};
  CHECK_FALSE(moshi_executor::guard::guard_input_embedding_unsupported{}(
      step_run, executor_ctx));
  CHECK_FALSE(moshi_executor::guard::guard_token_input_embedding_unsupported{}(
      step_run, executor_ctx));
  step.input_embedding =
      std::span<const float>{embedding.data(), static_cast<size_t>(lm.dim)};

  CHECK_FALSE(moshi_executor::guard::guard_input_text_embedding_succeeded{}(
      step_run, executor_ctx));
  CHECK(moshi_executor::guard::guard_input_text_embedding_failed{}(
      step_run, executor_ctx));
  step_ctx.input_text_embedding_ok = true;
  CHECK(moshi_executor::guard::guard_input_text_embedding_succeeded{}(
      step_run, executor_ctx));
  CHECK(moshi_executor::guard::guard_embedding_view_bind_failed{}(
      step_run, executor_ctx));
  step_ctx.embedding_view_bound = true;
  CHECK(moshi_executor::guard::guard_embedding_view_bound{}(step_run,
                                                            executor_ctx));
  CHECK(moshi_executor::guard::guard_embedding_row_failed{}(step_run,
                                                            executor_ctx));
  step_ctx.embedding_row_ok = true;
  CHECK(moshi_executor::guard::guard_embedding_row_succeeded{}(step_run,
                                                               executor_ctx));
  CHECK(moshi_executor::guard::guard_projection_view_bind_failed{}(
      step_run, executor_ctx));
  step_ctx.projection_view_bound = true;
  CHECK(moshi_executor::guard::guard_projection_view_bound{}(step_run,
                                                             executor_ctx));

  step_ctx.input_audio_codebook_index = -1;
  CHECK_FALSE(moshi_executor::guard::guard_current_input_audio_token_present{}(
      step_run, executor_ctx));
  CHECK_FALSE(moshi_executor::guard::guard_current_input_audio_token_zero{}(
      step_run, executor_ctx));
  step_ctx.input_audio_codebook_index = 0;
  CHECK(moshi_executor::guard::guard_current_input_audio_token_present{}(
      step_run, executor_ctx));
  input[1] = executor_ctx.policy.token_zero;
  CHECK(moshi_executor::guard::guard_current_input_audio_token_zero{}(
      step_run, executor_ctx));
  step.input_embedding = {};
  CHECK_FALSE(
      moshi_executor::guard::guard_current_input_audio_token_unsupported{}(
          step_run, executor_ctx));
  step.input_embedding =
      std::span<const float>{embedding.data(), static_cast<size_t>(lm.dim)};

  CHECK(moshi_executor::guard::guard_input_audio_embedding_failed{}(
      step_run, executor_ctx));
  step_ctx.input_audio_embedding_ok = true;
  CHECK(moshi_executor::guard::guard_input_audio_embedding_succeeded{}(
      step_run, executor_ctx));
  CHECK(moshi_executor::guard::guard_more_input_audio_codebooks{}(
      step_run, executor_ctx));
  step_ctx.input_audio_codebook_index = lm.n_q - 1;
  CHECK(moshi_executor::guard::guard_input_audio_codebooks_complete{}(
      step_run, executor_ctx));
  CHECK(moshi_executor::guard::guard_input_embedding_failed{}(step_run,
                                                              executor_ctx));
  step_ctx.input_embedding_ok = true;
  CHECK(moshi_executor::guard::guard_input_embedding_succeeded{}(step_run,
                                                                 executor_ctx));
  CHECK(moshi_executor::guard::guard_temporal_kv_binding_missing{}(
      step_run, executor_ctx));
  CHECK(moshi_executor::guard::guard_temporal_kv_bind_failed{}(step_run,
                                                               executor_ctx));

  std::array<uint16_t, 16> key_cache = {};
  std::array<uint16_t, 16> value_cache = {};
  std::array<size_t, 1> offsets = {0};
  step_ctx.temporal_kv.key_cache = std::span<uint16_t>{key_cache};
  step_ctx.temporal_kv.value_cache = std::span<uint16_t>{value_cache};
  step_ctx.temporal_kv.layer_cache_offsets = std::span<const size_t>{offsets};
  step_ctx.temporal_kv.layer_count = 1;
  step_ctx.temporal_kv.position_capacity = 1;
  step_ctx.temporal_kv.kv_dim = lm.dim;
  step_ctx.temporal_kv_bound = true;
  CHECK(
      moshi_executor::guard::guard_temporal_kv_bound{}(step_run, executor_ctx));

  CHECK_FALSE(moshi_executor::guard::guard_temporal_layer_norm_unsupported{}(
      step_run, executor_ctx));
  CHECK(moshi_executor::guard::guard_temporal_layer_norm_rms_failed{}(
      step_run, executor_ctx));
  step_ctx.temporal_layer_norm_rms_ok = true;
  CHECK(moshi_executor::guard::guard_temporal_layer_norm_rms_succeeded{}(
      step_run, executor_ctx));
  CHECK(moshi_executor::guard::guard_temporal_layer_norm_failed{}(
      step_run, executor_ctx));
  step_ctx.temporal_layer_norm_ok = true;
  CHECK(moshi_executor::guard::guard_temporal_layer_norm_succeeded{}(
      step_run, executor_ctx));
  CHECK_FALSE(
      moshi_executor::guard::guard_temporal_layer_projection_unsupported{}(
          step_run, executor_ctx));
  CHECK(moshi_executor::guard::guard_temporal_layer_projection_failed{}(
      step_run, executor_ctx));
  step_ctx.temporal_layer_projection_ok = true;
  CHECK(moshi_executor::guard::guard_temporal_layer_projection_succeeded{}(
      step_run, executor_ctx));
  step_ctx.temporal_position.logical_position = 0;
  step_ctx.temporal_position.physical_position = 0;
  CHECK_FALSE(moshi_executor::guard::guard_temporal_layer_rope_unsupported{}(
      step_run, executor_ctx));
  CHECK(moshi_executor::guard::guard_temporal_layer_rope_failed{}(
      step_run, executor_ctx));
  step_ctx.temporal_layer_rope_ok = true;
  CHECK(moshi_executor::guard::guard_temporal_layer_rope_succeeded{}(
      step_run, executor_ctx));
  CHECK(moshi_executor::guard::guard_temporal_layer_cache_write_unsupported{}(
      step_run, executor_ctx));
  CHECK(moshi_executor::guard::guard_temporal_layer_cache_write_failed{}(
      step_run, executor_ctx));
  step_ctx.temporal_layer_cache_write_ok = true;
  CHECK(moshi_executor::guard::guard_temporal_layer_cache_write_succeeded{}(
      step_run, executor_ctx));
  CHECK(moshi_executor::guard::guard_temporal_layer_attention_unsupported{}(
      step_run, executor_ctx));
  CHECK(moshi_executor::guard::guard_temporal_layer_attention_failed{}(
      step_run, executor_ctx));
  step_ctx.temporal_layer_attention_ok = true;
  CHECK(moshi_executor::guard::guard_temporal_layer_attention_succeeded{}(
      step_run, executor_ctx));
  CHECK_FALSE(
      moshi_executor::guard::guard_temporal_layer_out_projection_unsupported{}(
          step_run, executor_ctx));
  CHECK(moshi_executor::guard::guard_temporal_layer_out_projection_failed{}(
      step_run, executor_ctx));
  step_ctx.temporal_layer_out_projection_ok = true;
  CHECK(moshi_executor::guard::guard_temporal_layer_out_projection_succeeded{}(
      step_run, executor_ctx));
  CHECK(moshi_executor::guard::guard_temporal_layer_residual_failed{}(
      step_run, executor_ctx));
  step_ctx.temporal_layer_residual_ok = true;
  CHECK(moshi_executor::guard::guard_temporal_layer_residual_succeeded{}(
      step_run, executor_ctx));
}

TEST_CASE("speech_moshi_executor_embeds_input_before_unsupported_transformer") {
  auto fixture = load_fixture_or_skip("moshi-tiny-lm.gguf");
  if (fixture.model == nullptr) {
    return;
  }

  emel::kernel::sm kernel{};
  moshi_executor::sm executor{make_executor_dependencies(kernel)};
  emel::error::type err = emel::error::cast(moshi_executor::error::none);
  moshi_executor::event::initialize init{*fixture.model};
  init.error_out = &err;
  REQUIRE(executor.process_event(init));

  std::array<int32_t, moshi::event::k_max_codebooks> input = {};
  std::array<int32_t, moshi::event::k_max_codebooks> output = {};
  input.fill(-1);
  input[0] = fixture.model->moshi_lm.text_padding_id;
  int32_t text_token = -1;
  emel::memory::view::snapshot snapshot{};
  err = emel::error::cast(moshi_executor::error::none);

  moshi::event::graph_step step{
      *fixture.model, snapshot,
      std::span<const int32_t>{
          input.data(), static_cast<size_t>(fixture.model->moshi_lm.n_q + 1)},
      std::span<int32_t>{output.data(),
                         static_cast<size_t>(fixture.model->moshi_lm.dep_q)},
      text_token};
  step.error_out = &err;

  CHECK_FALSE(executor.process_event(step));
  CHECK(err ==
        emel::error::cast(moshi_executor::error::graph_execution_unsupported));
  CHECK(text_token == -1);
}

TEST_CASE(
    "speech_moshi_executor_writes_temporal_kv_and_text_before_depformer") {
  auto fixture = load_fixture_or_skip("moshi-tiny-lm.gguf");
  if (fixture.model == nullptr) {
    return;
  }

  temporal_kv_probe probe{};
  memory_streaming::sm temporal_positions{memory_streaming::dependencies{
      .capacity = fixture.model->moshi_lm.context}};
  initialize_streaming_position(temporal_positions);
  const auto views =
      prepare_kv_views(*fixture.model, &probe, nullptr, &temporal_positions);
  emel::kernel::sm kernel{};
  moshi_executor::sm executor{make_executor_dependencies(kernel, views)};
  emel::error::type err = emel::error::cast(moshi_executor::error::none);
  moshi_executor::event::initialize init{*fixture.model};
  init.error_out = &err;
  REQUIRE(executor.process_event(init));

  std::array<int32_t, moshi::event::k_max_codebooks> input = {};
  std::array<int32_t, moshi::event::k_max_codebooks> output = {};
  input.fill(-1);
  input[0] = fixture.model->moshi_lm.text_padding_id;
  int32_t text_token = -1;
  emel::memory::view::snapshot snapshot{};
  snapshot.max_sequences = 8;
  snapshot.sequence_active[7] = 1;
  snapshot.sequence_length_values[7] = 3;
  snapshot.sequence_kv_block_count[7] = 1;
  snapshot.sequence_kv_blocks[7][0] = 0;
  err = emel::error::cast(moshi_executor::error::none);

  moshi::event::graph_step step{
      *fixture.model, snapshot,
      std::span<const int32_t>{
          input.data(), static_cast<size_t>(fixture.model->moshi_lm.n_q + 1)},
      std::span<int32_t>{output.data(),
                         static_cast<size_t>(fixture.model->moshi_lm.dep_q)},
      text_token};
  step.sequence_id = 7;
  step.error_out = &err;

  CHECK_FALSE(executor.process_event(step));
  CHECK(err ==
        emel::error::cast(moshi_executor::error::graph_execution_unsupported));
  CHECK(probe.call_count == 1);
  const size_t write_offset = 0;
  const size_t write_end =
      write_offset + static_cast<size_t>(fixture.model->moshi_lm.dim);
  REQUIRE(write_end <= probe.key_cache.size());
  CHECK(std::any_of(
      probe.key_cache.begin() + static_cast<std::ptrdiff_t>(write_offset),
      probe.key_cache.begin() + static_cast<std::ptrdiff_t>(write_end),
      [](const uint16_t value) { return value != 0u; }));
  CHECK(std::any_of(
      probe.value_cache.begin() + static_cast<std::ptrdiff_t>(write_offset),
      probe.value_cache.begin() + static_cast<std::ptrdiff_t>(write_end),
      [](const uint16_t value) { return value != 0u; }));
  const int32_t last_layer = fixture.model->moshi_lm.num_layers - 1;
  REQUIRE(last_layer >= 0);
  const size_t last_layer_write_offset =
      probe.layer_offsets[static_cast<size_t>(last_layer)] + write_offset;
  const size_t last_layer_write_end =
      last_layer_write_offset +
      static_cast<size_t>(fixture.model->moshi_lm.dim);
  REQUIRE(last_layer_write_end <= probe.key_cache.size());
  CHECK(std::any_of(probe.key_cache.begin() +
                        static_cast<std::ptrdiff_t>(last_layer_write_offset),
                    probe.key_cache.begin() +
                        static_cast<std::ptrdiff_t>(last_layer_write_end),
                    [](const uint16_t value) { return value != 0u; }));
  CHECK(std::any_of(probe.value_cache.begin() +
                        static_cast<std::ptrdiff_t>(last_layer_write_offset),
                    probe.value_cache.begin() +
                        static_cast<std::ptrdiff_t>(last_layer_write_end),
                    [](const uint16_t value) { return value != 0u; }));
  CHECK(text_token >= 0);
  CHECK(text_token < fixture.model->moshi_lm.text_card);
}

TEST_CASE("speech_moshi_executor_accepts_personaplex_voice_embedding_step") {
  auto fixture = load_fixture_or_skip("moshi-tiny-lm.gguf");
  if (fixture.model == nullptr) {
    return;
  }

  temporal_kv_probe temporal_probe{};
  depformer_kv_probe depformer_probe{};
  memory_streaming::sm temporal_positions{memory_streaming::dependencies{
      .capacity = fixture.model->moshi_lm.context}};
  memory_streaming::sm depformer_positions{memory_streaming::dependencies{
      .capacity = fixture.model->moshi_lm.depformer_context}};
  initialize_streaming_position(temporal_positions);
  initialize_streaming_position(depformer_positions);
  emel::kernel::sm kernel{};
  const auto views =
      prepare_kv_views(*fixture.model, &temporal_probe, &depformer_probe,
                       &temporal_positions, &depformer_positions);
  moshi_executor::sm executor{make_executor_dependencies(kernel, views)};
  emel::error::type err = emel::error::cast(moshi_executor::error::none);
  moshi_executor::event::initialize init{*fixture.model};
  init.error_out = &err;
  REQUIRE(executor.process_event(init));

  std::array<float, moshi::event::k_max_voice_embedding_dim> embedding = {};
  for (int32_t index = 0; index < fixture.model->moshi_lm.dim; ++index) {
    embedding[static_cast<size_t>(index)] =
        0.001f * static_cast<float>(index + 1);
  }
  std::array<int32_t, moshi::event::k_max_codebooks> input = {};
  std::array<int32_t, moshi::event::k_max_codebooks> output = {};
  input.fill(-1);
  input[0] = fixture.model->moshi_lm.text_padding_id;
  int32_t text_token = -1;
  emel::memory::view::snapshot snapshot{};
  snapshot.max_sequences = 8;
  snapshot.sequence_active[3] = 1;
  snapshot.sequence_length_values[3] = 1;
  snapshot.sequence_kv_block_count[3] = 1;
  snapshot.sequence_kv_blocks[3][0] = 0;

  moshi::event::graph_step step{
      *fixture.model, snapshot,
      std::span<const int32_t>{
          input.data(), static_cast<size_t>(fixture.model->moshi_lm.n_q + 1)},
      std::span<int32_t>{output.data(),
                         static_cast<size_t>(fixture.model->moshi_lm.dep_q)},
      text_token};
  step.sequence_id = 3;
  step.input_embedding = std::span<const float>{
      embedding.data(), static_cast<size_t>(fixture.model->moshi_lm.dim)};
  step.forced_text_token = fixture.model->moshi_lm.text_padding_id;
  step.error_out = &err;

  REQUIRE(executor.process_event(step));
  CHECK(err == emel::error::cast(moshi_executor::error::none));
  CHECK(text_token == fixture.model->moshi_lm.text_padding_id);
  CHECK(temporal_probe.call_count == 1);
  CHECK(depformer_probe.call_count == 1);
  for (int32_t index = 0; index < fixture.model->moshi_lm.dep_q; ++index) {
    CHECK(output[static_cast<size_t>(index)] >= 0);
    CHECK(output[static_cast<size_t>(index)] < fixture.model->moshi_lm.card);
  }
}

TEST_CASE("speech_moshi_executor_generates_audio_tokens_with_injected_kv") {
  auto fixture = load_fixture_or_skip("moshi-tiny-lm.gguf");
  if (fixture.model == nullptr) {
    return;
  }

  temporal_kv_probe temporal_probe{};
  depformer_kv_probe depformer_probe{};
  memory_streaming::sm temporal_positions{memory_streaming::dependencies{
      .capacity = fixture.model->moshi_lm.context}};
  memory_streaming::sm depformer_positions{memory_streaming::dependencies{
      .capacity = fixture.model->moshi_lm.depformer_context}};
  initialize_streaming_position(temporal_positions);
  initialize_streaming_position(depformer_positions);
  emel::kernel::sm kernel{};
  const auto views =
      prepare_kv_views(*fixture.model, &temporal_probe, &depformer_probe,
                       &temporal_positions, &depformer_positions);
  moshi_executor::sm executor{make_executor_dependencies(kernel, views)};
  emel::error::type err = emel::error::cast(moshi_executor::error::none);
  moshi_executor::event::initialize init{*fixture.model};
  init.error_out = &err;
  REQUIRE(executor.process_event(init));

  std::array<int32_t, moshi::event::k_max_codebooks> input = {};
  std::array<int32_t, moshi::event::k_max_codebooks> output = {};
  input.fill(-1);
  input[0] = fixture.model->moshi_lm.text_padding_id;
  int32_t text_token = -1;
  emel::memory::view::snapshot snapshot{};
  snapshot.max_sequences = 8;
  snapshot.sequence_active[7] = 1;
  snapshot.sequence_length_values[7] = 3;
  snapshot.sequence_kv_block_count[7] = 1;
  snapshot.sequence_kv_blocks[7][0] = 0;
  err = emel::error::cast(moshi_executor::error::none);

  moshi::event::graph_step step{
      *fixture.model, snapshot,
      std::span<const int32_t>{
          input.data(), static_cast<size_t>(fixture.model->moshi_lm.n_q + 1)},
      std::span<int32_t>{output.data(),
                         static_cast<size_t>(fixture.model->moshi_lm.dep_q)},
      text_token};
  step.sequence_id = 7;
  step.error_out = &err;

  CHECK(executor.process_event(step));
  CHECK(err == emel::error::cast(moshi_executor::error::none));
  CHECK(temporal_probe.call_count == 1);
  CHECK(depformer_probe.call_count == 1);
  memory_streaming::window_view depformer_window = {};
  int32_t memory_err =
      static_cast<int32_t>(emel::error::cast(memory_streaming::error::none));
  REQUIRE(
      depformer_positions.process_event(memory_streaming::event::capture_view{
          .view_out = depformer_window, .error_out = memory_err}));
  CHECK(depformer_window.logical_end == fixture.model->moshi_lm.dep_q);
  CHECK(text_token >= 0);
  CHECK(text_token < fixture.model->moshi_lm.text_card);
  for (int32_t index = 0; index < fixture.model->moshi_lm.dep_q; ++index) {
    CHECK(output[static_cast<size_t>(index)] >= 0);
    CHECK(output[static_cast<size_t>(index)] < fixture.model->moshi_lm.card);
  }
  const int32_t last_layer = fixture.model->moshi_lm.depformer_num_layers - 1;
  REQUIRE(last_layer >= 0);
  const size_t last_layer_offset =
      depformer_probe.layer_offsets[static_cast<size_t>(last_layer)];
  const size_t last_layer_end =
      last_layer_offset +
      static_cast<size_t>(fixture.model->moshi_lm.depformer_context) *
          static_cast<size_t>(fixture.model->moshi_lm.depformer_dim);
  REQUIRE(last_layer_end <= depformer_probe.key_cache.size());
  CHECK(std::any_of(depformer_probe.key_cache.begin() +
                        static_cast<std::ptrdiff_t>(last_layer_offset),
                    depformer_probe.key_cache.begin() +
                        static_cast<std::ptrdiff_t>(last_layer_end),
                    [](const uint16_t value) { return value != 0u; }));
  CHECK(std::any_of(depformer_probe.value_cache.begin() +
                        static_cast<std::ptrdiff_t>(last_layer_offset),
                    depformer_probe.value_cache.begin() +
                        static_cast<std::ptrdiff_t>(last_layer_end),
                    [](const uint16_t value) { return value != 0u; }));
}

TEST_CASE("speech_moshi_executor_sampling_rng_is_actor_owned") {
  auto first_fixture = load_fixture_or_skip("moshi-tiny-lm.gguf");
  auto second_fixture = load_fixture_or_skip("moshi-tiny-lm.gguf");
  if (first_fixture.model == nullptr || second_fixture.model == nullptr) {
    return;
  }

  temporal_kv_probe first_temporal{};
  temporal_kv_probe second_temporal{};
  depformer_kv_probe first_depformer{};
  depformer_kv_probe second_depformer{};
  memory_streaming::sm first_temporal_positions{memory_streaming::dependencies{
      .capacity = first_fixture.model->moshi_lm.context}};
  memory_streaming::sm first_depformer_positions{memory_streaming::dependencies{
      .capacity = first_fixture.model->moshi_lm.depformer_context}};
  memory_streaming::sm second_temporal_positions{memory_streaming::dependencies{
      .capacity = second_fixture.model->moshi_lm.context}};
  memory_streaming::sm second_depformer_positions{
      memory_streaming::dependencies{
          .capacity = second_fixture.model->moshi_lm.depformer_context}};
  initialize_streaming_position(first_temporal_positions);
  initialize_streaming_position(first_depformer_positions);
  initialize_streaming_position(second_temporal_positions);
  initialize_streaming_position(second_depformer_positions);
  emel::kernel::sm first_kernel{};
  emel::kernel::sm second_kernel{};
  emel::logits::sampler::sm first_sampler{};
  emel::logits::sampler::sm second_sampler{};
  const auto first_views =
      prepare_kv_views(*first_fixture.model, &first_temporal, &first_depformer,
                       &first_temporal_positions, &first_depformer_positions);
  const auto second_views = prepare_kv_views(
      *second_fixture.model, &second_temporal, &second_depformer,
      &second_temporal_positions, &second_depformer_positions);
  moshi_executor::sm first{
      make_executor_dependencies(first_kernel, first_views, &first_sampler)};
  moshi_executor::sm second{
      make_executor_dependencies(second_kernel, second_views, &second_sampler)};
  emel::error::type first_err = emel::error::cast(moshi_executor::error::none);
  emel::error::type second_err = emel::error::cast(moshi_executor::error::none);
  moshi_executor::event::initialize first_init{*first_fixture.model};
  moshi_executor::event::initialize second_init{*second_fixture.model};
  for (auto *init : {&first_init, &second_init}) {
    init->sampling_enabled = true;
    init->sampling_audio_temperature = 0.8f;
    init->sampling_text_temperature = 0.7f;
    init->sampling_audio_top_k = 250;
    init->sampling_text_top_k = 25;
    init->sampling_seed = 1234u;
  }
  first_init.error_out = &first_err;
  second_init.error_out = &second_err;
  REQUIRE(first.process_event(first_init));
  REQUIRE(second.process_event(second_init));

  std::array<int32_t, moshi::event::k_max_codebooks> input = {};
  std::array<int32_t, moshi::event::k_max_codebooks> first_output = {};
  std::array<int32_t, moshi::event::k_max_codebooks> second_output = {};
  input.fill(-1);
  input[0] = first_fixture.model->moshi_lm.text_padding_id;
  int32_t first_text_token = -1;
  int32_t second_text_token = -1;
  emel::memory::view::snapshot snapshot{};
  snapshot.max_sequences = 1;
  snapshot.sequence_active[0] = 1;
  snapshot.sequence_length_values[0] = 1;
  snapshot.sequence_kv_block_count[0] = 1;
  snapshot.sequence_kv_blocks[0][0] = 0;
  const auto input_span = std::span<const int32_t>{
      input.data(), static_cast<size_t>(first_fixture.model->moshi_lm.n_q + 1)};
  moshi::event::graph_step first_step{
      *first_fixture.model, snapshot, input_span,
      std::span<int32_t>{
          first_output.data(),
          static_cast<size_t>(first_fixture.model->moshi_lm.dep_q)},
      first_text_token};
  moshi::event::graph_step second_step{
      *second_fixture.model, snapshot, input_span,
      std::span<int32_t>{
          second_output.data(),
          static_cast<size_t>(second_fixture.model->moshi_lm.dep_q)},
      second_text_token};
  first_step.error_out = &first_err;
  second_step.error_out = &second_err;

  REQUIRE(first.process_event(first_step));
  REQUIRE(second.process_event(second_step));
  CHECK(first_text_token == second_text_token);
  CHECK(std::equal(first_output.begin(),
                   first_output.begin() + first_fixture.model->moshi_lm.dep_q,
                   second_output.begin()));
}

TEST_CASE(
    "speech_moshi_predictor_uses_emel_executor_binding_without_fake_tokens") {
  auto fixture = load_fixture_or_skip("moshi-tiny-lm.gguf");
  if (fixture.model == nullptr) {
    return;
  }

  emel::memory::test::recording_kv_actor kv{};
  emel::memory::hybrid::sm memory{emel::memory::hybrid::bind_kv_actor(kv)};
  emel::kernel::sm kernel{};
  moshi_executor::sm executor{make_executor_dependencies(kernel)};

  moshi::sm generator{make_predictor_dependencies(executor, memory)};
  emel::error::type err = k_no_error;
  moshi::event::initialize init{*fixture.model};
  configure_predictor_initialize(init);
  init.error_out = &err;
  REQUIRE(generator.process_event(init));

  std::array<int32_t, moshi::event::k_max_codebooks> input = {};
  std::array<float, moshi_executor::detail::k_max_hidden_dim> temporal_state =
      {};
  moshi::event::predict::workspace workspace{};
  workspace.temporal_state = std::span<float>{
      temporal_state.data(), static_cast<size_t>(fixture.model->moshi_lm.dim)};
  moshi::event::predict predict{
      std::span<const int32_t>{
          input.data(), static_cast<size_t>(fixture.model->moshi_lm.n_q + 1)},
      workspace, 1, 1};
  predict.error_out = &err;
  REQUIRE(generator.process_event(predict));
  moshi::event::execute execute{
      workspace,
      std::span<const int32_t>{
          input.data(), static_cast<size_t>(fixture.model->moshi_lm.n_q + 1)}};
  execute.error_out = &err;

  CHECK_FALSE(generator.process_event(execute));
  CHECK(err == emel::error::cast(moshi::error::graph_runtime));
  CHECK(kv.allocate_slots_count == 1);
  CHECK(kv.capture_view_count == 1);
  CHECK(workspace.sampled_text_token == -1);
  CHECK(workspace.sampled_audio_tokens[0] == -1);
}
