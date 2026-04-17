#include <array>
#include <cstdint>
#include <string>

#include <boost/sml.hpp>
#include "doctest/doctest.h"

#include "emel/docs/detail.hpp"
#include "emel/embeddings/generator/errors.hpp"
#include "emel/embeddings/generator/sm.hpp"
#include "emel/error/error.hpp"
#include "emel/generator/initializer/sm.hpp"
#include "emel/generator/prefill/sm.hpp"
#include "emel/sm.hpp"
#include "emel/text/conditioner/sm.hpp"
#include "emel/text/formatter/format.hpp"
#include "emel/text/tokenizer/sm.hpp"
#include "te_fixture.hpp"

namespace {

namespace te_fixture = emel::tests::embeddings::te_fixture;

using te_fixture::cached_te_fixture;
using te_fixture::initialize_embedding_generator;
using te_fixture::l2_norm;
using te_fixture::load_te_fixture;
using te_fixture::make_rgba_square;
using te_fixture::make_sine_wave;
using te_fixture::max_abs_difference;
using te_fixture::read_text_file;
using te_fixture::te_assets_present;
using te_fixture::te_prompt_path;

inline constexpr int32_t k_audio_sample_rate = 16000;
inline constexpr uint64_t k_fnv_offset_basis = 1469598103934665603ull;
inline constexpr uint64_t k_fnv_prime = 1099511628211ull;

template <class... Ts, class fn>
constexpr void for_each_type(boost::sml::aux::type_list<Ts...>, fn && visitor) {
  (visitor.template operator()<Ts>(), ...);
}

template <size_t capacity>
void check_normalized(const std::array<float, capacity> & output,
                      const int32_t dimension) {
  CHECK(l2_norm(std::span<const float>{output.data(), static_cast<size_t>(dimension)}) ==
        doctest::Approx(1.0f).epsilon(1.0e-4f));
}

inline void hash_mix(uint64_t & hash, const uint64_t value) noexcept {
  hash ^= value;
  hash *= k_fnv_prime;
}

inline uint64_t hash_bytes(std::span<const uint8_t> bytes) noexcept {
  uint64_t hash = k_fnv_offset_basis;
  for (const uint8_t byte : bytes) {
    hash_mix(hash, byte);
  }
  return hash;
}

inline uint64_t hash_tensor_metadata(const emel::model::data & model) noexcept {
  uint64_t hash = k_fnv_offset_basis;
  for (uint32_t index = 0u; index < model.n_tensors; ++index) {
    const auto & tensor = model.tensors[index];
    hash_mix(hash, static_cast<uint64_t>(tensor.data_size));
    hash_mix(hash, static_cast<uint64_t>(tensor.name_offset));
    hash_mix(hash, static_cast<uint64_t>(tensor.name_length));
    hash_mix(hash, static_cast<uint64_t>(tensor.type));
    hash_mix(hash, static_cast<uint64_t>(tensor.n_dims));
    hash_mix(hash, static_cast<uint64_t>(tensor.data_offset));
    hash_mix(hash, static_cast<uint64_t>(tensor.file_offset));
    hash_mix(hash, static_cast<uint64_t>(tensor.file_index));
    hash_mix(hash, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tensor.data)));
    for (uint32_t dim = 0u; dim < tensor.dims.size(); ++dim) {
      hash_mix(hash, tensor.dims[dim]);
    }
  }
  return hash;
}

inline void warm_generator_sm_introspection_paths() {
  using initializer_machine = boost::sml::sm<emel::generator::initializer::model>;
  using initializer_states = typename initializer_machine::states;
  using initializer_transitions = typename initializer_machine::transitions;
  using prefill_machine = boost::sml::sm<emel::generator::prefill::model>;
  using prefill_states = typename prefill_machine::states;
  using prefill_transitions = typename prefill_machine::transitions;

  (void) emel::detail::type_list_contains<
      emel::generator::initializer::preparing_backend_decision,
      initializer_states>::value;
  (void) emel::detail::type_list_contains<
      emel::generator::initializer::binding_conditioner,
      initializer_states>::value;
  (void) emel::detail::type_list_contains<
      emel::generator::initializer::binding_conditioner_decision,
      initializer_states>::value;
  (void) emel::detail::type_list_contains<
      emel::generator::initializer::reserving_graph_decision,
      initializer_states>::value;
  (void) emel::detail::type_list_contains<
      emel::generator::initializer::configuring_sampler_decision,
      initializer_states>::value;
  (void) emel::detail::type_list_contains<
      emel::generator::prefill::contract_flash_decision,
      prefill_states>::value;
  (void) emel::detail::type_list_contains<
      emel::generator::prefill::contract_nonflash_decision,
      prefill_states>::value;
  (void) emel::detail::type_list_contains<
      emel::generator::prefill::compute_result_decision,
      prefill_states>::value;

  for_each_type(initializer_transitions{}, [&]<class transition_t>() {
    using event = typename transition_t::event;
    (void) emel::docs::detail::table_event_name<event>();
  });
  for_each_type(initializer_transitions{}, [&]<class transition_t>() {
    using src_state = typename transition_t::src_state;
    using dst_state = typename transition_t::dst_state;
    using event = typename transition_t::event;
    (void) emel::docs::detail::shorten_type_name(
        emel::docs::detail::raw_type_name<src_state>());
    (void) emel::docs::detail::shorten_type_name(
        emel::docs::detail::raw_type_name<dst_state>());
    (void) emel::docs::detail::table_event_name<event>();
  });
  for_each_type(prefill_transitions{}, [&]<class transition_t>() {
    using event = typename transition_t::event;
    (void) emel::docs::detail::table_event_name<event>();
  });
}

inline void exercise_shared_truncation_contract(const emel::model::data & model) {
  emel::text::tokenizer::sm tokenizer{};
  emel::text::conditioner::sm conditioner{};
  emel::embeddings::generator::sm embedding_generator{
    model,
    conditioner,
    nullptr,
    emel::text::formatter::format_raw,
  };

  emel::error::type initialize_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  initialize_embedding_generator(embedding_generator, initialize_error, tokenizer);

  const std::string red_square = read_text_file(te_prompt_path("red-square.txt"));
  const std::array text_messages = {
    emel::text::formatter::chat_message{.role = "user", .content = red_square},
  };
  const auto image = make_rgba_square(255u, 0u, 0u, 32, 32);
  const auto audio = make_sine_wave(440.0f);
  const std::array<int32_t, 5> supported_dimensions = {1280, 768, 512, 256, 128};

  for (const int32_t dimension : supported_dimensions) {
    std::array<float, 1280> text_output = {};
    std::array<float, 1280> image_output = {};
    std::array<float, 1280> audio_output = {};
    int32_t text_dimension = -1;
    int32_t image_dimension = -1;
    int32_t audio_dimension = -1;
    emel::error::type text_error =
        emel::error::cast(emel::embeddings::generator::error::none);
    emel::error::type image_error =
        emel::error::cast(emel::embeddings::generator::error::none);
    emel::error::type audio_error =
        emel::error::cast(emel::embeddings::generator::error::none);

    emel::embeddings::generator::event::embed_text text_request{
      text_messages,
      std::span<float>{text_output.data(), static_cast<size_t>(dimension)},
      text_dimension,
    };
    text_request.truncate_dimension = dimension == 1280 ? 0 : dimension;
    text_request.error_out = &text_error;
    REQUIRE(embedding_generator.process_event(text_request));
    CHECK(text_error == emel::error::cast(emel::embeddings::generator::error::none));
    CHECK(text_dimension == dimension);
    CHECK(embedding_generator.is(boost::sml::state<emel::embeddings::generator::state_done>));
    check_normalized(text_output, text_dimension);

    emel::embeddings::generator::event::embed_image image_request{
      image,
      32,
      32,
      std::span<float>{image_output.data(), static_cast<size_t>(dimension)},
      image_dimension,
    };
    image_request.truncate_dimension = dimension == 1280 ? 0 : dimension;
    image_request.error_out = &image_error;
    REQUIRE(embedding_generator.process_event(image_request));
    CHECK(image_error == emel::error::cast(emel::embeddings::generator::error::none));
    CHECK(image_dimension == dimension);
    CHECK(embedding_generator.is(boost::sml::state<emel::embeddings::generator::state_done>));
    check_normalized(image_output, image_dimension);

    emel::embeddings::generator::event::embed_audio audio_request{
      audio,
      k_audio_sample_rate,
      std::span<float>{audio_output.data(), static_cast<size_t>(dimension)},
      audio_dimension,
    };
    audio_request.truncate_dimension = dimension == 1280 ? 0 : dimension;
    audio_request.error_out = &audio_error;
    REQUIRE(embedding_generator.process_event(audio_request));
    CHECK(audio_error == emel::error::cast(emel::embeddings::generator::error::none));
    CHECK(audio_dimension == dimension);
    CHECK(embedding_generator.is(boost::sml::state<emel::embeddings::generator::state_done>));
    check_normalized(audio_output, audio_dimension);
  }
}

inline void exercise_audio_embedding_norm(const emel::model::data & model) {
  emel::text::tokenizer::sm tokenizer{};
  emel::text::conditioner::sm conditioner{};
  emel::embeddings::generator::sm embedding_generator{
    model,
    conditioner,
    nullptr,
    emel::text::formatter::format_raw,
  };

  emel::error::type initialize_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  initialize_embedding_generator(embedding_generator, initialize_error, tokenizer);

  const auto tone_440 = make_sine_wave(440.0f);
  const auto tone_880 = make_sine_wave(880.0f);
  std::array<float, 1280> embedding_440 = {};
  std::array<float, 1280> embedding_880 = {};
  int32_t dimension_440 = -1;
  int32_t dimension_880 = -1;
  emel::error::type error_440 =
      emel::error::cast(emel::embeddings::generator::error::none);
  emel::error::type error_880 =
      emel::error::cast(emel::embeddings::generator::error::none);

  emel::embeddings::generator::event::embed_audio request_440{
    tone_440,
    k_audio_sample_rate,
    embedding_440,
    dimension_440,
  };
  request_440.error_out = &error_440;
  REQUIRE(embedding_generator.process_event(request_440));
  CHECK(error_440 == emel::error::cast(emel::embeddings::generator::error::none));
  CHECK(dimension_440 == 1280);
  check_normalized(embedding_440, dimension_440);

  emel::embeddings::generator::event::embed_audio request_880{
    tone_880,
    k_audio_sample_rate,
    embedding_880,
    dimension_880,
  };
  request_880.error_out = &error_880;
  REQUIRE(embedding_generator.process_event(request_880));
  CHECK(error_880 == emel::error::cast(emel::embeddings::generator::error::none));
  CHECK(dimension_880 == 1280);
  check_normalized(embedding_880, dimension_880);
  CHECK(max_abs_difference(
            std::span<const float>{embedding_440.data(), static_cast<size_t>(dimension_440)},
            std::span<const float>{embedding_880.data(), static_cast<size_t>(dimension_880)}) >
        1.0e-5f);
}

}  // namespace

TEST_CASE("embeddings shared contract supports the same truncation surface across modalities") {
  if (!te_assets_present()) {
    MESSAGE("skipping TE shared-contract truncation test because maintained assets are not present");
    return;
  }

  const auto & fixture = cached_te_fixture();
  exercise_shared_truncation_contract(*fixture.model);
}

TEST_CASE("embeddings cached TE fixture survives generator docs warmup and shared reuse") {
  if (!te_assets_present()) {
    MESSAGE("skipping TE cached-fixture warmup regression because maintained assets are not present");
    return;
  }

  warm_generator_sm_introspection_paths();

  const auto & fixture = cached_te_fixture();
  const uint64_t file_hash_before =
      hash_bytes(std::span<const uint8_t>{fixture.file_bytes.data(), fixture.file_bytes.size()});
  const uint64_t tensor_hash_before = hash_tensor_metadata(*fixture.model);

  exercise_shared_truncation_contract(*fixture.model);

  CHECK(hash_bytes(std::span<const uint8_t>{fixture.file_bytes.data(), fixture.file_bytes.size()}) ==
        file_hash_before);
  CHECK(hash_tensor_metadata(*fixture.model) == tensor_hash_before);

  exercise_audio_embedding_norm(*fixture.model);
}

TEST_CASE("embeddings shared contract rejects unsupported truncation uniformly across modalities") {
  if (!te_assets_present()) {
    MESSAGE("skipping TE shared-contract rejection test because maintained assets are not present");
    return;
  }

  const auto & fixture = cached_te_fixture();
  emel::text::tokenizer::sm tokenizer{};
  emel::text::conditioner::sm conditioner{};
  emel::embeddings::generator::sm embedding_generator{
    *fixture.model,
    conditioner,
    nullptr,
    emel::text::formatter::format_raw,
  };

  emel::error::type initialize_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  initialize_embedding_generator(embedding_generator, initialize_error, tokenizer);

  const std::string red_square = read_text_file(te_prompt_path("red-square.txt"));
  const std::array text_messages = {
    emel::text::formatter::chat_message{.role = "user", .content = red_square},
  };
  const auto image = make_rgba_square(255u, 0u, 0u, 32, 32);
  const auto audio = make_sine_wave(440.0f);

  std::array<float, 1280> text_output = {};
  std::array<float, 1280> image_output = {};
  std::array<float, 1280> audio_output = {};
  int32_t text_dimension = -1;
  int32_t image_dimension = -1;
  int32_t audio_dimension = -1;
  emel::error::type text_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  emel::error::type image_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  emel::error::type audio_error =
      emel::error::cast(emel::embeddings::generator::error::none);

  emel::embeddings::generator::event::embed_text text_request{
    text_messages,
    text_output,
    text_dimension,
  };
  text_request.truncate_dimension = 640;
  text_request.error_out = &text_error;
  CHECK_FALSE(embedding_generator.process_event(text_request));
  CHECK(text_error == emel::error::cast(emel::embeddings::generator::error::invalid_request));
  CHECK(text_dimension == 0);
  CHECK(embedding_generator.is(boost::sml::state<emel::embeddings::generator::state_errored>));

  emel::embeddings::generator::event::embed_image image_request{
    image,
    32,
    32,
    image_output,
    image_dimension,
  };
  image_request.truncate_dimension = 640;
  image_request.error_out = &image_error;
  CHECK_FALSE(embedding_generator.process_event(image_request));
  CHECK(image_error == emel::error::cast(emel::embeddings::generator::error::invalid_request));
  CHECK(image_dimension == 0);
  CHECK(embedding_generator.is(boost::sml::state<emel::embeddings::generator::state_errored>));

  emel::embeddings::generator::event::embed_audio audio_request{
    audio,
    k_audio_sample_rate,
    audio_output,
    audio_dimension,
  };
  audio_request.truncate_dimension = 640;
  audio_request.error_out = &audio_error;
  CHECK_FALSE(embedding_generator.process_event(audio_request));
  CHECK(audio_error == emel::error::cast(emel::embeddings::generator::error::invalid_request));
  CHECK(audio_dimension == 0);
  CHECK(embedding_generator.is(boost::sml::state<emel::embeddings::generator::state_errored>));
}

TEST_CASE("embeddings initialize rejects omniembed fixtures missing a required modality family") {
  if (!te_assets_present()) {
    MESSAGE("skipping TE shared-contract validation test because maintained assets are not present");
    return;
  }

  auto fixture = load_te_fixture();
  bool removed_audio_projection = false;
  for (uint32_t index = 0; index < fixture.model->n_tensors; ++index) {
    auto & tensor = fixture.model->tensors[index];
    const auto name = emel::model::tensor_name_view(*fixture.model, tensor);
    if (!name.starts_with("audio_projection.")) {
      continue;
    }
    tensor.data = nullptr;
    tensor.data_size = 0u;
    removed_audio_projection = true;
  }
  REQUIRE(removed_audio_projection);

  emel::text::tokenizer::sm tokenizer{};
  emel::text::conditioner::sm conditioner{};
  emel::embeddings::generator::sm embedding_generator{
    *fixture.model,
    conditioner,
    nullptr,
    emel::text::formatter::format_raw,
  };

  emel::error::type initialize_error =
      emel::error::cast(emel::embeddings::generator::error::none);
  emel::embeddings::generator::event::initialize initialize{
    &tokenizer,
    te_fixture::tokenizer_bind_dispatch,
    te_fixture::tokenizer_tokenize_dispatch,
  };
  initialize.preprocessor_variant =
      emel::text::tokenizer::preprocessor::preprocessor_kind::wpm;
  initialize.encoder_variant = emel::text::encoders::encoder_kind::wpm;
  initialize.error_out = &initialize_error;

  CHECK_FALSE(embedding_generator.process_event(initialize));
  CHECK(initialize_error ==
      emel::error::cast(emel::embeddings::generator::error::model_invalid));
}
