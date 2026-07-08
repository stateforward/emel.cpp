#include <array>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "doctest/doctest.h"

#include "emel/embeddings/generator/detail.hpp"
#include "emel/embeddings/generator/sm.hpp"
#include "emel/machines.hpp"

namespace {

std::filesystem::path repo_root() {
  return std::filesystem::path{__FILE__}.parent_path().parent_path().parent_path();
}

std::string read_text_file(const std::filesystem::path & path) {
  std::ifstream stream(path);
  REQUIRE_MESSAGE(stream.good(), "failed to open source file: " << path.string());
  return std::string{
      std::istreambuf_iterator<char>{stream},
      std::istreambuf_iterator<char>{},
  };
}

void check_absent(const std::string & content,
                  const std::string_view needle,
                  const std::string_view label) {
  CHECK_MESSAGE(content.find(needle) == std::string::npos, label);
}

void check_present(const std::string & content,
                   const std::string_view needle,
                   const std::string_view label) {
  CHECK_MESSAGE(content.find(needle) != std::string::npos, label);
}

std::vector<std::filesystem::path> generic_generator_surfaces() {
  std::vector<std::filesystem::path> paths;
  const std::filesystem::path dir =
      repo_root() / "src" / "emel" / "embeddings" / "generator";

  for (const auto & entry : std::filesystem::directory_iterator{dir}) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const auto extension = entry.path().extension();
    if (extension == ".hpp" || extension == ".cpp") {
      paths.push_back(entry.path());
    }
  }

  return paths;
}

}  // namespace

TEST_CASE("embedding generator keeps component-level sm compatibility type") {
  static_assert(std::is_same_v<emel::embeddings::generator::sm,
                               emel::embeddings::generator::basic_sm<
                                   emel::embeddings::generator::route>>);
  static_assert(std::is_same_v<emel::EmbeddingsGenerator,
                               emel::embeddings::generator::sm>);
  static_assert(std::is_same_v<emel::OmniEmbedEmbeddingsGenerator,
                               emel::embeddings::generator::omniembed::sm>);
  emel::embeddings::generator::sm generator;
  CHECK(generator.is(
      stateforward::sml::state<emel::embeddings::generator::state_uninitialized>));
}

TEST_CASE("embedding generator no longer hides phase outcome latches in actions and guards") {
  const std::string actions = read_text_file(
      repo_root() / "src" / "emel" / "embeddings" / "generator" / "actions.hpp");
  const std::string guards = read_text_file(
      repo_root() / "src" / "emel" / "embeddings" / "generator" / "guards.hpp");
  const std::string events = read_text_file(
      repo_root() / "src" / "emel" / "embeddings" / "generator" / "events.hpp");
  const std::string detail = read_text_file(
      repo_root() / "src" / "emel" / "embeddings" / "generator" / "detail.hpp");
  const std::string sm = read_text_file(
      repo_root() / "src" / "emel" / "embeddings" / "generator" / "sm.hpp");

  check_absent(actions,
               "ev.ctx.prepare_result =",
               "image prepare action still hides outcome behind helper return");
  check_absent(actions,
               "ev.ctx.prepare_result =",
               "audio prepare action still hides outcome behind helper return");
  check_absent(actions,
               "ev.ctx.encode_result =",
               "text encode action still hides outcome behind helper return");
  check_absent(actions,
               "ev.ctx.encode_result =",
               "image encode action still hides outcome behind helper return");
  check_absent(actions,
               "ev.ctx.encode_result =",
               "audio encode action still hides outcome behind helper return");

  check_absent(guards,
               "prepare_result",
               "prepare guards still route on a latched phase result");
  check_absent(guards,
               "encode_result",
               "encode guards still route on a latched phase result");

  check_absent(events,
               "phase_result_kind",
               "runtime events still expose phase result latches");
  check_absent(events,
               "prepare_result",
               "runtime events still expose prepare result latches");
  check_absent(events,
               "encode_result",
               "runtime events still expose encode result latches");

  check_absent(detail,
               "phase_result_from_success",
               "detail still uses bool-to-phase-result routing helpers");
  check_absent(actions,
               "detail::prepare_image_input_mobilenetv4_error",
               "image prepare action still routes through a detail error wrapper");
  check_absent(actions,
               "detail::prepare_audio_input_efficientat_error",
               "audio prepare action still routes through a detail error wrapper");
  check_absent(actions,
               "detail::run_text_embedding_bert_error",
               "text encode action still routes through a detail error wrapper");
  check_absent(actions,
               "detail::run_image_embedding_mobilenetv4_error",
               "image encode action still routes through a detail error wrapper");
  check_absent(actions,
               "detail::run_audio_embedding_efficientat_error",
               "audio encode action still routes through a detail error wrapper");
  check_absent(detail,
               "prepare_image_input_mobilenetv4_error",
               "detail still exposes image prepare error routing wrapper");
  check_absent(detail,
               "prepare_audio_input_efficientat_error",
               "detail still exposes audio prepare error routing wrapper");
  check_absent(detail,
               "run_text_embedding_bert_error",
               "detail still exposes text encode error routing wrapper");
  check_absent(detail,
               "run_image_embedding_mobilenetv4_error",
               "detail still exposes image encode error routing wrapper");
  check_absent(detail,
               "run_audio_embedding_efficientat_error",
               "detail still exposes audio encode error routing wrapper");
  check_absent(sm,
               "guard::guard_image_prepare_success",
               "state machine still routes image prepare via a post-action success latch");
  check_absent(sm,
               "guard::guard_audio_prepare_success",
               "state machine still routes audio prepare via a post-action success latch");
  check_absent(actions,
               "(void) detail::run_text_embedding(",
               "text encode action still discards the embedding kernel result");
  check_absent(actions,
               "(void) detail::run_image_embedding(",
               "image encode action still discards the embedding kernel result");
  check_absent(actions,
               "(void) detail::run_audio_embedding(",
               "audio encode action still discards the embedding kernel result");
  check_present(sm,
                "guard::guard_embedding_failed",
                "state machine no longer routes runtime embedding failures to error publication");
}

TEST_CASE("embedding generator generic surfaces do not expose omniembed contracts") {
  const auto surfaces = generic_generator_surfaces();
  REQUIRE_FALSE(surfaces.empty());

  for (const auto & surface : surfaces) {
    const std::string source = read_text_file(surface);
    check_absent(source, "omniembed", "generic embeddings generator surface exposes OmniEmbed");
    check_absent(source, "OmniEmbed", "generic embeddings generator surface exposes OmniEmbed");
    check_absent(source,
                 "model/omniembed",
                 "generic embeddings generator surface includes OmniEmbed model detail");
    check_absent(source,
                 "model::omniembed",
                 "generic embeddings generator surface stores OmniEmbed model detail");
  }
}

TEST_CASE("embedding generator orchestration surfaces do not expose family route names") {
  const auto surfaces = generic_generator_surfaces();
  REQUIRE_FALSE(surfaces.empty());

  for (const auto & surface : surfaces) {
    const std::string source = read_text_file(surface);
    check_absent(source, "bert", "generic embeddings generator route exposes BERT");
    check_absent(source, "BERT", "generic embeddings generator route exposes BERT");
    check_absent(source, "mobilenet", "generic embeddings generator route exposes MobileNet");
    check_absent(source, "MobileNet", "generic embeddings generator route exposes MobileNet");
    check_absent(source, "efficientat", "generic embeddings generator route exposes EfficientAT");
    check_absent(source, "EfficientAT", "generic embeddings generator route exposes EfficientAT");
    check_absent(source,
                 "edge_residual",
                 "generic embeddings generator route exposes OmniEmbed vision block kinds");
    check_absent(source,
                 "universal_inverted",
                 "generic embeddings generator route exposes OmniEmbed vision block kinds");
    check_absent(source,
                 "audio_inverted",
                 "generic embeddings generator route exposes OmniEmbed audio block kinds");
    check_absent(source,
                 "has_dw",
                 "generic embeddings generator route exposes OmniEmbed block control flags");
    check_absent(source,
                 "has_expand",
                 "generic embeddings generator route exposes OmniEmbed block control flags");
    check_absent(source,
                 "use_hardswish",
                 "generic embeddings generator route exposes OmniEmbed activation controls");
    check_absent(source,
                 "k_max_blocks",
                 "generic embeddings generator route exposes OmniEmbed runtime sizing");
    check_absent(source,
                 "matrix_view",
                 "generic embeddings generator route exposes OmniEmbed tensor binding state");
    check_absent(source,
                 "conv2d_view",
                 "generic embeddings generator route exposes OmniEmbed tensor binding state");
    check_absent(source,
                 "batch_norm",
                 "generic embeddings generator route exposes OmniEmbed tensor binding state");
  }
}

TEST_CASE("embedding generator detail does not use model-family validation as a route gate") {
  const std::string source = read_text_file(
      repo_root() / "src" / "emel" / "embeddings" / "generator" / "detail.hpp");

  check_absent(source,
               "build_execution_contract",
               "generic embeddings generator detail exposes a contract-building route gate");
  check_absent(source,
               "validate_execution_contract",
               "generic embeddings generator detail dispatches through model-family validation");
}

TEST_CASE("embedding generator truncate dimensions stay inside the validated embedding size") {
  emel::embeddings::generator::action::context ctx = {};
  ctx.execution_contract.embedding_length = 1280;
  ctx.execution_contract.matryoshka_dimension_count = 2u;
  ctx.execution_contract.matryoshka_dimensions[0] = 1280;
  ctx.execution_contract.matryoshka_dimensions[1] = 512;

  CHECK(emel::embeddings::generator::detail::is_supported_truncate_dimension(ctx, 1280));
  CHECK(emel::embeddings::generator::detail::is_supported_truncate_dimension(ctx, 512));
  CHECK_FALSE(emel::embeddings::generator::detail::is_supported_truncate_dimension(ctx, 2048));

  ctx.execution_contract.matryoshka_dimensions[1] = 2048;
  CHECK_FALSE(emel::embeddings::generator::detail::is_supported_truncate_dimension(ctx, 2048));
}

TEST_CASE("embedding generator generic error writers publish the failed output dimension") {
  std::array<float, 1> output = {};

  int32_t text_dimension = 99;
  emel::embeddings::generator::event::embed_text text_request{
      std::span<const emel::text::formatter::chat_message>{},
      output,
      text_dimension,
  };
  emel::embeddings::generator::event::embed_text_ctx text_ctx{};
  text_ctx.output_dimension = 0;
  emel::embeddings::generator::detail::write_embed_error_out({text_request, text_ctx});
  CHECK(text_dimension == 0);

  int32_t image_dimension = 99;
  emel::embeddings::generator::event::embed_image image_request{
      std::span<const uint8_t>{},
      0,
      0,
      output,
      image_dimension,
  };
  emel::embeddings::generator::event::embed_image_ctx image_ctx{};
  image_ctx.output_dimension = 0;
  emel::embeddings::generator::detail::write_embed_error_out({image_request, image_ctx});
  CHECK(image_dimension == 0);

  int32_t audio_dimension = 99;
  emel::embeddings::generator::event::embed_audio audio_request{
      std::span<const float>{},
      0,
      output,
      audio_dimension,
  };
  emel::embeddings::generator::event::embed_audio_ctx audio_ctx{};
  audio_ctx.output_dimension = 0;
  emel::embeddings::generator::detail::write_embed_error_out({audio_request, audio_ctx});
  CHECK(audio_dimension == 0);
}
