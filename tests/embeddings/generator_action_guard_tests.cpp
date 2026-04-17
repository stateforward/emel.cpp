#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

#include "doctest/doctest.h"

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

}  // namespace

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
