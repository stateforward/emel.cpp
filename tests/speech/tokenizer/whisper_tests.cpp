#include "doctest/doctest.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <string>

#include "emel/error/error.hpp"
#include "emel/speech/tokenizer/any.hpp"
#include "emel/speech/tokenizer/whisper/any.hpp"
#include "emel/speech/tokenizer/whisper/detail.hpp"
#include "emel/speech/tokenizer/whisper/events.hpp"
#include "emel/speech/tokenizer/whisper/guards.hpp"
#include "emel/speech/tokenizer/whisper/sm.hpp"

namespace {

std::filesystem::path repo_root() {
#ifdef EMEL_TEST_REPO_ROOT
  return std::filesystem::path{EMEL_TEST_REPO_ROOT};
#else
  return std::filesystem::current_path();
#endif
}

std::string read_text_file(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  REQUIRE(stream.good());
  return std::string{std::istreambuf_iterator<char>{stream},
                     std::istreambuf_iterator<char>{}};
}

} // namespace

TEST_CASE("speech whisper tokenizer asset provides required control roles") {
  const auto tokenizer_path =
      repo_root() / "tests" / "models" / "tokenizer-tiny.json";
  REQUIRE(std::filesystem::exists(tokenizer_path));

  const std::string tokenizer_json = read_text_file(tokenizer_path);
  CHECK(emel::speech::tokenizer::whisper::detail::validate_tiny_control_tokens(
      tokenizer_json));

  const auto tokens =
      emel::speech::tokenizer::whisper::detail::k_tiny_control_tokens;
  CHECK(tokens.eot == 50257);
  CHECK(tokens.sot == 50258);
  CHECK(tokens.language_en == 50259);
  CHECK(tokens.translate == 50358);
  CHECK(tokens.transcribe == 50359);
  CHECK(tokens.no_speech == 50362);
  CHECK(tokens.notimestamps == 50363);
  CHECK(tokens.timestamp_begin == 50364);
  CHECK(tokens.space == 220);

  const auto policy =
      emel::speech::tokenizer::whisper::detail::k_tiny_asr_decode_policy;
  CHECK(policy.language ==
        emel::speech::tokenizer::whisper::detail::language_role::english);
  CHECK(policy.task ==
        emel::speech::tokenizer::whisper::detail::task_role::transcribe);
  CHECK(
      policy.timestamps ==
      emel::speech::tokenizer::whisper::detail::timestamp_mode::timestamp_tokens);
  CHECK(policy.suppress_translate);
  CHECK(policy.prompt_tokens[0] == tokens.sot);
  CHECK(policy.prompt_tokens[1] == tokens.language_en);
  CHECK(policy.prompt_tokens[2] == tokens.transcribe);

  int32_t token_ids[] = {50364, 542, 33, 898, 60, 50414};
  char transcript[16] = {};
  const uint64_t transcript_size =
      emel::speech::tokenizer::whisper::detail::decode_token_ids(
          tokenizer_json, std::span<const int32_t>{token_ids}, transcript,
          sizeof(transcript));
  CHECK(std::string_view{transcript, static_cast<size_t>(transcript_size)} ==
        "[Bell]");
}

TEST_CASE("speech whisper tokenizer policy support rejects drifted fields") {
  namespace whisper = emel::speech::tokenizer::whisper;

  auto policy = whisper::tiny_asr_decode_policy();
  CHECK(whisper::is_tiny_asr_decode_policy_supported(policy));

  policy = whisper::tiny_asr_decode_policy();
  policy.language = static_cast<whisper::language_role>(99u);
  CHECK_FALSE(whisper::is_tiny_asr_decode_policy_supported(policy));

  policy = whisper::tiny_asr_decode_policy();
  policy.task = static_cast<whisper::task_role>(99u);
  CHECK_FALSE(whisper::is_tiny_asr_decode_policy_supported(policy));

  policy = whisper::tiny_asr_decode_policy();
  policy.suppress_translate = false;
  CHECK_FALSE(whisper::is_tiny_asr_decode_policy_supported(policy));

  policy = whisper::tiny_asr_decode_policy();
  policy.tokens.eot = 0;
  CHECK_FALSE(whisper::is_tiny_asr_decode_policy_supported(policy));

  policy = whisper::tiny_asr_decode_policy();
  policy.tokens.translate = 0;
  CHECK_FALSE(whisper::is_tiny_asr_decode_policy_supported(policy));

  policy = whisper::tiny_asr_decode_policy();
  policy.tokens.no_speech = 0;
  CHECK_FALSE(whisper::is_tiny_asr_decode_policy_supported(policy));

  policy = whisper::tiny_asr_decode_policy();
  policy.tokens.timestamp_begin = 0;
  CHECK_FALSE(whisper::is_tiny_asr_decode_policy_supported(policy));

  policy = whisper::tiny_asr_decode_policy();
  policy.tokens.space = 0;
  CHECK_FALSE(whisper::is_tiny_asr_decode_policy_supported(policy));

  policy = whisper::tiny_asr_decode_policy();
  policy.prompt_tokens[0] = 0;
  CHECK_FALSE(whisper::is_tiny_asr_decode_policy_supported(policy));

  policy = whisper::tiny_asr_decode_policy();
  policy.prompt_tokens[1] = 0;
  CHECK_FALSE(whisper::is_tiny_asr_decode_policy_supported(policy));

  policy = whisper::tiny_asr_decode_policy();
  policy.prompt_tokens[2] = 0;
  CHECK_FALSE(whisper::is_tiny_asr_decode_policy_supported(policy));
}

TEST_CASE("speech whisper tokenizer rejects malformed token metadata") {
  namespace whisper = emel::speech::tokenizer::whisper::detail;

  char negative_id[8] = {};
  const uint32_t negative_id_size = whisper::write_i32(-42, negative_id);
  CHECK(std::string_view{negative_id, negative_id_size} == "-42");

  CHECK_FALSE(whisper::validate_tiny_control_tokens("{}"));
  CHECK_FALSE(whisper::contains_token_role("{}", "<|endoftext|>",
                                           whisper::k_tiny_control_tokens.eot));

  std::string_view token_text = {};
  CHECK_FALSE(whisper::find_vocab_token_text("{\"A\": 33x}", 33, token_text));
  CHECK_FALSE(whisper::find_vocab_token_text(":33,\n", 33, token_text));
  CHECK_FALSE(whisper::find_vocab_token_text("x\" :33,\n", 33, token_text));
  CHECK_FALSE(whisper::find_vocab_token_text("{\"A\": 34,\n}", 33,
                                             token_text));
}

TEST_CASE("speech whisper tokenizer restricts id lookup to vocab entries") {
  namespace whisper = emel::speech::tokenizer::whisper::detail;

  const std::string tokenizer_json =
      "{\"added_tokens\":[{\"content\":\"metadata\",\"type_id\":0,\"id\":1}],"
      "\"model\":{\"vocab\":{\"!\":0,\"real\":1}}}";

  CHECK(whisper::find_max_vocab_token_bytes(tokenizer_json) == 4u);
  CHECK(whisper::required_transcript_capacity(tokenizer_json, 2u) == 8u);

  std::string_view token_text = {};
  REQUIRE(whisper::find_vocab_token_text(tokenizer_json, 0, token_text));
  CHECK(token_text == "!");

  token_text = {};
  REQUIRE(whisper::find_vocab_token_text(tokenizer_json, 1, token_text));
  CHECK(token_text == "real");

  int32_t token_ids[] = {0, 1};
  char transcript[16] = {};
  const uint64_t transcript_size = whisper::decode_token_ids(
      tokenizer_json, std::span<const int32_t>{token_ids}, transcript,
      sizeof(transcript));
  CHECK(std::string_view{transcript, static_cast<size_t>(transcript_size)} ==
        "!real");
}

TEST_CASE("speech whisper tokenizer decodes escaped vocab token text") {
  namespace whisper = emel::speech::tokenizer::whisper::detail;

  const std::string tokenizer_json =
      R"({"model":{"vocab":{"\"":1,"\\":2,"\/":3,)"
      R"("\b":4,"\f":5,"\n":6,"\r":7,"\t":8,"plain":9}}})";
  int32_t token_ids[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  char transcript[24] = {};
  const uint64_t transcript_size = whisper::decode_token_ids(
      tokenizer_json, std::span<const int32_t>{token_ids}, transcript,
      sizeof(transcript));

  CHECK(std::string_view{transcript, static_cast<size_t>(transcript_size)} ==
        "\"\\/\b\f\n\r\tplain");
}

TEST_CASE("speech whisper tokenizer trims leading spaces and respects capacity") {
  namespace whisper = emel::speech::tokenizer::whisper::detail;

  const std::string tokenizer_json =
      "{\"ĠA\": 33,\n\"Bell\": 898,\n\"!\": 0,\n}";
  int32_t token_ids[] = {
      whisper::k_tiny_control_tokens.sot,
      33,
      898,
      0,
      whisper::k_tiny_control_tokens.eot,
  };
  char transcript[6] = {};
  const uint64_t transcript_size = whisper::decode_token_ids(
      tokenizer_json, std::span<const int32_t>{token_ids}, transcript,
      sizeof(transcript));

  CHECK(transcript_size == 5u);
  CHECK(std::string_view{transcript, static_cast<size_t>(transcript_size)} ==
        "ABell");
}

TEST_CASE("speech whisper tokenizer guard rejects malformed detokenize spans") {
  namespace whisper = emel::speech::tokenizer::whisper;

  const auto tokenizer_path =
      repo_root() / "tests" / "models" / "tokenizer-tiny.json";
  REQUIRE(std::filesystem::exists(tokenizer_path));
  const std::string tokenizer_json = read_text_file(tokenizer_path);

  int32_t token_ids[] = {whisper::detail::k_tiny_control_tokens.timestamp_begin};
  char transcript[16] = {};
  int32_t transcript_size = 0;

  // A well-formed request with valid JSON and valid spans passes the guard.
  whisper::event::detokenize good_request{
      tokenizer_json, std::span<const int32_t>{token_ids},
      std::span<char>{transcript}, transcript_size};
  whisper::event::detokenize_ctx good_ctx{};
  whisper::event::detokenize_run good_run{good_request, good_ctx};
  CHECK(whisper::guard::guard_tokenizer_json_valid{}(good_run));
  CHECK_FALSE(whisper::guard::guard_tokenizer_json_invalid{}(good_run));

  // A non-empty transcript span with a null data pointer must be rejected so the
  // decode action never writes through the null pointer.
  whisper::event::detokenize null_transcript_request{
      tokenizer_json, std::span<const int32_t>{token_ids},
      std::span<char>{static_cast<char *>(nullptr), 8u}, transcript_size};
  whisper::event::detokenize_ctx null_transcript_ctx{};
  whisper::event::detokenize_run null_transcript_run{null_transcript_request,
                                                     null_transcript_ctx};
  CHECK_FALSE(whisper::guard::guard_tokenizer_json_valid{}(null_transcript_run));
  CHECK(whisper::guard::guard_tokenizer_json_invalid{}(null_transcript_run));

  // A non-empty token span with a null data pointer must be rejected so the
  // decode action never iterates the null span.
  whisper::event::detokenize null_tokens_request{
      tokenizer_json,
      std::span<const int32_t>{static_cast<const int32_t *>(nullptr), 4u},
      std::span<char>{transcript}, transcript_size};
  whisper::event::detokenize_ctx null_tokens_ctx{};
  whisper::event::detokenize_run null_tokens_run{null_tokens_request,
                                                 null_tokens_ctx};
  CHECK_FALSE(whisper::guard::guard_tokenizer_json_valid{}(null_tokens_run));
  CHECK(whisper::guard::guard_tokenizer_json_invalid{}(null_tokens_run));
}

TEST_CASE("speech whisper tokenizer clears stale transcript size on rejection") {
  namespace tokenizer = emel::speech::tokenizer;

  const auto tokenizer_path =
      repo_root() / "tests" / "models" / "tokenizer-tiny.json";
  REQUIRE(std::filesystem::exists(tokenizer_path));
  const std::string tokenizer_json = read_text_file(tokenizer_path);

  tokenizer::whisper::sm machine{};
  int32_t token_ids[] = {542, 33, 898, 60};
  char transcript[16] = {};
  int32_t transcript_size = -1;
  emel::error::type err =
      emel::error::cast(tokenizer::whisper::error::internal_error);

  tokenizer::event::detokenize good_request{
      tokenizer_json, std::span<const int32_t>{token_ids},
      std::span<char>{transcript}, transcript_size};
  good_request.error_out = &err;
  REQUIRE(machine.process_event(good_request));
  CHECK(err == emel::error::cast(tokenizer::whisper::error::none));
  REQUIRE(transcript_size > 0);

  // A rejected request must clear the caller's out-parameter instead of
  // leaving the previous dispatch's positive size next to a failure result.
  tokenizer::event::detokenize bad_request{"{}",
                                           std::span<const int32_t>{token_ids},
                                           std::span<char>{transcript},
                                           transcript_size};
  bad_request.error_out = &err;
  CHECK_FALSE(machine.process_event(bad_request));
  CHECK(err ==
        emel::error::cast(tokenizer::whisper::error::tokenizer_json_invalid));
  CHECK(transcript_size == 0);
}

TEST_CASE("speech tokenizer facade preserves unsupported kinds") {
  namespace tokenizer = emel::speech::tokenizer;

  static_assert(
      tokenizer::any::is_supported(tokenizer::tokenizer_kind::whisper));
  static_assert(
      !tokenizer::any::is_supported(tokenizer::tokenizer_kind::unsupported));
  static_assert(
      !tokenizer::any::is_supported(static_cast<tokenizer::tokenizer_kind>(42)));

  // kind() reports the requested value so owners can reject dispatch to an
  // unsupported facade instead of silently running the default variant.
  tokenizer::any unsupported_facade{tokenizer::tokenizer_kind::unsupported};
  CHECK(unsupported_facade.kind() == tokenizer::tokenizer_kind::unsupported);
  CHECK_FALSE(tokenizer::any::is_supported(unsupported_facade.kind()));

  tokenizer::any cast_facade{static_cast<tokenizer::tokenizer_kind>(42)};
  CHECK(cast_facade.kind() == static_cast<tokenizer::tokenizer_kind>(42));
  CHECK_FALSE(tokenizer::any::is_supported(cast_facade.kind()));

  cast_facade.set_kind(tokenizer::tokenizer_kind::whisper);
  CHECK(cast_facade.kind() == tokenizer::tokenizer_kind::whisper);
  CHECK(tokenizer::any::is_supported(cast_facade.kind()));
}

TEST_CASE("speech whisper tokenizer clamps compacted output to capacity") {
  namespace whisper = emel::speech::tokenizer::whisper::detail;

  const std::string tokenizer_json =
      "{\"ĠLong\": 33,\n\"Transcript\": 34,\n\"Tail\": 35,\n}";
  int32_t token_ids[] = {
      33,
      34,
      35,
  };
  char transcript[4] = {};
  const uint64_t transcript_size = whisper::decode_token_ids(
      tokenizer_json, std::span<const int32_t>{token_ids}, transcript,
      sizeof(transcript));

  CHECK(transcript_size <= sizeof(transcript));
  CHECK(transcript_size == 3u);
  CHECK(std::string_view{transcript, static_cast<size_t>(transcript_size)} ==
        "Lon");
}
