#include "doctest/doctest.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <string>

#include "emel/speech/tokenizer/whisper/any.hpp"
#include "emel/speech/tokenizer/whisper/detail.hpp"

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
